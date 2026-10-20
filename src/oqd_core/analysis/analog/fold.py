# Copyright 2024-2025 Open Quantum Design

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

########################################################################################


from __future__ import annotations

import operator
from functools import reduce
from typing import Type, Union

from oqd_compiler_infrastructure import (
    CFGBlock,
    ForwardDataflowAnalysis,
    Lattice,
    LatticeTop,
    MapLatticeValue,
    Post,
    RewriteRule,
    gen_pass,
    maplattice,
)

from oqd_core.interface.analog import Access, Constant, Declaration, Neg, Not, Pos
from oqd_core.interface.analog.expr import BinaryOp, Operator, UnaryOp

########################################################################################


class ConstantFoldingError(Exception): ...


########################################################################################


class CLatticeTop(LatticeTop): ...


class CInvalid(CLatticeTop): ...


ConstantFoldingValue = Union[Type[CLatticeTop], Constant, Operator, BinaryOp, UnaryOp]

########################################################################################


class ConstantLattice(Lattice[ConstantFoldingValue]):
    def top(self):
        return CLatticeTop

    def bottom(self):
        return CInvalid

    def leq(self, t1, t2):
        if self.join(t1, t2) == t1:
            return True
        return False

    def join(self, t1, t2):
        if t1 == t2:
            return t1

        if t1 == self.bottom():
            return t2

        if t2 == self.bottom():
            return t1

        return self.top()

    def meet(self, t1, t2):
        if t1 == t2:
            return t1

        if t1 == self.top():
            return t2

        if t2 == self.top():
            return t1

        return self.bottom()


########################################################################################

FoldEnv = MapLatticeValue[ConstantFoldingValue]


class ConstantFolding(ForwardDataflowAnalysis[int, CFGBlock, FoldEnv]):
    lattice = maplattice(ConstantLattice)()

    def initial_state(self, nodes):
        return {
            node: self.lattice.bottom() if node == 0 else self.lattice.top()
            for node in nodes
        }

    def merge(self, states):
        return self.lattice.merge_intersection(states)

    @gen_pass(rule_type="conversion", walk=Post, method=True)
    def _fold(self, model, operands, *, env):
        match model:
            case list():
                return operands
            case Constant() | Operator():
                return model

            case BinaryOp() if all(
                expr != self.lattice.element_lattice.bottom()
                for expr in operands["exprs"]
            ):
                return EvaluateConstant()(model)

            case UnaryOp() if operands["expr"] != self.lattice.element_lattice.bottom():
                return EvaluateConstant()(model)

            case Access():
                return env.get(model.name, self.lattice.element_lattice.bottom())

            case Declaration():
                env[model.name] = operands["value"]

            case _:
                return self.lattice.element_lattice.bottom()

    def transfer(self, graph, node_id, state_in):
        block = graph[node_id]

        state_out = state_in.copy() if isinstance(state_in, dict) else {}

        self._fold(block, env=state_out)
        return state_out


########################################################################################


class EvaluateConstant(RewriteRule):
    def map_Add(self, model):
        exprs = []

        for expr in model.exprs:
            if exprs and isinstance(exprs[-1], Constant) and isinstance(expr, Constant):
                exprs[-1] = Constant(value=exprs[-1].value + expr.value)
                continue

            exprs.append(expr)

        return exprs[0] if len(exprs) == 1 else model.__class__(exprs=exprs)

    def map_Sub(self, model):
        exprs = []

        for expr in model.exprs:
            if exprs and isinstance(exprs[-1], Constant) and isinstance(expr, Constant):
                exprs[-1] = (
                    Constant(value=exprs[-1].value - expr.value)
                    if len(exprs) == 1
                    else Constant(value=exprs[-1].value + expr.value)
                )
                continue

            exprs.append(expr)

        return exprs[0] if len(exprs) == 1 else model.__class__(exprs=exprs)

    def map_Mul(self, model):
        exprs = []

        for expr in model.exprs:
            if exprs and isinstance(exprs[-1], Constant) and isinstance(expr, Constant):
                exprs[-1] = Constant(value=exprs[-1].value * expr.value)
                continue

            exprs.append(expr)

        return exprs[0] if len(exprs) == 1 else model.__class__(exprs=exprs)

    def map_Div(self, model):
        exprs = []

        for expr in model.exprs:
            if exprs and isinstance(exprs[-1], Constant) and isinstance(expr, Constant):
                exprs[-1] = (
                    Constant(value=exprs[-1].value / expr.value)
                    if len(exprs) == 1
                    else Constant(value=exprs[-1].value * expr.value)
                )
                continue

            exprs.append(expr)

        return exprs[0] if len(exprs) == 1 else model.__class__(exprs=exprs)

    def map_Pow(self, model):
        exprs = []
        stop = False

        for expr in reversed(model.exprs):
            if (
                exprs
                and not stop
                and isinstance(exprs[-1], Constant)
                and isinstance(expr, Constant)
            ):
                exprs[-1] = Constant(value=expr.value ** exprs[-1].value)
                continue

            if exprs:
                stop = True
            exprs.append(expr)

        return exprs[0] if len(exprs) == 1 else model.__class__(exprs=reversed(exprs))

    def map_UnaryOp(self, model):
        if isinstance(model.expr, Constant):
            return Constant(
                value={Pos: operator.pos, Neg: operator.neg, Not: operator.not_}[
                    model.__class__
                ](model.expr.value)
            )


class ConstantFoldingRule(EvaluateConstant, RewriteRule):
    def __init__(self, fold_res):
        super().__init__()

        self.fold_res = fold_res

    def map_CFGBlock(self, model):
        self._current_block = model.register_id

    def map_Access(self, model):
        if (
            value := self.fold_res.in_states[self._current_block].get(model.name, None)
        ) not in [CInvalid, None]:
            return value
