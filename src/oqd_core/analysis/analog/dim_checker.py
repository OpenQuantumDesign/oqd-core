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

from functools import reduce
from typing import List, Union

from oqd_compiler_infrastructure import (
    CFGBlock,
    ForwardDataflowAnalysis,
    LatticeBase,
    LatticeTop,
    maplattice,
)

from oqd_core.interface.analog import (
    Access,
    Add,
    AnalogList,
    Annihilation,
    Break,
    Constant,
    Continue,
    Creation,
    Declaration,
    Evolve,
    Extract,
    Identity,
    Kron,
    ModeRegister,
    Mul,
    Neg,
    PauliI,
    PauliX,
    PauliY,
    PauliZ,
    Pos,
    QuantumRegister,
    Sub,
)

########################################################################################


class DAny(LatticeTop): ...


class DInvalid(DAny): ...


DQuantum = List[Union[int, "DQuantum"]]

DLatticeValue = Union[DAny, DInvalid, DQuantum]


class DimensionLattice(LatticeBase[DLatticeValue]):
    """Quantum dimension lattice for analog layer."""

    def top(self):
        return DAny

    def bottom(self):
        return DInvalid

    def leq(self, t1: DLatticeValue, t2: DLatticeValue) -> bool:
        if self.join(t1, t2) == t1:
            return True
        return False

    def join(self, t1: DLatticeValue, t2: DLatticeValue) -> DLatticeValue:
        if t1 == t2:
            return t1

        if t1 == self.bottom():
            return t2

        if t2 == self.bottom():
            return t1

        return self.top()

    def meet(self, t1: DLatticeValue, t2: DLatticeValue) -> DLatticeValue:
        if t1 == t2:
            return t1

        if t1 == self.top():
            return t2

        if t2 == self.top():
            return t1

        return self.bottom()


########################################################################################


class DimensionError(Exception): ...


########################################################################################


class DimensionChecker(ForwardDataflowAnalysis[int, CFGBlock, DLatticeValue]):
    """Forward dataflow dimension checker over the Control Flow Graph."""

    lattice = maplattice(DimensionLattice, default_mode="top")()

    def _is_integer(self, expr):
        return isinstance(expr, Constant) and type(expr.value) is int

    def _infer_dim(self, expr, *, env):
        match expr:
            case Access():
                return (
                    self.lattice.element_lattice.top()
                    if env is self.lattice.top()
                    else env.get(expr.name, self.lattice.element_lattice.top())
                )

            case PauliX() | PauliY() | PauliZ() | PauliI() if self._is_integer(
                expr.dim
            ):
                return [expr.dim.value]

            case Annihilation() | Creation() | Identity():
                return [-1]

            case QuantumRegister() if self._is_integer(expr.size) and self._is_integer(
                expr.dim
            ):
                return [expr.dim.value] * expr.size.value

            case ModeRegister() if self._is_integer(expr.size):
                return [-1] * expr.size.value

            case QuantumRegister():
                raise DimensionError(
                    "Dimension checker only works for constant args for QuantumRegister"
                )

            case ModeRegister():
                raise DimensionError(
                    "Dimension checker only works for constant args for QuantumRegister"
                )

            case AnalogList():
                return [self._infer_dim(element, env=env) for element in expr.values]

            case Extract() if (
                isinstance(expr.index, Constant) and type(expr.index.value) is int
            ):
                value = self._infer_dim(expr.access, env=env)

                return (
                    value
                    if value == self.lattice.element_lattice.bottom()
                    else [value[expr.index.value]]
                )

            case Neg() | Pos():
                arg = self._infer_dim(expr.expr, env=env)
                return arg

            case Add() | Sub() | Mul():
                binop = {Add: "add", Sub: "subtract", Mul: "multiply"}[expr.__class__]

                args = [self._infer_dim(e, env=env) for e in expr.exprs]
                op_args = list(
                    filter(
                        lambda x: x is not self.lattice.element_lattice.bottom(),
                        args,
                    )
                )

                dim = reduce(
                    self.lattice.element_lattice.meet,
                    op_args,
                    self.lattice.element_lattice.top(),
                )

                if op_args and dim is self.lattice.element_lattice.bottom():
                    raise DimensionError(
                        f"attempted to {binop} operators of different dimensions {tuple(args)}"
                    )

                return dim

            case Kron():
                args = [self._infer_dim(e, env=env) for e in expr.exprs]

                return reduce(lambda x, y: x + y, args, [])

            case Evolve():
                hamiltonian_dim = self._infer_dim(expr.hamiltonian, env=env)
                jumps_dim = [self._infer_dim(L, env=env) for L in expr.jumps.values]

                match expr.targets:
                    case AnalogList():
                        targets_dim = [
                            d
                            for target in expr.targets.values
                            for d in self._infer_dim(target, env=env)
                        ]
                    case _:
                        targets_dim = self._infer_dim(expr.targets, env=env)

                args = (hamiltonian_dim, *jumps_dim, targets_dim)

                if all(
                    self.lattice.element_lattice.equal(arg1, arg2)
                    for arg1, arg2 in zip(args[:-1], args[1:])
                ):
                    return self.lattice.element_lattice.bottom()

                raise DimensionError(
                    f"evolve got inconsistent Hamiltonian dimensions ({args[0]}), jump operators dimensions {args[1:-1]} and target dimensions ({args[-1]})"
                )

            case _:
                return self.lattice.element_lattice.bottom()

    def merge(self, states):
        return self.lattice.merge_meet(states)

    def transfer(self, graph, node_id, state_in):
        block = graph[node_id]

        state_out = {} if state_in == self.lattice.top() else state_in.copy()
        for stmt in block.stmts:
            match stmt:
                case Declaration():
                    state_out[stmt.name] = self._infer_dim(stmt.value, env=state_out)
                case _:
                    self._infer_dim(stmt, env=state_out)

        return state_out
