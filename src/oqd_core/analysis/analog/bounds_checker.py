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

from __future__ import annotations

from typing import List, Tuple, Type, Union
from warnings import warn

from oqd_compiler_infrastructure import (
    CFG,
    CFGBlock,
    ForwardDataflowAnalysis,
    Lattice,
    LatticeTop,
    MapLatticeValue,
    Post,
    gen_pass,
    maplattice,
)

from oqd_core.analysis.utils import AnalysisError, AnalysisWarning
from oqd_core.frontend.analog import serialize_analog
from oqd_core.interface.analog import (
    Access,
    AnalogList,
    Constant,
    Declaration,
    Extract,
    ModeRegister,
    QuantumRegister,
)

########################################################################################


class OutOfBoundsError(AnalysisError): ...


class OutOfBoundsWarning(AnalysisWarning): ...


class BLatticeTop(LatticeTop): ...


_BLatticeValue = List["_BLatticeValue"]

BLatticeValue = Union[_BLatticeValue, Type[BLatticeTop]]


########################################################################################


class AnalogBoundsLattice(Lattice[BLatticeValue]):
    """Lattice for bounds of AnalogList, QuantumRegister and ModeRegister."""

    def _get(self, value, index, default):
        if index + 1 > len(value) or index < -len(value):
            return default

        return value[index]

    def top(self):
        return BLatticeTop

    def bottom(self):
        return []

    def leq(self, t1: BLatticeValue, t2: BLatticeValue) -> bool:
        if t1 is self.bottom() or t2 is self.top() or t1 is None:
            return True

        if t1 is self.top() or t2 is self.bottom() or t2 is None:
            return False

        if len(t1) <= len(t2):
            return all([self.leq(v1, v2) for v1, v2 in zip(t1, t2[: len(t1)])])

        return False

    def join(self, t1: BLatticeValue, t2: BLatticeValue) -> BLatticeValue:
        if t1 is self.top() or t2 is self.top():
            return self.top()
        if t1 is self.bottom() or t1 is None:
            return t2
        if t2 is self.bottom() or t2 is None:
            return t1

        v = []
        for i in range(max(len(t1), len(t2))):
            v.append(self.join(self._get(t1, i, []), self._get(t2, i, [])))

        return v

    def meet(self, t1: BLatticeValue, t2: BLatticeValue) -> BLatticeValue:
        if t1 is self.bottom() or t2 is self.bottom():
            return self.bottom()
        if t1 is self.top():
            return t2
        if t2 is self.top():
            return t1

        v = []
        for i in range(min(len(t1), len(t2))):
            v.append(
                self.meet(self._get(t1, i, BLatticeTop), self._get(t2, i, BLatticeTop))
            )

        return v


########################################################################################

BoundsEnv = MapLatticeValue[BLatticeValue]


class AnalogBoundsChecker(ForwardDataflowAnalysis[int, CFGBlock, BoundsEnv]):
    lattice = maplattice(lattice=AnalogBoundsLattice, default_mode="top")()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def merge(self, states):
        return self.lattice.merge_intersection(states)

    def _is_integer(self, expr):
        return isinstance(expr, Constant) and type(expr.value) is int

    @gen_pass(rule_type="conversion", walk=Post, method=True)
    def _infer_bounds(self, expr, operands, *, env: BoundsEnv, loc: Tuple[int, int]):

        match expr:
            case Access():
                return (
                    self.lattice.element_lattice.top()
                    if env is self.lattice.top()
                    else env.get(expr.name, self.lattice.element_lattice.top())
                )

            case list():
                return operands

            case AnalogList():
                return operands["values"]

            case QuantumRegister() | ModeRegister() if self._is_integer(expr.size):
                return [[]] * expr.size.value

            case QuantumRegister() | ModeRegister():
                warn(
                    f"(block={loc[0]}, stmt={loc[1]}) AnalogBoundschecker only checks bounds for list and registers of constant size,"
                    f" got `{serialize_analog(expr)}`",
                    OutOfBoundsWarning,
                )
                return self.lattice.element_lattice.top()

            case Extract() if self._is_integer(expr.index):
                lval = self._infer_bounds(expr.value, env=env, loc=loc)

                if lval is self.lattice.element_lattice.top():
                    return self.lattice.element_lattice.top()

                if expr.index.value + 1 > len(lval):
                    raise OutOfBoundsError(
                        f"Index {expr.index.value} out of bounds for {serialize_analog(expr)} for some path",
                        loc=loc,
                    )

                return lval[expr.index.value]

            case Extract():
                warn(
                    f"(block={loc[0]}, stmt={loc[1]}) AnalogBoundschecker only checks bounds for an index of a constant value,"
                    f" got `{serialize_analog(expr)}`",
                    OutOfBoundsWarning,
                )
                return self.lattice.element_lattice.top()

            case _:
                return self.lattice.element_lattice.bottom()

    def transfer(self, graph: CFG, node_id: int, state_in: BoundsEnv) -> BoundsEnv:
        block = graph[node_id]

        state_out = {} if state_in == self.lattice.top() else state_in.copy()
        for n, stmt in enumerate(block.stmts):
            match stmt:
                case Declaration():
                    state_out[stmt.name] = self._infer_bounds(
                        stmt.value, env=state_out, loc=(node_id, n)
                    )
                case _:
                    self._infer_bounds(stmt, env=state_out, loc=(node_id, n))

        return state_out
