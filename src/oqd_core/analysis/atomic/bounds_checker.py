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

from typing import List, Type, Union

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

from oqd_core.frontend.atomic import serialize_atomic
from oqd_core.interface.atomic import (
    Access,
    AtomicList,
    Constant,
    Declaration,
    Extract,
    IonRegister,
)

########################################################################################


class OutOfBoundsError(Exception): ...


class BLatticeTop(LatticeTop): ...


_BLatticeValue = List["_BLatticeValue"]

BLatticeValue = Union[_BLatticeValue, Type[BLatticeTop]]


########################################################################################


class AtomicBoundsLattice(Lattice[BLatticeValue]):
    """Lattice for bounds of AtomicList and IonRegister"""

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


class AtomicBoundsChecker(ForwardDataflowAnalysis[int, CFGBlock, BoundsEnv]):
    lattice = maplattice(lattice=AtomicBoundsLattice, default_mode="top")()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def merge(self, states):
        return self.lattice.merge_intersection(states)

    def _is_integer(self, expr):
        return isinstance(expr, Constant) and type(expr.value) is int

    @gen_pass(rule_type="conversion", walk=Post, method=True)
    def _infer_bounds(self, expr, operands, *, env: BoundsEnv):

        match expr:
            case Access():
                return (
                    self.lattice.element_lattice.top()
                    if env is self.lattice.top()
                    else env.get(expr.name, self.lattice.element_lattice.top())
                )

            case list():
                return operands

            case AtomicList():
                return operands["values"]

            case IonRegister() if self._is_integer(expr.size):
                return [[]] * expr.size.value

            case IonRegister():
                raise OutOfBoundsError(
                    "Out of bounds checker only works for constant args for IonRegister"
                )

            case Extract() if self._is_integer(expr.index):
                lval = self._infer_bounds(expr.value, env=env)

                if expr.index.value + 1 > len(lval):
                    raise OutOfBoundsError(
                        f"Index {expr.index.value} out of bounds for {serialize_atomic(expr)} for some path"
                    )

                return lval[expr.index.value]

            case Extract():
                raise OutOfBoundsError(
                    "Out of bounds checker only works for constant index for Extract"
                )

            case _:
                return self.lattice.element_lattice.bottom()

    def transfer(self, graph: CFG, node_id: int, state_in: BoundsEnv) -> BoundsEnv:
        block = graph[node_id]

        state_out = {} if state_in == self.lattice.top() else state_in.copy()
        for stmt in block.stmts:
            match stmt:
                case Declaration():
                    state_out[stmt.name] = self._infer_bounds(stmt.value, env=state_out)
                case _:
                    self._infer_bounds(stmt, env=state_out)

        return state_out
