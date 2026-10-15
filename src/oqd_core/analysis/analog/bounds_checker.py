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

from typing import Dict, Union

from oqd_compiler_infrastructure import (
    CFG,
    CFGBlock,
    ForwardDataflowAnalysis,
    LatticeBase,
    LatticeTop,
    Post,
    gen_pass,
    maplattice,
)

from oqd_core.interface.analog import (
    AnalogList,
    Declaration,
    Extract,
    ModeRegister,
    QuantumRegister,
)


class OutOfBoundsError(Exception): ...


BLatticeValue = Union[int, LatticeTop]
BoundsEnv = Dict[str, BLatticeValue]

class AnalogBoundsLattice(LatticeBase[BLatticeValue]):
    """Lattice for bounds of AnalogList, QuantumRegister and ModeRegister."""
    def leq(self, t1: BLatticeValue, t2: BLatticeValue) -> bool:
        if t1 is self.bottom() or t2 is self.top() or t1 is None:
            return True
        if t1 is self.top() or t2 is self.bottom() or t2 is None:
            return False
        return t1 <= t2

    def join(self, t1: BLatticeValue, t2: BLatticeValue) -> BLatticeValue:
        if t1 is self.top() or t2 is self.top():
            return self.top()
        if t1 is self.bottom() or t1 is None:
            return t2
        if  t2 is self.bottom() or t2 is None:
            return t1
        return max(t1, t2)

    def meet(self, t1: BLatticeValue, t2: BLatticeValue) -> BLatticeValue:
        if t1 is self.bottom() or t2 is self.bottom():
            return self.bottom()
        if t1 is self.top():
            return t2
        if  t2 is self.top():
            return t1
        if t1 is None or t2 is None:
            return self.bottom()
        return min(t1, t2)


class AnalogBoundsChecker(ForwardDataflowAnalysis[int, CFGBlock, BoundsEnv]):
    lattice = maplattice(lattice = AnalogBoundsLattice, default_mode="top")()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def merge(self, states):
        return self.lattice.merge_intersection(states)

    @gen_pass(rule_type="rewrite", walk=Post, method=True)
    def _infer_bounds(self, expr, *, env: BoundsEnv):

        if isinstance(expr, Extract) and isinstance(env, Dict):
            if expr.access.name not in env:
                raise OutOfBoundsError(f"Cannot index into variable: {expr.access.name}")
            if expr.index + 1 > env[expr.access.name]:
                raise OutOfBoundsError(f"Index {expr.index} out of bounds for variable: {expr.access.name}")
    

    def transfer(self, graph: CFG, node_id: int, state_in: BoundsEnv) -> BoundsEnv:
        block = graph[node_id]

        state_out = {} if state_in == self.lattice.top() else state_in.copy()

        for stmt in block.stmts:

            self._infer_bounds(stmt, env=state_out)
            
            if isinstance(stmt, Declaration):
                if isinstance(stmt.value, AnalogList):
                    state_out[stmt.name] = len(stmt.value.values)
                if isinstance(stmt.value, (QuantumRegister, ModeRegister)):
                    state_out[stmt.name] = stmt.value.size
        
        return state_out
