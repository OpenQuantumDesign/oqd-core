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
import inspect
from typing import Dict, Iterable, Union

from oqd_compiler_infrastructure import (
    DataflowResult, 
    ForwardDataflowAnalysis,
    CFGBlock,
    CFG,
    LatticeBase,
    LatticeBottom,
    LatticeTop,
    maplattice,
)

from oqd_core.interface.analog import (
    AnalogList,
    BoolEq,
    BoolNot,
    BoolNotEq,
    Break,
    Continue,
    Declaration,
    Evolve,
    Extract,
    Initialize,
    MathFunc,
    Measure,
    ModeRegister,
    OperatorMul,
    QuantumRegister,
)


class OutOfBoundsError(Exception): ...


BLatticeValue = Union[int, LatticeTop]
BoundsEnv = Dict[str, BLatticeValue]

class AnalogBoundsLattice(LatticeBase[BLatticeValue]):
    """Lattice for bounds of AnalogList, QuantumRegister and ModeRegister."""
    def leq(self, t1: BLatticeValue, t2: BLatticeValue) -> bool:
        # print("t1: ", t1)
        # print("t2: ", t2)
        if t1 is self.bottom() or t2 is self.top():
            return True
        if t1 is self.top() or t2 is self.bottom():
            return False
        return t1 <= t2

    def join(self, t1: BLatticeValue, t2: BLatticeValue) -> BLatticeValue:
        print("t1: ", t1)
        print("t2: ", t2)
        if t1 is self.top() or t2 is self.top():
            return self.top()
        if t1 is self.bottom():
            return t2
        if  t2 is self.bottom():
            return t1
        return max(t1, t2)

    def meet(self, t1: BLatticeValue, t2: BLatticeValue) -> BLatticeValue:
        if t1 is self.bottom() or t2 is self.bottom():
            return self.bottom()
        if t1 is self.top():
            return t2
        if  t2 is self.top():
            return t1
        return min(t1, t2)


class AnalogBoundsChecker(ForwardDataflowAnalysis[int, CFGBlock, BoundsEnv]):
    lattice = maplattice(AnalogBoundsLattice)()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def merge(self, states):
        return self.lattice.merge_meet(states)

    def _infer_bounds(self, expr, *, env: BoundsEnv):
        match expr:
            case Extract():
                if expr.access.name not in env:
                    raise OutOfBoundsError(f"Cannot index into variable: {expr.access.name}")
                if not self.lattice.leq(expr.index + 1, env[expr.access.name]):
                    raise OutOfBoundsError(f"Index {expr.index} out of bounds for variable: {expr.access.name}")
            case (
                QuantumRegister()
                | ModeRegister()
            ):
                return expr.size
            case AnalogList():
                for val in expr.values:
                    self._infer_bounds(val, env=env)
                return len(expr.values)
            case _:
                if isinstance(expr, (int, float, str)):
                    return
                for a in getattr(expr, "model_fields_set"):
                    self._infer_bounds(getattr(expr, a), env=env)
                
    
    
    def transfer(self, graph: CFG, node_id: int, state_in: BoundsEnv) -> BoundsEnv:
        block = graph[node_id]

        state_out = {} if state_in == self.lattice.top() else state_in.copy()
        if block.edge_labels:
            return state_out
        
        for stmt in block.stmts:
            if isinstance(stmt, (Break, Continue)):
                continue
            if isinstance(stmt, Declaration):
                val = self._infer_bounds(stmt.value, env=state_out)
                if val is not None:
                    state_out[stmt.name] = val
                continue

            self._infer_bounds(stmt, env=state_out)
        
        return state_out
