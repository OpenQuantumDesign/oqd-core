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

from typing import Dict, Iterable

from oqd_compiler_infrastructure.dataflow import DataflowResult, ForwardDataflowAnalysis
from oqd_compiler_infrastructure.lattice import (
    LatticeBase,
    LatticeBottom,
    LatticeTop,
    maplattice,
)

from oqd_core.analysis.analog.types import BIN_SIG_TABLE, OP_TABLE, AnalogTypeError
from oqd_core.analysis.utils.control_flow import (
    Block,
    ControlFlowGraph,
)
from oqd_core.interface.analog import (
    Access,
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
    OperatorMul,
)


class AnalogUndefinedVarError(AnalogTypeError):
    pass

class AnalogAssignmentLattice(LatticeBase):
    """Type lattice for analog assignments."""
    pass

DefEnv = Dict[str, LatticeTop]

class AnalogDefiniteAssignmentChecker(ForwardDataflowAnalysis[int, DefEnv]):
    def __init__(self, graph: ControlFlowGraph) -> None:
        self.lattice = maplattice(AnalogAssignmentLattice)()
        self.blocks: Dict[int, Block] = graph.blocks
        self.dataflow_result: DataflowResult = self.analyze(graph, self.merge_def)
    
    def merge_def(self, states: Iterable[DefEnv]) -> DefEnv:
        states_list = list(states)
        if not states_list:
            return self.lattice.bottom()
        merged = {} if states_list[0] is LatticeBottom else dict(states_list[0])
        for state in states_list[1:]:
            if state is LatticeBottom:
                continue
            for name in set(merged).union(state):
                b1 = merged.get(name)
                b2 = state.get(name)
                if b1 is None:
                    merged[name] = b2
                elif b2 is None:
                    continue
        return merged
    
    def infer_def(self, expr, env):
        
        if isinstance(expr, AnalogList):
            for v in expr.values:
                self.infer_def(v, env)

        if isinstance(expr, Extract):
            if expr.access.name not in env:
                raise AnalogUndefinedVarError(f"Encountered undefined variable: {expr.access.name}")
    
        if isinstance(expr, Access):
            if expr.name not in env:
                raise AnalogUndefinedVarError(f"Encountered undefined variable: {expr.name}")
        
        sig = BIN_SIG_TABLE.get(type(expr))
        if sig is not None or isinstance(expr, (BoolEq, BoolNotEq)):
            self.infer_def(expr.expr1, env)
            self.infer_def(expr.expr2, env)
        
        sig = OP_TABLE.get(type(expr))
        if sig is not None or isinstance(expr, OperatorMul):
            self.infer_def(expr.op1, env)
            self.infer_def(expr.op2, env)
        
        if isinstance(expr, BoolNot):
            self.infer_def(expr.expr, env)
        
        if isinstance(expr, MathFunc):
            arg = expr.expr
            if isinstance(arg, list):
                for v in arg:
                    self.infer_def(v, env)
            else:
                self.infer_def(arg, env)
            
        if isinstance(expr, (Initialize, Measure)):
            self.infer_def(expr.targets, env)
        
        if isinstance(expr, Evolve):
            self.infer_def(expr.targets, env)
            self.infer_def(expr.duration, env)
            self.infer_def(expr.hamiltonian, env)
    
    
    def transfer(self, node_id: int, state_in: DefEnv) -> DefEnv:
        env = {} if state_in is LatticeBottom else dict(state_in)
        
        if self.blocks[node_id].preds == [] or self.blocks[node_id].succs == []:
            return env
        
        stmts = self.blocks[node_id].stmts
        
        for stmt in stmts:
            if isinstance(stmt, (Break, Continue)):
                continue
            
            if isinstance(stmt, Declaration):
                self.infer_def(stmt.value, env)
                env[stmt.name] = LatticeTop
                continue
            
            self.infer_def(stmt, env)
            
        return env
