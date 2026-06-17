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

from typing import Union, Iterable
from oqd_compiler_infrastructure.dataflow import ForwardDataflowAnalysis
from oqd_compiler_infrastructure.lattice import (
    Lattice,
    LatticeBottom,
    LatticeTop,
    maplattice,
)
from oqd_core.compiler.analog.passes.canonicalize import (
    resolve_operator_expr,
    canonicalize_operator_expr,
)
from oqd_core.analysis.utils.control_flow import ControlFlowGraph
from oqd_core.interface.analog import Declaration, Evolve
from oqd_core.interface.analog.expr import OperatorExpr
from oqd_core.compiler.analog.error import AnalogCompilerError

OperatorEnv = dict[str, OperatorExpr]




class OperatorExprLattice(Lattice[Union[OperatorExpr, type[LatticeTop]]]):
    def top(self) -> type[LatticeTop]:
        return LatticeTop
    
    def bottom(self) -> type[LatticeTop]:
        return LatticeBottom
    
    def leq(self, t1, t2) -> bool:
        if t1 is LatticeBottom:
            return True
        if t2 is LatticeTop:
            return True
        if t1 is LatticeTop:
            return t2 is LatticeTop
        if t2 is LatticeBottom:
            return False
        return t1 == t2
    
    def join(self, t1, t2):
        if t1 is LatticeTop or t2 is LatticeTop:
            return LatticeTop
        if t1 is LatticeBottom:
            return t2
        if t2 is LatticeBottom:
            return t1
        if t1 == t2:
            return t1
        return LatticeTop
    
    def meet(self, t1, t2):
        if t1 is LatticeBottom or t2 is LatticeBottom:
            return LatticeBottom
        if t1 is LatticeTop:
            return t2
        if t2 is LatticeTop:
            return t1
        if t1 == t2:
            return t1
        return LatticeBottom



class OperatorEnvBuilder(ForwardDataflowAnalysis[int, OperatorEnv]):
    """Forward dataflow for operator binding."""
    def __init__(self, graph: ControlFlowGraph) -> None:
        
        self.lattice = maplattice(OperatorExprLattice)()
        self.blocks = graph.blocks
        
        self.dataflow_result = self.analyze(graph, self.merge_operator_env)

    
    def merge_operator_env(self, states: Iterable[OperatorEnv]) -> OperatorEnv:
        states_list = list(states)
        if not states_list:
            return self.lattice.bottom()
        
        merged = {} if states_list[0] is LatticeBottom else dict(states_list[0])
        for state in states_list[1:]:
            if state is LatticeBottom:
                continue
            for name in set(merged).union(state):
                op1 = merged.get(name)
                op2 = state.get(name)
                if op1 is None:
                    merged[name] = op2
                elif op2 is None:
                    continue
                elif op1 != op2:
                    raise AnalogCompilerError(f"Incompatible operator bindings for {name}")
        return merged
    
    def transfer(self, node_id: int, state_in: OperatorEnv) -> OperatorEnv:
        env = {} if state_in is LatticeBottom else dict(state_in)
        stmt = self.blocks[node_id].stmt
        
        if isinstance(stmt, Declaration) and isinstance(stmt.value, OperatorExpr):
            value = canonicalize_operator_expr(stmt.value)
            stmt.value = value
            state_out = dict(env)
            state_out[stmt.name] = value
            return state_out
        
        if isinstance(stmt, Evolve):
            resolved = resolve_operator_expr(stmt.hamiltonian, env)
            stmt.hamiltonian = canonicalize_operator_expr(resolved)
            return env
        
        return env
    


def canonicalize_operators_cfg(cfg: ControlFlowGraph):
    OperatorEnvBuilder(cfg)
    return cfg

