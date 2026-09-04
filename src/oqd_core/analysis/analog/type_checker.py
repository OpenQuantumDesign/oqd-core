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

from oqd_compiler_infrastructure.dataflow import DataflowResult, ForwardDataflowAnalysis
from oqd_compiler_infrastructure.lattice import LatticeBottom, maplattice

from oqd_core.analysis.analog.semantics import AnalogSemantics
from oqd_core.analysis.analog.types import (
    AnalogTypeError,
    AnalogTypeLattice,
    TBool,
    TypeEnv,
)
from oqd_core.analysis.utils.control_flow import (
    Block,
    ControlFlowGraph,
)
from oqd_core.interface.analog import Break, Continue, Declaration


class AnalogTypeChecker(ForwardDataflowAnalysis[int, TypeEnv]):
    """Forward dataflow type checker over the Control Flow Graph."""
    def __init__(self, graph: ControlFlowGraph) -> None:
        self.value_lattice = AnalogTypeLattice()
        self.semantics = AnalogSemantics(self.value_lattice)
        self.lattice = maplattice(AnalogTypeLattice)()
        self.blocks: dict[int, Block] = graph.blocks
        
        self.dataflow_result: DataflowResult = self.analyze(graph, self.merge_union)

    def transfer(self, node_id: int, state_in: TypeEnv) -> TypeEnv:
        env = {} if state_in is LatticeBottom else dict(state_in)
        stmt = self.blocks[node_id].stmt

        if isinstance(stmt, Declaration):
            state_out = dict(env)
            state_out[stmt.name] = self.semantics.infer_type(stmt.value, env)
            return state_out

        if isinstance(stmt, (Break, Continue)):
            return env

        t = self.semantics.infer_type(stmt, env)
        if self.blocks[node_id].edge_labels and t is not TBool:
            raise AnalogTypeError("branch condition must be bool")

        return env

