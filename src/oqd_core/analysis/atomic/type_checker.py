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

from oqd_compiler_infrastructure.dataflow import ForwardDataflowAnalysis
from oqd_compiler_infrastructure.lattice import LatticeBottom, maplattice

from oqd_core.analysis.atomic.semantics import AtomicSemantics
from oqd_core.analysis.atomic.types import (
    AtomicTypeError,
    AtomicTypeLattice,
    TBool,
    TPulse,
    TypeEnv,
    type_name,
)
from oqd_core.analysis.utils.control_flow import ControlFlowGraph
from oqd_core.interface.atomic import (
    Break,
    Continue,
    Declaration,
    ParallelProtocol,
    SerialProtocol,
)

########################################################################################


class AtomicTypeChecker(ForwardDataflowAnalysis[int, TypeEnv]):
    """Forward dataflow type checker over the Control Flow Graph."""
    def __init__(self, graph: ControlFlowGraph) -> None:
        self.value_lattice = AtomicTypeLattice()
        self.semantics = AtomicSemantics(self.value_lattice)
        self.lattice = maplattice(AtomicTypeLattice)()
        self.blocks = graph.blocks
        
        self.dataflow_result = self.analyze(graph, self.merge_union)
    
    def check_protocol(self, protocol, env: TypeEnv) -> TypeEnv:
        if isinstance(protocol, SerialProtocol):
            curr_env = env
            for child in protocol.pulses:
                curr_env = self.check_protocol_stmt(child, curr_env)
            return curr_env

        if isinstance(protocol, ParallelProtocol):
            for child in protocol.pulses:
                self.check_protocol_stmt(child, env)
            return env

        raise AtomicTypeError(f"Unsupported protocol node: {type(protocol).__name__}")

    def check_protocol_stmt(self, stmt, env: TypeEnv) -> TypeEnv:
        if isinstance(stmt, (ParallelProtocol, SerialProtocol)):
            return self.check_protocol(stmt, env)

        if isinstance(stmt, Declaration):
            t = self.semantics.infer_type(stmt.value, env)
            if not self.value_lattice.leq(t, TPulse):
                raise AtomicTypeError(
                    f"Parallel/Serial blocks expect only Pulse statements, got {type_name(t)}"
                )
            out = dict(env)
            out[stmt.name] = t
            return out

        t = self.semantics.infer_type(stmt, env)
        if not self.value_lattice.leq(t, TPulse):
            raise AtomicTypeError(
                f"Parallel/Serial blocks expect only Pulse statements, got {type_name(t)}"
            )
        return env
    
    def transfer(self, node_id: int, state_in: TypeEnv) -> TypeEnv:
        env = {} if state_in is LatticeBottom else dict(state_in)
        stmt = self.blocks[node_id].stmt
        
        if isinstance(stmt, Declaration):
            state_out = dict(env)
            state_out[stmt.name] = self.semantics.infer_type(stmt.value, env)
            return state_out
        
        if isinstance(stmt, (Break, Continue)):
            return env
        
        if isinstance(stmt, (ParallelProtocol, SerialProtocol)):
            self.check_protocol(stmt, env)
            return env
        
        t = self.semantics.infer_type(stmt, env)
        if self.blocks[node_id].edge_labels and t is not TBool:
            raise AtomicTypeError("branch condition must be bool")
        
        return env

