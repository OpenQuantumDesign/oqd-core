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

from typing import Tuple

from oqd_compiler_infrastructure import (
    CFGBlock,
    ForwardDataflowAnalysis,
    Post,
    PowersetLattice,
    PowersetLatticeValue,
    gen_pass,
)

from oqd_core.frontend.analog import serialize_analog
from oqd_core.interface.analog import (
    Access,
    AnalogExpr,
    Declaration,
)

########################################################################################


class AvailableVariableError(Exception): ...


########################################################################################

ReachingDefinitionValue = PowersetLatticeValue[Tuple[int, str]]


class ReachingDefinition(
    ForwardDataflowAnalysis[int, CFGBlock, ReachingDefinitionValue]
):
    lattice = PowersetLattice[ReachingDefinitionValue]()

    def initial_state(self, nodes):
        return {node: self.lattice.bottom() for node in nodes}

    def merge(self, states):
        return self.lattice.merge_union(states)

    def transfer(self, graph, node_id, state_in):
        block = graph[node_id]

        state_out = state_in.copy()

        block_def = {
            (node_id, stmt.name)
            for stmt in block.stmts
            if isinstance(stmt, Declaration)
        }

        [
            state_out.discard(e)
            for e in state_out.copy()
            if e[1] in list(map(lambda x: x[1], block_def))
        ]

        return self.lattice.join(block_def, state_out)


########################################################################################

AvailableVariableValue = PowersetLatticeValue[str]


class AvailableVariableAnalysis(
    ForwardDataflowAnalysis[int, CFGBlock, AvailableVariableValue]
):
    lattice = PowersetLattice[AvailableVariableValue]()

    @gen_pass(rule_type="rewrite", walk=Post, method=True)
    def _check_variables_available(self, expr, stmt, *, env):
        if isinstance(expr, Access) and expr.name not in env:
            raise AvailableVariableError(
                f"Use of variable ({expr.name}) in statement ({serialize_analog(stmt)})"
                " that may be undefined for some path through the program"
            )

    def initial_state(self, nodes):
        return {node: self.lattice.bottom() for node in nodes}

    def merge(self, states):
        return self.lattice.merge_intersection(states)

    def transfer(self, graph, node_id, state_in):
        block = graph[node_id]

        block_def = set()
        for stmt in block.stmts:
            match stmt:
                case _ if block.edge_labels:
                    self._check_variables_available(
                        block.stmts[0], stmt, env=self.lattice.join(block_def, state_in)
                    )

                case Declaration():
                    block_def.add(stmt.name)
                    self._check_variables_available(
                        stmt.value, stmt, env=self.lattice.join(block_def, state_in)
                    )

                case _:
                    self._check_variables_available(
                        stmt, stmt, env=self.lattice.join(block_def, state_in)
                    )

        return self.lattice.join(block_def, state_in)
