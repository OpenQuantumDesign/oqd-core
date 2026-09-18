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

from oqd_compiler_infrastructure import (
    BackwardDataflowAnalysis,
    CFGBlock,
    ForwardDataflowAnalysis,
    LatticeTop,
    PowersetLattice,
    PowersetValue,
)

########################################################################################


class DominatorTreeAnalysis(ForwardDataflowAnalysis[int, CFGBlock, PowersetValue]):
    lattice = PowersetLattice()

    def initial_state(self, nodes):
        return {node: {0} if n == 0 else LatticeTop for n, node in enumerate(nodes)}

    def merge(self, states):
        return self.merge_intersection(states)

    def transfer(self, graph, node_id: int, state_in: PowersetValue) -> PowersetValue:
        return self.lattice.join(state_in, {node_id})


########################################################################################


class PostDominatorTreeAnalysis(BackwardDataflowAnalysis[int, CFGBlock, PowersetValue]):
    lattice = PowersetLattice()

    def initial_state(self, nodes):
        return {
            node: {node} if n == len(nodes) - 1 else LatticeTop
            for n, node in enumerate(nodes)
        }

    def merge(self, states):
        return self.merge_intersection(states)

    def transfer(self, graph, node_id: int, state_in: PowersetValue) -> PowersetValue:
        return self.lattice.join(state_in, {node_id})
