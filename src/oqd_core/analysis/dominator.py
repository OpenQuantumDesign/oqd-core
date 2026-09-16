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

from collections import deque

from oqd_compiler_infrastructure import (
    CFGBlock,
    ForwardDataflowAnalysis,
    LatticeTop,
    PowersetLattice,
)
from oqd_compiler_infrastructure.lattice import PowersetValue

########################################################################################


class DominatorTreeAnalysis(ForwardDataflowAnalysis[int, CFGBlock, PowersetValue]):
    lattice = PowersetLattice()

    def init_state(self, nodes):
        return {node: {0} if n == 0 else LatticeTop for n, node in enumerate(nodes)}

    def analyze(self, graph):
        nodes = list(graph.nodes())
        boundary = self.init_state(nodes)
        result = self.init_state(nodes)

        worklist = deque(nodes)
        iterations = 0

        while worklist:
            node = worklist.popleft()
            iterations += 1

            srcs = list(self.sources(graph, node))
            if srcs:
                merged_input = self.merge_intersection(result[n] for n in srcs)
            else:
                merged_input = result[node]

            if not self.lattice.equal(boundary[node], merged_input):
                boundary[node] = merged_input

            next_result = self.transfer(graph, node, merged_input)
            if self.lattice.equal(result[node], next_result):
                continue

            result[node] = next_result
            for target in self.targets(graph, node):
                if target not in worklist:
                    worklist.append(target)

        return self.result(boundary, result, iterations)

    def transfer(self, graph, node_id: int, state_in: PowersetValue) -> PowersetValue:
        return self.lattice.join(state_in, {node_id})
