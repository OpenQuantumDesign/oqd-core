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

from collections.abc import MutableMapping
from typing import Dict, Set, Tuple, Union

import graphviz
from oqd_compiler_infrastructure import (
    CFG,
    BackwardDataflowAnalysis,
    CFGBlock,
    DataflowResult,
    ForwardDataflowAnalysis,
    GraphProtocol,
    LatticeTop,
    PowersetLattice,
    PowersetLatticeValue,
)
from pydantic import BaseModel, computed_field

########################################################################################


DominatorTreeNode = Tuple[int, Set[int], Set[int]]


class DominatorTree(BaseModel, GraphProtocol, MutableMapping[int, DominatorTreeNode]):
    dominator_tree: Dict[int, Tuple[int, Set[int], Set[int]]]

    def __len__(self):
        return len(self.dominator_tree)

    def __getitem__(self, idx):
        return self.dominator_tree[idx]

    def __setitem__(self, idx, value):
        self.dominator_tree[idx] = value

    def __delitem__(self, idx):
        del self.dominator_tree[idx]

    def __iter__(self):
        return iter(self.dominator_tree)

    def nodes(self):
        return self.keys()

    def predecessors(self, n):
        return self.dominator_tree[n][1]

    def successors(self, n):
        return self.dominator_tree[n][2]

    def to_dot(self):
        G = graphviz.Digraph()
        for node in self.nodes():
            G.node(str(node))
            for pred in self.predecessors(node):
                G.edge(str(pred), str(node))

        return G


class DominatorDataflowResult(DataflowResult):
    dataflow_analysis: Union[DominatorAnalysis, PostDominatorAnalysis]
    graph: CFG
    in_states: Dict[int, DominatorLatticeValue]
    out_states: Dict[int, DominatorLatticeValue]
    iterations: int
    post: bool

    @computed_field
    @property
    def dominators(self) -> Dict[int, Set[int]]:
        return {k: set() if v is LatticeTop else v for k, v in self.out_states.items()}

    def dominates(self, d, n):
        return d in self.dominators[n]

    @computed_field
    @property
    def strict_dominators(self) -> Dict[int, Set[int]]:
        return {k: v - {k} for k, v in self.dominators.items()}

    def strictly_dominates(self, d, n):
        return d in self.strict_dominators[n]

    @computed_field
    @property
    def immediate_dominators(self) -> Dict[int, Set[int]]:
        return {
            k: set(
                filter(
                    lambda s: all([s not in self.strict_dominators[n] for n in v]),
                    v,
                )
            )
            for k, v in self.strict_dominators.items()
        }

    def immediately_dominates(self, d, n):
        return d in self.immediate_dominators[n]

    @computed_field
    @property
    def dominator_tree(self) -> DominatorTree:
        tree = {}

        nodes = sorted(self.out_states.keys(), reverse=self.post)

        while nodes:
            n = nodes.pop(0)

            try:
                idom = self.immediate_dominators[n]

                if idom:
                    idom = next(iter(idom))
                    tree[n] = (n, {idom}, set())
                    tree[idom] = (*tree[idom][:2], {n})
                else:
                    tree[n] = (n, set(), set())
            except KeyError:
                nodes.append(n)

        return DominatorTree(dominator_tree=tree)

    @computed_field
    @property
    def dominance_frontier(self) -> Dict[int, Set[int]]:
        return {
            d: set(
                filter(
                    lambda n: (
                        any(
                            [
                                self.dominates(d, m)
                                for m in self.dataflow_analysis.sources(self.graph, n)
                            ]
                        )
                        and not self.strictly_dominates(d, n)
                    ),
                    self.graph.nodes(),
                )
            )
            for d in self.dominators.keys()
        }


DominatorLatticeValue = PowersetLatticeValue[int]


class DominatorAnalysis(ForwardDataflowAnalysis[int, CFGBlock, DominatorLatticeValue]):
    lattice = PowersetLattice[DominatorLatticeValue]()

    def initial_state(self, nodes):
        return {
            node: {0} if n == 0 else self.lattice.top() for n, node in enumerate(nodes)
        }

    def merge(self, states):
        return self.lattice.merge_meet(states)

    def transfer(
        self, graph: CFG, node_id: int, state_in: DominatorLatticeValue
    ) -> DominatorLatticeValue:
        return self.lattice.join(state_in, {node_id})

    def result(self, graph, in_states, out_states, iterations):
        return DominatorDataflowResult(
            dataflow_analysis=self,
            graph=graph,
            in_states=in_states,
            out_states=out_states,
            iterations=iterations,
            post=False,
        )


########################################################################################


class PostDominatorAnalysis(
    BackwardDataflowAnalysis[int, CFGBlock, DominatorLatticeValue]
):
    lattice = PowersetLattice[DominatorLatticeValue]()

    def initial_state(self, nodes):
        return {
            node: {node} if n == len(nodes) - 1 else LatticeTop
            for n, node in enumerate(nodes)
        }

    def merge(self, states):
        return self.lattice.merge_meet(states)

    def transfer(
        self, graph, node_id: int, state_in: DominatorLatticeValue
    ) -> DominatorLatticeValue:
        return self.lattice.join(state_in, {node_id})

    def result(self, graph, in_states, out_states, iterations):
        return DominatorDataflowResult(
            dataflow_analysis=self,
            graph=graph,
            in_states=in_states,
            out_states=out_states,
            iterations=iterations,
            post=True,
        )
