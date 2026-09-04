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

import graphviz

from oqd_core.analysis.utils import ControlFlowGraph


def cfg_to_dot(cfg: ControlFlowGraph) -> graphviz.Digraph:
    dot = graphviz.Digraph()

    for node_id, block in sorted(cfg.blocks.items()):
        stmt_label = getattr(block.stmt, "class__", type(block.stmt).__name__)
        if stmt_label == "Declaration":
            stmt_label = f"{block.stmt.name} = ..."
        if stmt_label in ("ParallelProtocol", "SerialProtocol"):
            stmt_label = f"{stmt_label}({len(block.stmt.pulses)})"
        label = (
            f"{node_id}: {stmt_label}\\n"
        )
        dot.node(str(node_id), label)

        for succ in block.succs:
            dot.edge(str(node_id), str(succ.register_id))

    return dot
