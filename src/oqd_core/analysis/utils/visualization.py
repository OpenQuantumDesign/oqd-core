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


import graphviz
from oqd_compiler_infrastructure import RewriteRule

from oqd_core.analysis.utils import ControlFlowGraph
from oqd_core.frontend.analog.serialize import serialize_analog
from oqd_core.frontend.atomic.serialize import serialize_atomic

########################################################################################


class CFGtoDot(RewriteRule):
    def __init__(self, *, mode="analog", max_lines=5):
        match mode:
            case "analog":
                self.serialize = serialize_analog
            case "atomic":
                self.serialize = serialize_atomic
            case _:
                raise ValueError("CFGtoDot only supports analog and atomic mode")

        self.max_lines = max_lines

    def map_ControlFlowGraph(self, model):
        self.dot = graphviz.Digraph()

        for block in model.blocks.values():
            self(block)

        return self.dot

    def map_Block(self, model):

        if model.edge_labels:
            label = [f"Condition: {self(model.stmts[0])}"]
        else:
            label = [self(stmt) for stmt in model.stmts]

        if len(label) > self.max_lines and self.max_lines >= 0:
            label = label[: self.max_lines] + ["..."]

        self.dot.node(
            str(model.register_id),
            f"Block #{model.register_id}\n{'-' * 16}\n" + "\n".join(label),
        )

        for succ in model.succs:
            self.dot.edge(str(model.register_id), str(succ))

    def map_VisitableBaseModel(self, model):
        return f"{self.serialize(model)}"


def cfg_to_dot(
    cfg: ControlFlowGraph, *, mode="analog", max_lines=5
) -> graphviz.Digraph:
    _cfg2dot = CFGtoDot(mode=mode, max_lines=max_lines)

    return _cfg2dot(cfg)
