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

from collections.abc import Iterable
from functools import reduce

from oqd_compiler_infrastructure import CFG, CFGBlock, RewriteRule

from oqd_core.interface.analog import (
    AnalogCircuit,
    Break,
    Continue,
    IfElse,
    While,
)


class AnalogCFGBuilder(RewriteRule):
    def new_node(self, preds, stmt, tags=None):
        node = CFGBlock(
            register_id=self.index,
            stmts=[stmt] if stmt else [],
            preds=preds,
            tags=tags if tags else {},
        )
        self.blocks[node.register_id] = node
        self.index += 1

        explicit_labels = self.edge_labels or {}
        self.edge_labels = None

        for pred in node.preds:
            label = explicit_labels.get(pred)
            if label is None:
                label = self.fallthrough_labels.pop(pred, None)

            self.blocks[pred].add_succ(node.register_id, label=label)

        return node.register_id

    def walk_stmt(self, stmt, preds, edge_labels=None):
        old = self.preds
        old_labels = self.edge_labels
        self.preds = preds
        self.edge_labels = edge_labels
        result = self(stmt)
        self.preds = old
        self.edge_labels = old_labels
        return result

    def walk_block(self, statements, preds, entry_label=None):
        edge_labels = {}

        if entry_label:
            edge_labels = {preds[0]: entry_label}

        for stmt in statements:
            preds = self.walk_stmt(stmt, preds, edge_labels=edge_labels)
            edge_labels = None
        return preds

    def map_AnalogCircuit(self, model: AnalogCircuit) -> CFG:
        self.index = 0
        self.blocks = {}
        self.loop_stack = []
        self.preds = []
        self.edge_labels = None
        self.fallthrough_labels = {}
        node = self.new_node([], [])
        node = self.walk_block(model.statements, [node])
        node = self.new_node(node, [])
        return CFG(blocks=self.blocks)

    def map_IfElse(self, model: IfElse):
        node = self.new_node(
            self.preds,
            model.condition,
            tags={"__scf__": model.__class__.__qualname__},
        )

        then_branch = (
            self.walk_block(model.then_branch, [node], entry_label="true")
            if model.then_branch
            else [node]
        )
        else_branch = (
            self.walk_block(model.else_branch, [node], entry_label="false")
            if model.else_branch
            else [node]
        )

        fallthrough_labels = []
        if model.then_branch == []:
            fallthrough_labels.append("true")
        if model.else_branch == []:
            fallthrough_labels.append("false")

        self.fallthrough_labels[node] = fallthrough_labels

        return then_branch + else_branch

    def map_While(self, model: While):
        node = self.new_node(
            self.preds,
            model.condition,
            tags={"__scf__": model.__class__.__qualname__},
        )

        if model.body:
            self.loop_stack.append(node)
            body = self.walk_block(model.body, [node], entry_label="true")
            self.loop_stack.pop()

            self.blocks[node].add_preds(body)
            for loop_back in body:
                label = self.fallthrough_labels.pop(loop_back, None)
                self.blocks[loop_back].add_succ(node, label=label)
        else:
            self.blocks[node].add_succ(node, label="true")

        self.fallthrough_labels[node] = "false"

        return list(self.blocks[node].exit_nodes) + [node]

    def map_Break(self, model: Break):
        if not self.loop_stack:
            raise TypeError("break statement used outside loop")
        break_node = self.new_node(self.preds, model)
        self.blocks[self.loop_stack[-1]].exit_nodes.add(break_node)
        return []

    def map_Continue(self, model: Continue):
        if not self.loop_stack:
            raise TypeError("continue statement used outside loop")
        continue_node = self.new_node(self.preds, model)
        self.blocks[self.loop_stack[-1]].add_pred(continue_node)
        self.blocks[continue_node].add_succ(self.loop_stack[-1])
        return []

    def generic_map(self, model):
        return [self.new_node(self.preds, model)]


class AnalogCFGtoAST(RewriteRule):
    def __init__(self, pdom_result):
        super().__init__()
        self._pdom_result = pdom_result

    @property
    def pdom(self):
        return self._pdom_result.out_states

    @property
    def dataflow_analysis(self):
        return self._pdom_result.dataflow_analysis

    @property
    def lattice(self):
        return self._pdom_result.dataflow_analysis.lattice

    def _consume_ifelse(self, current_block, blocks):
        ifelse_until = reduce(
            self.lattice.meet,
            [self.pdom[succ] for succ in current_block.succs],
        )

        then_block, then_succ = self._consume(
            blocks,
            start=current_block.edge_labels["true"],
            until=ifelse_until,
        )

        else_block, else_succ = self._consume(
            blocks,
            start=current_block.edge_labels["false"],
            until=ifelse_until,
        )

        return IfElse(
            condition=current_block.stmts[0],
            then_branch=then_block,
            else_branch=else_block,
        ), else_succ

    def _get_while_dead_blocks(self, blocks, end):
        return [
            b
            for b in blocks.keys()
            if (len(blocks[b].preds) == 0 and end in self.pdom[b])
        ]

    def _consume_while(self, current_block, blocks):
        while_until = reduce(
            self.lattice.meet,
            [self.pdom[succ] for succ in current_block.succs],
        )

        loop_block, loop_succ = self._consume(
            blocks,
            start=current_block.edge_labels["true"],
            until=while_until,
        )

        while_dead_blocks = self._get_while_dead_blocks(
            blocks, current_block.edge_labels["false"]
        )

        for b in sorted(while_dead_blocks):
            loop_block.extend(
                self._consume(blocks, start=b, until={current_block.register_id})[0]
            )

        return While(
            condition=current_block.stmts[0], body=loop_block
        ), current_block.edge_labels["false"]

    def _consume(self, blocks, start=0, until=None):
        succ = start
        statements = []
        while succ in blocks.keys():
            if until and succ in until:
                break

            current_block = blocks.pop(succ)

            if len(current_block.succs) == 0:
                statements.extend(current_block.stmts)
                break

            match current_block.tags.get("__scf__", None):
                case "IfElse":
                    ifelse_statement, ifelse_succ = self._consume_ifelse(
                        current_block, blocks
                    )
                    statements.append(ifelse_statement)
                    succ = ifelse_succ

                case "While":
                    while_statement, while_succ = self._consume_while(
                        current_block, blocks
                    )
                    statements.append(while_statement)
                    succ = while_succ

                case _:
                    statements.extend(current_block.stmts)
                    succ = list(current_block.succs)[0]

        return statements, succ

    def map_CFG(self, model):
        circuit = AnalogCircuit()

        statements, _ = self._consume(model.blocks.copy())
        circuit.statements.extend(statements)

        return circuit
