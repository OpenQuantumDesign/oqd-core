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

from functools import reduce

from oqd_compiler_infrastructure import RewriteRule

from oqd_core.analysis.utils.control_flow import (
    Block,
    ControlFlowGraph,
)
from oqd_core.interface.analog import (
    AnalogCircuit,
    Break,
    Continue,
    IfElse,
    While,
)


class Accumulator(RewriteRule):
    
    def __init__(self):
        self.blocks = {}
    def _accumulate(self, block1, block2):
        self.blocks[block1].stmts += self.blocks[block2].stmts
        self.blocks[block1].succs = self.blocks[block2].succs
        self.blocks[block1].edge_labels = self.blocks[block2].edge_labels
        for succ in self.blocks[block2].succs:
            self.blocks[succ].preds[self.blocks[succ].preds.index(block2)] = block1
        self.blocks.pop(block2)
        
        return block1
    
    def map_ControlFlowGraph(self, model: ControlFlowGraph):
        self.blocks = model.blocks
        accumulated_blocks = []
        for block in self.blocks.values():
            if block.register_id == 0 or not block.succs:
                continue
            if 0 in block.preds and not block.edge_labels:
                acc = self(block)
                if len(acc) > 1:
                    accumulated_blocks.append(acc)
            if any(self.blocks[pred].edge_labels for pred in block.preds) and not block.edge_labels:
                acc = self(block)
                if len(acc) > 1:
                    accumulated_blocks.append(acc)
                    
                    
        for acc in accumulated_blocks:
            reduce(self._accumulate, acc)
            
        # print(accumulated_blocks)
        return ControlFlowGraph(blocks=self.blocks)
    
    
    def map_Block(self, model: Block):
        block = model
        blocks = []
        while True:
            if len(block.succs) != 1:
                break
            if len(block.preds) > 1 and block != model:
                break 
            blocks.append(block.register_id)
            block = self.blocks[block.succs[0]]
        # print(blocks)
        return blocks
    
class AnalogCFGBuilder(RewriteRule):
    def __init__(self):
        super().__init__()
        self.index = 0
        self.blocks = {}
        self.loop_stack = []
        self.preds = []
        self.edge_labels = None
        self.fallthrough_labels = {}
        
    def new_node(self, preds, stmt):
        # print("edge_labels: ", self.edge_labels)
        # print("fallthrough: ", self.index, self.fallthrough_labels)
        node = Block(register_id=self.index, stmts=[stmt], preds=preds)
        self.blocks[node.register_id] = node
        self.index += 1
        
        explicit_labels = self.edge_labels or {}
        self.edge_labels = None
        
        for pred in node.preds:
            label = explicit_labels.get(pred)
            if label is None:
                label = self.fallthrough_labels.pop(pred, None)
                # print(pred, label)
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
    
    def run(self, circuit: AnalogCircuit) -> ControlFlowGraph:
        self.index = 0
        self.blocks = {}
        self.loop_stack = []
        self.edge_labels = None
        self.fallthrough_labels = {}
        node = self.new_node([], {})
        node = self.walk_stmt(circuit, [node])
        node = self.new_node(node, {})
        return ControlFlowGraph(blocks=self.blocks)
    
    def map_AnalogCircuit(self, model: AnalogCircuit):
        return self.walk_block(model.statements, self.preds)
    
    def map_IfElse(self, model: IfElse):
        node = self.new_node(self.preds, model.condition)
        then_branch = self.walk_block(model.then_branch, [node], entry_label="true")
        if model.else_branch:
            else_branch = self.walk_block(model.else_branch, [node], entry_label="false")
            return then_branch + else_branch
        
        self.fallthrough_labels[node] = "false"
        return then_branch + [node]
    
    def map_While(self, model: While): 
        node = self.new_node(self.preds, model.condition)
        self.fallthrough_labels[node] = "false"
        self.loop_stack.append(node)
        body = self.walk_block(model.body, [node], entry_label="true")
        self.loop_stack.pop()
        
        self.blocks[node].add_preds(body)
        for s in body:
            label = self.fallthrough_labels.pop(s, None)
            self.blocks[s].add_succ(node, label=label)
        
        return self.blocks[node].exit_nodes + [node]
    
    def map_Break(self, model: Break):
        if not self.loop_stack:
            raise TypeError("break statement used outside loop")
        break_node = self.new_node(self.preds, model)
        self.blocks[self.loop_stack[-1]].exit_nodes.append(break_node)
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

