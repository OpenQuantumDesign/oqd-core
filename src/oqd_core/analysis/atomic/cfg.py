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

from oqd_compiler_infrastructure import RewriteRule

from oqd_core.analysis.utils.control_flow import (
    Block,
    ControlFlowGraph,
)
from oqd_core.interface.atomic import (
    AtomicCircuit,
    Break,
    Continue,
    IfElse,
    ParallelProtocol,
    SerialProtocol,
    While,
)


class AtomicCFGBuilder(RewriteRule):
    def __init__(self):
        super().__init__()
        self.registry = 0
        self.blocks = {}
        self.loop_stack = []
        self.preds = []
        self.edge_labels = None
        self.fallthrough_labels = {}
        
    
    def new_node(self, preds, stmt, kind = "stmt"):
        node = Block(register_id=self.registry, stmt=stmt,preds=preds)
        self.blocks[node.register_id] = node
        self.registry += 1
        
        explicit_labels = self.edge_labels or {}
        self.edge_labels = None
        
        for pred in node.preds:
            label = explicit_labels.get(pred.register_id)
            if label is None:
                label = self.fallthrough_labels.pop(pred.register_id, None)
            pred.add_succ(node, label=label)
        
        return node
    
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
            edge_labels = {preds[0].register_id: entry_label}
    
        for stmt in statements:
            preds = self.walk_stmt(stmt, preds, edge_labels=edge_labels)
            edge_labels = None
        return preds
    
    def run(self, circuit: AtomicCircuit) -> ControlFlowGraph:
        self.registry = 0
        self.blocks = {}
        self.loop_stack = []
        self.edge_labels = None
        self.fallthrough_labels = {}
        self.walk_stmt(circuit, [])
        return ControlFlowGraph(blocks=self.blocks)
    
    def map_AtomicCircuit(self, model: AtomicCircuit):
        return self.walk_block(model.statements, self.preds)
    
    def map_IfElse(self, model: IfElse):
        node = self.new_node(self.preds, model.condition)
        then_branch = self.walk_block(model.then_branch, [node], entry_label="true")
        if model.else_branch:
            else_branch = self.walk_block(model.else_branch, [node], entry_label="false")
            return list(then_branch) + (list(else_branch))
        
        self.fallthrough_labels[node.register_id] = "false"
        
        return list(then_branch) + [node]
    
    def map_While(self, model: While):
        node = self.new_node(self.preds, model.condition)
        self.loop_stack.append(node)
        body = self.walk_block(model.body, [node], entry_label="true")
        self.loop_stack.pop()
        
        node.add_preds(body)
        self.fallthrough_labels[node.register_id] = "false"
        
        return node.exit_nodes + [node]
    
    def map_Break(self, model: Break):
        if not self.loop_stack:
            raise TypeError("break statement used outside loop")
        break_node = self.new_node(self.preds, model)
        self.loop_stack[-1].exit_nodes.append(break_node)
        self.fallthrough_labels[break_node.register_id] = "break"
        return []
    
    def map_Continue(self, model: Continue):
        if not self.loop_stack:
            raise TypeError("continue statement used outside loop")
        continue_node = self.new_node(self.preds, model)
        self.loop_stack[-1].add_pred(continue_node, label="continue")
        return []
    
    def map_ParallelProtocol(self, model: ParallelProtocol):
        node = self.new_node(self.preds, model)
        return self.walk_block(model.pulses, [node])
    
    def map_SerialProtocol(self, model: SerialProtocol):
        node = self.new_node(self.preds, model)
        return self.walk_block(model.pulses, [node])
    
    def generic_map(self, model):
        return [self.new_node(self.preds, model)]
    

