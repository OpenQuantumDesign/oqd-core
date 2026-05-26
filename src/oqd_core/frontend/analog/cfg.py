# Copyright 2024-2025 Open Quantum Design

# Licensed under the Apache License, Version 2.0 (the "License")
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

from oqd_core.analysis.utils import CFGNode
from oqd_core.interface.analog import (
    AnalogCircuit,
    Break,
    Continue,
    IfElse,
    While,
)


class AnalogCFGBuilder(RewriteRule):
    def __init__(self):
        super().__init__()
        self.registry = 0
        self.cache = {}
        self.loop_stack = []
        self.preds = []
        self.founder = None
        self.last_node = None
        self.edge_labels = None
        self.fallthrough_labels = {}
        
    
    def new_node(self, preds, stmt, kind = "stmt"):
        node = CFGNode(register_id=self.registry, stmt=stmt,preds=preds, kind=kind)
        self.cache[node.register_id] = node
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
        pred = preds
        first = True
        for stmt in statements:
            edge_labels = None
            if first and entry_label is not None:
                edge_labels = {p.register_id: entry_label for p in pred}
            pred = self.walk_stmt(stmt, pred, edge_labels=edge_labels)
            first = False
        return pred
    
    def run(self, circuit: AnalogCircuit):
        self.registry = 0
        self.cache = {}
        self.loop_stack = []
        self.edge_labels = None
        self.fallthrough_labels = {}
        self.founder = self.new_node([], "start", kind="start")
        exits = self.walk_stmt(circuit, [self.founder])
        self.last_node = self.new_node(exits, "stop", kind="stop")
        return self.cache
    
    def map_AnalogCircuit(self, model: AnalogCircuit):
        return self.walk_block(model.statements, self.preds)
    
    def map_IfElse(self, model: IfElse):
        node = self.new_node(self.preds, model.condition, kind="branch")
        then_branch = self.walk_block(model.then_branch, [node], entry_label="true")
        if model.else_branch:
            else_branch = self.walk_block(model.else_branch, [node], entry_label="false")
            return list(then_branch) + (list(else_branch))
        
        self.fallthrough_labels[node.register_id] = "false"
        
        return list(then_branch) + [node]
    
    def map_While(self, model: While):
        node = self.new_node(self.preds, model.condition, kind="branch")
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
    
    def generic_map(self, model):
        return [self.new_node(self.preds, model)]
    

