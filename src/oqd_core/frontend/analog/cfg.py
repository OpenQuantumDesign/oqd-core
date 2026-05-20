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

from oqd_core.interface.analog import (
    AnalogCircuit,
    Bool,
    Break,
    Continue,
    IfElse,
    While,
)


class CFGNode:
    def __init__(self, register_id, stmt,  preds = None, kind = "stmt"):
        self.register_id = register_id
        self.stmt = stmt
        self.preds = list(preds) if preds is not None else []
        self.succs = []
        self.kind = kind
        self.exit_nodes = []
        self.edge_labels = {}
    
    def add_succ(self, succ, label=None):
        if succ not in self.succs:
            self.succs.append(succ)
        if label is not None:
            self.edge_labels[succ.register_id] = label

    def add_pred(self, pred, label=None):
        if pred not in self.preds:
            self.preds.append(pred)
        pred.add_succ(self, label=label)

    def add_preds(self, preds, label=None):
        for pred in preds:
            self.add_pred(pred, label=label)
    
    def to_dict(self):
        if isinstance(self.stmt, str):
            stmt_repr = self.stmt
        elif hasattr(self.stmt, "class_"):
            stmt_repr = self.stmt.class_
        else:
            stmt_repr = type(self.stmt).__name__
        return {
            "id": self.register_id,
            "kind": self.kind,
            "stmt": stmt_repr,
            "preds": [p.register_id for p in self.preds],
            "succs": [c.register_id for c in self.succs],
            "edges": [
                {"to": c.register_id, "label": self.edge_labels.get(c.register_id)}
                for c in self.succs
            ],
            "exit_nodes": [n.register_id for n in self.exit_nodes],
        }



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
    

class SCCAnalysis:
    def __init__(self, cfg):
        self.cfg = cfg
        self.time = 0
        self.disc = {nid: -1 for nid in cfg}
        self.low = {nid: -1 for nid in cfg}
        self.on_stack = {nid: False for nid in cfg}
        self.stack = []
        self.sccs = []
    
    def dfs(self, u):
        self.disc[u] = self.time
        self.low[u] = self.time
        self.time += 1
        self.stack.append(u)
        self.on_stack[u] = True
        for succ in self.cfg[u].succs:
            v = succ.register_id
            if self.disc[v] == -1:
                self.dfs(v)
                self.low[u] = min(self.low[u], self.low[v])
            elif self.on_stack[v]:
                self.low[u] = min(self.low[u], self.disc[v])
        if self.low[u] == self.disc[u]:
            comp = set()
            while True:
                w = self.stack.pop()
                self.on_stack[w] = False
                comp.add(w)
                if w == u:
                    break
            self.sccs.append(comp)

    def run(self):
        for nid in self.cfg:
            if self.disc[nid] == -1:
                self.dfs(nid)
        return self.sccs
    
    def edge_feasible(self, src, dst_id):
        if src.kind == "branch" and isinstance(src.stmt, Bool):
            label = src.edge_labels.get(dst_id)
            if src.stmt.value is True and label == "false":
                return False
            if src.stmt.value is False and label == "true":
                return False
        return True
    
    def infinite_loop_check(self):
        sccs = self.run()
        stop_ids = {nid for nid, node in self.cfg.items() if node.kind == "stop"}
        for comp in sccs:
            has_cycle = len(comp) > 1 or any(
                succ.register_id == nid
                for nid in comp
                for succ in self.cfg[nid].succs
            )
            if not has_cycle:
                continue

            has_exit = any(
                (succ.register_id not in comp) and self.edge_feasible(self.cfg[nid], succ.register_id)
                for nid in comp
                for succ in self.cfg[nid].succs
            )
            
            stack = list(comp)
            seen = set(comp)
            can_reach_stop = False
            while stack:
                curr = stack.pop()
                if curr in stop_ids:
                    can_reach_stop = True
                    break
                for succ in self.cfg[curr].succs:
                    sid = succ.register_id
                    if not self.edge_feasible(self.cfg[curr], sid):
                        continue
                    if sid not in seen:
                        seen.add(sid)
                        stack.append(sid)

            if not has_exit and not can_reach_stop:
                raise TypeError(
                    f"Infinite loop detected in circuit: {sorted(comp)}"
                )
