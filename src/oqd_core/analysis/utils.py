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

from types import UnionType
from typing import Annotated, Iterable, Union, get_args, get_origin

from oqd_compiler_infrastructure.dataflow import GraphProtocol


def alias_types(alias: object) -> tuple[type, ...]:
    """Flatten `Annotated`/`Union` aliases into a tuple of concrete Python types."""
    origin = get_origin(alias)
    if origin is Annotated:
        return alias_types(get_args(alias)[0])
    
    if origin in (Union, UnionType):
        out: list[type] = []
        for arg in get_args(alias):
            out.extend(alias_types(arg))
        return tuple(dict.fromkeys(out))
    
    if isinstance(alias, type):
        return (alias,)
    return ()


class ControlFlowGraph(GraphProtocol[int]):
    """Defines a Control Flow Graph (CFG) with the GraphProtocol required by DataflowAnalysis."""
    def __init__(self, cfg_nodes: CFGNode):
        self.cfg_nodes = cfg_nodes
    def nodes(self) -> Iterable[int]:
        return self.cfg_nodes.keys()
    def predecessors(self, node: int) -> Iterable[int]:
        return (pred.register_id for pred in self.cfg_nodes[node].preds)
    def successors(self, node: int) -> Iterable[int]:
        return (succ.register_id for succ in self.cfg_nodes[node].succs)


class CFGNode:
    """Represents one control flow node with incoming / outgoing edges and metadata."""
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


class SCCAnalysis:
    """
    Tarjan's algorithm to identify strongly connected components (SCCs)
    of the CFG and check for infinite loops in the program.
    """
    def __init__(self, graph: ControlFlowGraph):
        self.cfg = graph.cfg_nodes
        self.time = 0
        self.disc = {nid: -1 for nid in self.cfg}
        self.low = {nid: -1 for nid in self.cfg}
        self.on_stack = {nid: False for nid in self.cfg}
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
        if src.kind == "branch":
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

