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
    def __init__(self, blocks: dict[int, Block]):
        self.blocks = blocks
    def nodes(self) -> Iterable[int]:
        return self.blocks.keys()
    def predecessors(self, node: int) -> Iterable[int]:
        return (pred.register_id for pred in self.blocks[node].preds)
    def successors(self, node: int) -> Iterable[int]:
        return (succ.register_id for succ in self.blocks[node].succs)


class Block:
    """Represents one control flow node with incoming / outgoing edges and metadata."""
    def __init__(self, register_id: int, stmt: object,  preds: Iterable[Block] | None = None, \
        kind: str = "stmt", scope: int = 0) -> None:
        self.register_id = register_id
        self.stmt = stmt
        self.preds = list(preds) if preds is not None else []
        self.succs = []
        self.kind = kind
        self.scope = scope
        self.exit_nodes = []
        self.edge_labels = {}
    
    def add_succ(self, succ: Block, label: str | None = None) -> None:
        if succ not in self.succs:
            self.succs.append(succ)
        if label is not None:
            self.edge_labels[succ.register_id] = label

    def add_pred(self, pred: Block, label: str | None = None) -> None:
        if pred not in self.preds:
            self.preds.append(pred)
        pred.add_succ(self, label=label)

    def add_preds(self, preds: Iterable[Block], label: str | None = None) -> None:
        for pred in preds:
            self.add_pred(pred, label=label)
    
    def to_dict(self) -> dict[str, object]:
        if isinstance(self.stmt, str):
            stmt_repr = self.stmt
        elif hasattr(self.stmt, "class_"):
            stmt_repr = self.stmt.class_
        else:
            stmt_repr = type(self.stmt).__name__
        return {
            "id": self.register_id,
            "kind": self.kind,
            "scope": self.scope,
            "stmt": stmt_repr,
            "preds": [p.register_id for p in self.preds],
            "succs": [c.register_id for c in self.succs],
            "edges": [
                {"to": c.register_id, "label": self.edge_labels.get(c.register_id)}
                for c in self.succs
            ],
            "exit_nodes": [n.register_id for n in self.exit_nodes],
        }

