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
from typing import Annotated, Iterable, Literal, Union, get_args, get_origin

from oqd_compiler_infrastructure import VisitableBaseModel
from pydantic import BaseModel, Field


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


class CFGStart(VisitableBaseModel):
    pass

class CFGStop(VisitableBaseModel):
    pass


class Block(BaseModel):
    """Represents one control flow node with incoming / outgoing edges and metadata."""
    
    register_id: int
    stmt: VisitableBaseModel
    preds: list[Block] = Field(default_factory=list)
    succs: list[Block] = Field(default_factory=list)
    kind: Literal["start", "stop", "branch", "stmt"] = "stmt"
    exit_nodes: list[Block] = Field(default_factory=list)
    edge_labels: dict[int, str] = Field(default_factory=dict)
    
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


class ControlFlowGraph(BaseModel):
    """Defines a Control Flow Graph (CFG) with the GraphProtocol required by DataflowAnalysis."""
    blocks: dict[int, Block]
    
    def nodes(self) -> Iterable[int]:
        return self.blocks.keys()
    
    def predecessors(self, node: int) -> Iterable[int]:
        return (pred.register_id for pred in self.blocks[node].preds)
    
    def successors(self, node: int) -> Iterable[int]:
        return (succ.register_id for succ in self.blocks[node].succs)
    
    def to_dict(self) -> dict:
        return {
            node_id: {
                "register_id": block.register_id,
                "kind": block.kind,
                "stmt": block.stmt.model_dump(),
                "preds": [p.register_id for p in block.preds],
                "succs": [s.register_id for s in block.succs],
                "exit_nodes": [e.register_id for e in block.exit_nodes],
                "edge_labels": block.edge_labels,
            }
            for node_id, block in self.blocks.items()
        }

