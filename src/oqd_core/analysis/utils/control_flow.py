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
from types import UnionType
from typing import Annotated, Iterable, List, Union, get_args, get_origin

from oqd_compiler_infrastructure import RewriteRule, VisitableBaseModel
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


class Block(BaseModel):
    """Represents one control flow node with incoming / outgoing edges and metadata."""
    
    register_id: int
    stmts: List[VisitableBaseModel] = Field(default_factory=list)
    preds: List[int] = Field(default_factory=list)
    succs: List[int] = Field(default_factory=list)
    exit_nodes: List[int] = Field(default_factory=list)
    edge_labels: dict[int, str] = Field(default_factory=dict)
    
    def add_succ(self, succ: int, label: str | None = None) -> None:
        if succ not in self.succs:
            self.succs.append(succ)
        if label is not None:
            self.edge_labels[succ] = label

    def add_pred(self, pred: int) -> None:
        if pred not in self.preds:
            self.preds.append(pred)

    def add_preds(self, preds: Iterable[int]) -> None:
        for pred in preds:
            self.add_pred(pred)


class ControlFlowGraph(BaseModel):
    """Defines a Control Flow Graph (CFG) with the GraphProtocol required by DataflowAnalysis."""
    blocks: dict[int, Block]
    
    def nodes(self) -> Iterable[int]:
        return self.blocks.keys()
    
    def predecessors(self, node: int) -> Iterable[int]:
        return self.blocks[node].preds
    
    def successors(self, node: int) -> Iterable[int]:
        return self.blocks[node].succs
    
    def to_dict(self) -> dict:
        return {
            node_id: {
                "register_id": block.register_id,
                "stmts": [stmt.model_dump() for stmt in block.stmts],
                "preds": block.preds,
                "succs": block.succs,
                "exit_nodes": block.exit_nodes,
                "edge_labels": block.edge_labels,
            }
            for node_id, block in self.blocks.items()
        }


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
    
