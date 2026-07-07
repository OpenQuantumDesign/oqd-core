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
from oqd_compiler_infrastructure import Post
from oqd_core.analysis.utils.control_flow import ControlFlowGraph
from oqd_core.compiler.atomic.canonicalize import ResolveNestedProtocol, ResolveRelativeTime
from oqd_core.compiler.atomic.cfg_passes.walk import iter_stmt_blocks
from oqd_core.interface.atomic import (
    AtomicCircuit,
    Declaration,
    ParallelProtocol,
    Pulse,
    SerialProtocol,
    IfElse,
    While,
)

PROTOCOL_TYPES = (Pulse, ParallelProtocol, SerialProtocol)


def apply_protocol_passes(stmt):
    stmt = Post(ResolveNestedProtocol())(stmt)
    return Post(ResolveRelativeTime())(stmt)


def canonicalize_protocol_tree(stmt):
    if isinstance(stmt, IfElse):
        stmt.then_branch = [canonicalize_protocol_tree(s) for s in stmt.then_branch]
        stmt.else_branch = [canonicalize_protocol_tree(s) for s in stmt.else_branch]
        return stmt
    if isinstance(stmt, While):
        stmt.body = [canonicalize_protocol_tree(s) for s in stmt.body]
        return stmt
    if isinstance(stmt, Declaration) and isinstance(stmt.value, PROTOCOL_TYPES):
        stmt.value = apply_protocol_passes(stmt.value)
        return stmt
    if isinstance(stmt, PROTOCOL_TYPES):
        return apply_protocol_passes(stmt)
    return stmt


def canonicalize_protocol_cfg(cfg: ControlFlowGraph, model: AtomicCircuit) -> ControlFlowGraph:
    old_top = list(model.statements)
    model.statements = [canonicalize_protocol_tree(s) for s in model.statements]
    
    for old, new in zip(old_top, model.statements):
        if old is not new:
            for _, block in iter_stmt_blocks(cfg):
                if block.stmt is old:
                    block.stmt = new
    return cfg

