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

from oqd_core.analysis.utils.control_flow import ControlFlowGraph, Block
from oqd_core.compiler.analog.math.passes import canonicalize_math_expr
from oqd_core.interface.analog import Declaration, Evolve
from oqd_core.interface.analog.expr import MathExpr


def iter_stmt_blocks(cfg: ControlFlowGraph):
    for node_id, block in cfg.blocks.items():
        if isinstance(block.stmt, str):
            continue
        yield node_id, block


def canonicalize_math_block(block: Block):
    stmt = block.stmt
    
    if block.kind == "branch":
        if isinstance(stmt, MathExpr):
            block.stmt = canonicalize_math_expr(stmt)
        return
    
    if isinstance(stmt, Evolve):
        stmt.duration = canonicalize_math_expr(stmt.duration)
    elif isinstance(stmt, Declaration):
        if isinstance(stmt.value, MathExpr):
            stmt.value = canonicalize_math_expr(stmt.value)


def canonicalize_math_cfg(cfg: ControlFlowGraph):
    for _, block in iter_stmt_blocks(cfg):
        canonicalize_math_block(block)
    return cfg
            

