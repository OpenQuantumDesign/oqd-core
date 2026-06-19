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


from oqd_core.analysis.utils.control_flow import Block, ControlFlowGraph
from oqd_core.compiler.atomic.math.passes import canonicalize_math_expr
from oqd_core.interface.atomic import (
    AtomicList,
    Beam,
    Declaration,
    ParallelProtocol,
    Pulse,
    SerialProtocol,
)
from oqd_core.interface.atomic.expr import AtomicExpr

def iter_stmt_blocks(cfg: ControlFlowGraph):
    for node_id, block in cfg.blocks.items():
        if isinstance(block.stmt, str):
            continue
        yield node_id, block

def canonicalize_expr(expr):
    if isinstance(expr, AtomicExpr):
        return canonicalize_math_expr(expr)
    return expr

def canonicalize_beam(beam: Beam) -> Beam:
    beam.frequency = canonicalize_expr(beam.frequency)
    beam.rabi = canonicalize_expr(beam.rabi)
    beam.phase = canonicalize_expr(beam.phase)
    beam.polarization = canonicalize_expr(beam.polarization)
    beam.wavevector = canonicalize_expr(beam.wavevector)
    return beam

def canonicalize_atomic_list(values: AtomicList) -> AtomicList:
    values.values = [canonicalize_expr(v) for v in values.values]
    return values

def canonicalize_math_block(block: Block):
    stmt = block.stmt
    
    if block.kind == "branch":
        block.stmt = canonicalize_expr(stmt)
        return
    
    if isinstance(stmt, Pulse):
        stmt.duration = canonicalize_expr(stmt.duration)
        stmt.target = canonicalize_expr(stmt.target)
        stmt.measured = canonicalize_expr(stmt.measured)
        if isinstance(stmt.beam, Beam):
            stmt.beam = canonicalize_beam(stmt.beam)
        else:
            stmt.beam = canonicalize_expr(stmt.beam)
        return
    
    if isinstance(stmt, Declaration):
        if isinstance(stmt.value, Beam):
            stmt.value = canonicalize_beam(stmt.value)
        elif isinstance(stmt.value, AtomicList):
            stmt.value = canonicalize_atomic_list(stmt.value)
        else:
            stmt.value = canonicalize_expr(stmt.value)
        return
    
    if isinstance(stmt, (ParallelProtocol, SerialProtocol)):
        return


def canonicalize_math_cfg(cfg: ControlFlowGraph):
    for _, block in iter_stmt_blocks(cfg):
        canonicalize_math_block(block)
    return cfg


