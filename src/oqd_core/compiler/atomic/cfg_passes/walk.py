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


from oqd_core.analysis.utils.control_flow import ControlFlowGraph, CFGStart, CFGStop
from oqd_core.compiler.atomic.math.passes import canonicalize_math_expr
from oqd_core.interface.atomic import (
    Beam,
    Pulse,
    Bool,
    Declaration,
)
from oqd_core.interface.atomic.expr import MathExpr

def iter_stmt_blocks(cfg: ControlFlowGraph):
    for node_id, block in cfg.blocks.items():
        if isinstance(block.stmt, (CFGStart, CFGStop)):
            continue
        yield node_id, block

def canonicalize_scalar_expr(expr):
    if isinstance(expr, Bool):
        return expr
    return canonicalize_math_expr(expr)

def canonicalize_beam(beam: Beam) -> Beam:
    beam.frequency = canonicalize_scalar_expr(beam.frequency)
    beam.rabi = canonicalize_scalar_expr(beam.rabi)
    beam.phase = canonicalize_scalar_expr(beam.phase)
    beam.polarization = canonicalize_scalar_expr(beam.polarization)
    beam.wavevector = canonicalize_scalar_expr(beam.wavevector)
    return beam

def canonicalize_pulse(pulse: Pulse) -> Pulse:
    pulse.duration = canonicalize_scalar_expr(pulse.duration)
    if isinstance(pulse.beam, Beam):
        pulse.beam = canonicalize_beam(pulse.beam)
    return pulse

def canonicalize_declarations(expr):
    if isinstance(expr, Beam):
        return canonicalize_beam(expr)
    if isinstance(expr, Pulse):
        return canonicalize_pulse(expr)
    if isinstance(expr, MathExpr):
        return canonicalize_math_expr(expr)
    return expr
    

def canonicalize_declarations_cfg(cfg: ControlFlowGraph) -> ControlFlowGraph:

    for _, block in iter_stmt_blocks(cfg):
        stmt = block.stmt
        if block.kind == "branch":
            if isinstance(stmt, MathExpr):
                block.stmt = canonicalize_math_expr(stmt)
            return
        
        if isinstance(stmt, Declaration):
            stmt.value = canonicalize_declarations(stmt.value)

    return cfg

