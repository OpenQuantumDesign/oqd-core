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
from oqd_core.compiler.atomic.canonicalize import ResolveNestedProtocol, ResolveRelativeTime
from oqd_core.compiler.atomic.math.rules import _is_constant_math
from oqd_core.compiler.atomic.verify.passes import iter_pulses
from oqd_core.interface.atomic import (
    Declaration,
    ParallelProtocol,
    Pulse,
    SerialProtocol,
    IfElse,
    While,
    AtomicCircuit,
)

PROTOCOL_TYPES = (Pulse, ParallelProtocol, SerialProtocol)


def apply_protocol_passes(stmt):
    if not all(_is_constant_math(pulse.duration) for pulse in iter_pulses(stmt)):
        return stmt
    stmt = Post(ResolveNestedProtocol())(stmt)
    stmt = Post(ResolveRelativeTime())(stmt)
    return stmt


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


def canonicalize_protocol_circuit(circuit: AtomicCircuit) -> AtomicCircuit:
    circuit.statements = [canonicalize_protocol_tree(s) for s in circuit.statements]
    return circuit

