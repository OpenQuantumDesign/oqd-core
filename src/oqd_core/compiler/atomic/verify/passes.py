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
from collections.abc import Iterator
from oqd_core.analysis.atomic.symbol_table import AtomicSymbolTable, target_dim
from oqd_core.analysis.utils.control_flow import ControlFlowGraph
from oqd_core.compiler.atomic.cfg_passes.walk import iter_stmt_blocks
from oqd_core.interface.atomic import Declaration, ParallelProtocol, Pulse, SerialProtocol

__all__ = [
    "verify_pulse_target_dim",
]


def iter_pulses(stmt):
    if isinstance(stmt, Pulse):
        yield stmt
    elif isinstance(stmt, Declaration) and isinstance(stmt.value, Pulse):
        yield stmt.value
    elif isinstance(stmt, (ParallelProtocol, SerialProtocol)):
        for child in stmt.pulses:
            yield from iter_pulses(child)


def iter_pulse_targets(stmt) -> Iterator:
    for pulse in iter_pulses(stmt):
        yield pulse.target


def verify_pulse_target_dim(cfg: ControlFlowGraph, symbol_table: AtomicSymbolTable):
    """
    Check pulse targets are in range at each use.
    """
    for node_id, block in iter_stmt_blocks(cfg):
        stmt = block.stmt
        env = symbol_table.in_env[node_id]
        for target in iter_pulse_targets(stmt):
            target_dim(target, env)
    return cfg

