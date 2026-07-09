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

from oqd_core.analysis.atomic.symbol_table import AtomicSymbolTable
from oqd_core.analysis.utils.control_flow import ControlFlowGraph
from oqd_core.compiler.atomic.cfg_passes.walk import canonicalize_declarations_cfg
from oqd_core.compiler.atomic.cfg_passes.protocol import canonicalize_protocol_cfg
from oqd_core.compiler.atomic.verify.passes import verify_pulse_target_dim

__all__ = ["compile_atomic_circuit"]


def compile_atomic_circuit(
    cfg: ControlFlowGraph,
    symbol_table: AtomicSymbolTable,
):
    canonicalize_declarations_cfg(cfg)
    canonicalize_protocol_cfg(cfg)
    verify_pulse_target_dim(cfg, symbol_table)
    return cfg

