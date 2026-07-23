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

from oqd_core.compiler.analog.cfg_passes.walk import canonicalize_math_cfg, canonicalize_operators_cfg
from oqd_core.compiler.analog.verify.passes import verify_hamiltonian_target_dim, verify_register_access_dim
from oqd_core.analysis.analog.symbol_table import AnalogSymbolTable
from oqd_core.analysis.utils.control_flow import ControlFlowGraph
from oqd_core.interface.analog import AnalogCircuit

########################################################################################

__all__ = [ "compile_analog_circuit" ]

########################################################################################

def compile_analog_circuit(circuit: AnalogCircuit, cfg: ControlFlowGraph, symbol_table: AnalogSymbolTable) \
    -> tuple[AnalogCircuit, ControlFlowGraph]:
    
    canonicalize_operators_cfg(cfg)
    canonicalize_math_cfg(cfg)
    
    verify_register_access_dim(cfg, symbol_table)
    verify_hamiltonian_target_dim(cfg, symbol_table)
    
    return circuit, cfg

