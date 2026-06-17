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

########################################################################################
from oqd_core.interface.analog import Evolve
from oqd_core.compiler.analog.error import AnalogCompilerError
from oqd_core.compiler.analog.operator.dim import operator_dim
from oqd_core.compiler.analog.cfg.walk import iter_stmt_blocks
from oqd_core.analysis.utils.control_flow import ControlFlowGraph


########################################################################################

__all__ = [
    "infer_analog_circuit_dim_cfg",
]

########################################################################################

def infer_analog_circuit_dim_cfg(cfg: ControlFlowGraph):
    """
    This pass assigns n_qreg and n_qmode in the analog circuit and then verifies the assignment

    Args:
        model (AnalogCircuit): n_qreg and n_qmode fields of [`AnalogCircuit`][oqd_core.interface.analog.operations.AnalogCircuit] are not assigned

    Returns:
        model (AnalogCircuit): n_qreg and n_qmode fields of [`AnalogCircuit`][oqd_core.interface.analog.operations.AnalogCircuit] are assigned

    Assumptions:
        All [`Operator`][oqd_core.interface.analog.operator.Operator] inside [`AnalogCircuit`][oqd_core.interface.analog.operations.AnalogCircuit] must be canonicalized
    """
    dim = None
    for _, block in iter_stmt_blocks(cfg):
        if not isinstance(block.stmt, Evolve):
            continue
        d = operator_dim(block.stmt.hamiltonian)
        if dim is None:
            dim = d
        elif dim != d:
            raise AnalogCompilerError("Inconsistent Hilbert space dimensions between Evolve statements")
    return dim or (0, 0)

