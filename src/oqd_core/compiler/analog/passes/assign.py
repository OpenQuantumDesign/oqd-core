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

from oqd_compiler_infrastructure import Post

########################################################################################
from oqd_core.compiler.analog.rewrite.assign import InferAnalogCircuitDim
from oqd_core.interface.analog import AnalogCircuit

########################################################################################

__all__ = [
    "infer_analog_circuit_dim",
]

########################################################################################


def infer_analog_circuit_dim(model: AnalogCircuit) -> tuple[int, int]:
    """
    This pass assigns n_qreg and n_qmode in the analog circuit and then verifies the assignment

    Args:
        model (AnalogCircuit): n_qreg and n_qmode fields of [`AnalogCircuit`][oqd_core.interface.analog.operations.AnalogCircuit] are not assigned

    Returns:
        model (AnalogCircuit): n_qreg and n_qmode fields of [`AnalogCircuit`][oqd_core.interface.analog.operations.AnalogCircuit] are assigned

    Assumptions:
        All [`Operator`][oqd_core.interface.analog.operator.Operator] inside [`AnalogCircuit`][oqd_core.interface.analog.operations.AnalogCircuit] must be canonicalized
    """
    rule = InferAnalogCircuitDim()
    Post(rule)(model)
    return rule.dim or (0, 0)

