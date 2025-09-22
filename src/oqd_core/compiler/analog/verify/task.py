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

from oqd_compiler_infrastructure import RewriteRule

from oqd_core.compiler.analog.passes.analysis import analysis_canonical_hamiltonian_dim

########################################################################################
from oqd_core.interface.analog import AnalogGate

########################################################################################

__all__ = [
    "VerifyAnalogCircuitDim",
]

########################################################################################


class VerifyAnalogCircuitDim(RewriteRule):
    """
    Checks  whether hilbert space dimensions are consistent between  [`AnalogGate`][oqd_core.interface.analog.operations.AnalogGate] objects.
    and whether they match the n_qreg and n_qmode given as input.

    Args:
        model (VisitableBaseModel):
            The rule only verifies [`AnalogCircuit`][oqd_core.interface.analog.operations.AnalogCircuit] in Analog level

    Returns:
        model (VisitableBaseModel): unchanged

    Assumptions:
        All [`Operator`][oqd_core.interface.analog.operator.Operator] inside  [`AnalogCircuit`][oqd_core.interface.analog.operations.AnalogCircuit] are canonicalized
    """

    def __init__(self, n_qreg, n_qmode):
        super().__init__()
        self._dim: tuple = (n_qreg, n_qmode)

    def map_AnalogGate(self, model: AnalogGate):
        assert self._dim == analysis_canonical_hamiltonian_dim(
            model.hamiltonian
        ), "Inconsistent Hilbert space dimension between Analog Gates"
