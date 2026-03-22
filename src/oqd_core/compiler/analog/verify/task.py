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

from oqd_core.backend.metric import Expectation
from oqd_core.compiler.analog.passes.analysis import analysis_canonical_hamiltonian_dim

########################################################################################

__all__ = [
    "VerifyAnalogCircuitDim",
    "VerifyAnalogArgsDim",
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


class VerifyAnalogArgsDim(RewriteRule):
    """
    Checks whether hilbert space dimensions are consistent between Expectation in args
    and whether they match the n_qreg and n_qmode given as input.

    Args:
        model (VisitableBaseModel):
            The rule only verfies Expectation inside TaskArgsAnalog in Analog layer

    Returns:
        model (VisitableBaseModel): unchanged

    Assumptions:
        All [`Operator`][oqd_core.interface.analog.operator.Operator] inside  [`AnalogCircuit`][oqd_core.interface.analog.operations.AnalogCircuit] are canonicalized
    """

    def __init__(self, n_qreg, n_qmode):
        super().__init__()
        self._dim: tuple = (n_qreg, n_qmode)

    def map_Expectation(self, model: Expectation):
        assert self._dim == analysis_canonical_hamiltonian_dim(model.operator), (
            "Inconsistent Hilbert space dimension in Expectation metric"
        )
