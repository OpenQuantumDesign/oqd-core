# Copyright 2024 Open Quantum Design

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

from oqd_compiler_infrastructure import RewriteRule

########################################################################################

from oqd_core.interface.analog import AnalogCircuit
from oqd_core.compiler.analog.passes.analysis import analysis_canonical_hamiltonian_dim

########################################################################################

__all__ = [
    "AssignAnalogIRDim",
]

########################################################################################


class AssignAnalogIRDim(RewriteRule):
    """
    RewriteRule which gets the dimensions from analysis pass
    analysis_canonical_hamiltonian_dim and then inserts the dimension in the Analog IR

    Args:
        model (VisitableBaseModel): The rule only modifies [`AnalogCircuit`][oqd_core.interface.analog.operations.AnalogCircuit] in Analog level

    Returns:
        model  (VisitableBaseModel):

    Assumptions:
        - All [`Operator`][oqd_core.interface.analog.operator.Operator] inside  [`AnalogCircuit`][oqd_core.interface.analog.operations.AnalogCircuit] must be in canonical form
    """

    def __init__(self):
        super().__init__()
        self.dim: Union[tuple, None] = None

    def map_AnalogCircuit(self, model: AnalogCircuit):
        model.n_qreg = self.dim[0]
        model.n_qmode = self.dim[1]
        return model

    def map_AnalogGate(self, model):
        if self.dim is None:
            self.dim = analysis_canonical_hamiltonian_dim(model.hamiltonian)
