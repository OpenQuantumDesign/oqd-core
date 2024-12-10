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

from typing import List, Literal, Union
from pydantic.types import NonNegativeInt

# %%
from oqd_compiler_infrastructure import VisitableBaseModel, TypeReflectBaseModel

from oqd_core.interface.analog.operator import OperatorSubtypes


__all__ = [
    "AnalogCircuit",
    "AnalogGate",
    "AnalogOperation",
    "Evolve",
    "Measure",
    "Initialize",
]


########################################################################################


class AnalogGate(TypeReflectBaseModel):
    """
    Class representing an analog gate composed of Hamiltonian terms and dissipation terms

    Attributes:
        hamiltonian (Operator): Hamiltonian terms of the gate
    """

    hamiltonian: OperatorSubtypes


# %%
class AnalogOperation(VisitableBaseModel):
    """
    Class representing an analog operation applied to the quantum system
    """

    pass


class Evolve(AnalogOperation):
    """
    Class representing an evolution by an analog gate in the analog circuit

    Attributes:
        duration (float): Duration of the evolution
        gate (AnalogGate): Analog gate to evolve by
    """

    key: Literal["evolve"] = "evolve"
    duration: float
    gate: Union[AnalogGate, str]


class Measure(AnalogOperation):
    """
    Class representing a measurement in the analog circuit
    """

    key: Literal["measure"] = "measure"


class Initialize(AnalogOperation):
    """
    Class representing a initialization in the analog circuit
    """

    key: Literal["initialize"] = "initialize"


"""
Union of classes 
"""
Statement = Union[Measure, Evolve, Initialize]


class AnalogCircuit(AnalogOperation):
    """
    Class representing a quantum information experiment represented in terms of analog operations.

    Attributes:
        sequence (List[Union[Measure, Evolve, Initialize]]): Sequence of statements, including initialize, evolve, measure

    """

    sequence: List[Statement] = []

    n_qreg: Union[NonNegativeInt, None] = None
    n_qmode: Union[NonNegativeInt, None] = None

    def evolve(self, gate: AnalogGate, duration: float):
        self.sequence.append(Evolve(duration=duration, gate=gate))

    def initialize(self):
        self.sequence.append(Initialize())

    def measure(self):
        self.sequence.append(Measure())
