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

from typing import Annotated, List, Optional, Union

from oqd_compiler_infrastructure import TypeReflectBaseModel
from pydantic import Discriminator
from pydantic.types import NonNegativeInt

from oqd_core.interface.analog.operator import OperatorSubTypes

__all__ = [
    "AnalogCircuit",
    "AnalogGate",
    "AnalogOperation",
    "Evolve",
    "Measure",
    "Initialize",
]


########################################################################################


class AnalogOperation(TypeReflectBaseModel):
    """
    Class representing an analog operation applied to the quantum system
    """

    pass


########################################################################################


class AnalogGate(TypeReflectBaseModel):
    """
    Class representing an analog gate composed of Hamiltonian terms and dissipation terms

    Attributes:
        hamiltonian (Operator): Hamiltonian terms of the gate
    """

    hamiltonian: OperatorSubTypes


class Evolve(AnalogOperation):
    """
    Class representing an evolution by an analog gate in the analog circuit

    Attributes:
        duration (float): Duration of the evolution
        gate (AnalogGate): Analog gate to evolve by
    """

    duration: float
    gate: AnalogGate


########################################################################################


class Measure(AnalogOperation):
    """
    Class representing a measurement in the analog circuit
    """

    targets: Optional[List[int]] = None


class Initialize(AnalogOperation):
    """
    Class representing a initialization in the analog circuit
    """

    targets: Optional[List[int]] = None


########################################################################################

"""
Union of classes
"""
AnalogOperationSubTypes = Annotated[
    Union[Measure, Evolve, Initialize], Discriminator(discriminator="class_")
]


########################################################################################


class AnalogCircuit(AnalogOperation):
    """
    Class representing a quantum information experiment represented in terms of analog operations.

    Attributes:
        sequence (List[Union[Measure, Evolve, Initialize]]): Sequence of statements, including initialize, evolve, measure

    """

    sequence: List[AnalogOperationSubTypes] = []

    n_qreg: Union[NonNegativeInt, None] = None
    n_qmode: Union[NonNegativeInt, None] = None

    def evolve(self, gate: AnalogGate, duration: float):
        self.sequence.append(Evolve(duration=duration, gate=gate))

    def initialize(self, targets=None):
        self.sequence.append(Initialize(targets=targets))

    def measure(self, targets=None):
        self.sequence.append(Measure(targets=targets))
