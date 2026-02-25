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
from typing import List, Union, Annotated

from oqd_compiler_infrastructure import TypeReflectBaseModel
from pydantic.types import NonNegativeInt
from pydantic import AfterValidator
from oqd_core.interface.bool import BoolExprSubtypes

########################################################################################
from .operator import OperatorSubtypes

########################################################################################
__all__ = [
    "AnalogCircuit",
    "AnalogGate",
    "AnalogOperation",
    "Evolve",
    "Measure",
    "Initialize",
    "QuantumBit",
    "QuantumRegister",
    "Declaration",
    "MyList",
    "Access",
    "AtomicTypes",
    "Identifier",
    "IfElse",
    "While"
]

########################################################################################

def _is_varname(value: str) -> str:
    if not value.isidentifier():
        raise ValueError(f"{value!r} is not a valid identifier")
    return value


Identifier = Annotated[str, AfterValidator(_is_varname)]

class QuantumBit(TypeReflectBaseModel):
    name: str
    index: NonNegativeInt


class QuantumRegister(TypeReflectBaseModel):
    size: NonNegativeInt


class Access(TypeReflectBaseModel):
    name: Identifier


class MyList(TypeReflectBaseModel):
    values: List[AtomicTypes]


AtomicTypes = Union[QuantumBit, QuantumRegister, MyList, Access]


class Declaration(TypeReflectBaseModel):
    name: Identifier
    value: Union[AtomicTypes, BoolExprSubtypes, OperatorSubtypes]


class Evolve(TypeReflectBaseModel):
    """
    Class representing an evolution by an analog gate in the analog circuit

    Attributes:
        hamiltonian (OperatorSubtypes): Function to evolve by
        duration (float): Duration of the evolution
        targets (AtomicTypes): Indices and Quanutm objects on which to apply the Hamiltonian
    """

    hamiltonian: OperatorSubtypes
    duration: float
    targets: AtomicTypes


class Measure(TypeReflectBaseModel):
    """
    Class representing a measurement in the analog circuit
    """

    pass


class Initialize(TypeReflectBaseModel):
    """
    Class representing a initialization in the analog circuit
    """

    pass

class IfElse(TypeReflectBaseModel):
    """
    Class representing a conditional branch in the analog circuit
    """
    condition: BoolExprSubtypes
    then_branch: List[Statement] = []
    else_branch: List[Statement] = []
    
class While(TypeReflectBaseModel):
    """
    Class representing a while loop in the analog circuit
    """
    condition : BoolExprSubtypes
    body: List[Statement] = []

"""
Union of classes 
"""
Statement = Union[Declaration, Measure, Evolve, Initialize, IfElse, While]


class AnalogCircuit(TypeReflectBaseModel):
    """
    Class representing a quantum information experiment represented in terms of analog operations.

    Attributes:
        sequence (List[Union[Measure, Evolve, Initialize]]): Sequence of statements, including initialize, evolve, measure

    """

    sequence: List[Statement] = []

    def evolve(self, hamiltonian: OperatorSubtypes, duration: float, targets: AtomicTypes):
        self.sequence.append(Evolve(hamiltonian=hamiltonian, duration=duration, targets=targets))

    def initialize(self):
        self.sequence.append(Initialize())

    def measure(self):
        self.sequence.append(Measure())
