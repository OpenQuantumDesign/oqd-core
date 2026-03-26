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
from typing import List, Union

from oqd_compiler_infrastructure import TypeReflectBaseModel
from pydantic.types import NonNegativeInt

from .expression import Expr, Access, Identifier
from .bool import BoolExprSubtypes, CastBool
from .math import CastMathExpr, MathExprSubtypes
from .operator import OperatorSubtypes, CastOperator

########################################################################################
__all__ = [
    "AnalogCircuit",
    "Evolve",
    "Measure",
    "Initialize",
    "QuantumBit",
    "QuantumRegister",
    "Declaration",
    "MyList",
    "Extract",
    "IfElse",
    "While",
    "ModeBit",
    "ModeRegister",
    "Break", 
    "Continue",
    "AnalogExprSubtypes"
]

########################################################################################

class MyList(Expr):
    values: List[AnalogExprSubtypes]

class Declaration(TypeReflectBaseModel):
    name: Identifier
    value: AnalogExprSubtypes

class QuantumBit(TypeReflectBaseModel):
    access: Access
    index: NonNegativeInt


class QuantumRegister(TypeReflectBaseModel):
    size: NonNegativeInt

class ModeBit(TypeReflectBaseModel):
    access: Access
    index: NonNegativeInt


class ModeRegister(TypeReflectBaseModel):
    size: NonNegativeInt


class Extract(TypeReflectBaseModel):
    access: Access
    index: NonNegativeInt


class Evolve(TypeReflectBaseModel):
    """
    Class representing an evolution by an analog gate in the analog circuit

    Attributes:
        hamiltonian (Expr): Function to evolve by
        duration (Expr): Duration of the evolution
        targets (Expr): Indices and Quantum objects on which to apply the Hamiltonian
    """

    hamiltonian: CastOperator
    duration: CastMathExpr
    targets: AnalogExprSubtypes


class Measure(TypeReflectBaseModel):
    """
    Class representing a measurement in the analog circuit
    """
    targets: AnalogExprSubtypes


class Initialize(TypeReflectBaseModel):
    """
    Class representing a initialization in the analog circuit
    """
    targets: AnalogExprSubtypes


class IfElse(TypeReflectBaseModel):
    """
    Class representing a conditional branch in the analog circuit
    """
    condition: CastBool
    then_branch: List[Statement] = []
    else_branch: List[Statement] = []
    
    
class While(TypeReflectBaseModel):
    """
    Class representing a while loop in the analog circuit
    """
    condition : CastBool
    body: List[Statement] = []


class Break(TypeReflectBaseModel):
    """
    Class representing a break statement to exit the innermost loop
    """
    pass


class Continue(TypeReflectBaseModel):
    """
    Class representing a continue statement to jump to the next loop iteration
    """
    pass


"""
Union of classes 
"""

AnalogExprSubtypes = Union[
    MathExprSubtypes,
    OperatorSubtypes,
    BoolExprSubtypes,
    MyList,
    Access,
    QuantumBit,
    QuantumRegister,
    ModeBit,
    ModeRegister,
    Extract,
]

Statement = Union[Declaration, Measure, Evolve, Initialize, IfElse, While, Break, Continue]


class AnalogCircuit(TypeReflectBaseModel):
    """
    Class representing a quantum information experiment represented in terms of analog operations.

    Attributes:
        statements (List[Union[Measure, Evolve, Initialize]]): List of statements, including initialize, evolve, measure

    """

    statements: List[Statement] = []

    def evolve(self, hamiltonian: CastOperator, duration: CastMathExpr, targets: AnalogExprSubtypes):
        self.statements.append(Evolve(hamiltonian=hamiltonian, duration=duration, targets=targets))

    def initialize(self, targets: AnalogExprSubtypes):
        self.statements.append(Initialize(targets=targets))

    def measure(self, targets: AnalogExprSubtypes):
        self.statements.append(Measure(targets=targets))

