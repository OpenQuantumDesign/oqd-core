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
from pydantic import (
    NonNegativeInt,
)
from .expression import Expr, Access, Identifier
from .bool import BoolExprSubtypes, CastBool
from .math import CastMathExpr, MathExprSubtypes

########################################################################################

__all__ = [
    "IonQubit",
    "IonRegister",
    "Declaration",
    "MyList",
    "Extract",
    "Statement",
    "IfElse",
    "While",
    "Break", 
    "Continue",
    "Beam",
    "Pulse",
    "ParallelProtocol",
    "AtomicExprSubtypes",
]

########################################################################################

class MyList(Expr):
    values: List[AtomicExprSubtypes]
    
    
class IonQubit(TypeReflectBaseModel):
    access: Access
    index: NonNegativeInt

class IonRegister(TypeReflectBaseModel):
    size: NonNegativeInt
    
class Declaration(TypeReflectBaseModel):
    name: Identifier
    value: AtomicExprSubtypes

class Extract(Expr):
    access: Access
    index: NonNegativeInt

########################################################################################

class ParallelProtocol(TypeReflectBaseModel):
    pulses: List[Pulse]

class Pulse(TypeReflectBaseModel):
    """
    Class representing the application of the beam for some duration.

    Attributes:
        beam: Optical channel/beam to turn on.
        duration: Period of time to turn the optical channel on for.
        target: Target ion of the beam.
        measured: Boolean that tracks if the pulse has been measured.
    """
    duration: CastMathExpr
    target: AtomicExprSubtypes
    beam: Union[Access, Beam]
    measured: CastBool
    

class Beam(TypeReflectBaseModel):
    """
    Class representing a referenced optical channel/beam for the trapped-ion device.

    Attributes:
        rabi: Rabi frequency of the referenced transition driven by the beam.
        phase: Phase relative to the ion's clock.
        polarization: Polarization of the beam.
        wavevector: Wavevector of the beam.
    """
    frequency: CastMathExpr
    rabi: CastMathExpr
    phase: CastMathExpr
    polarization: Union[MyList, Access]
    wavevector: Union[MyList, Access]
    

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
    condition: CastBool
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

AtomicExprSubtypes = Union[
    MathExprSubtypes,
    BoolExprSubtypes,
    MyList,
    Access,
    Beam
]

Statement = Union[Declaration, IfElse, While, Break, Continue, Pulse, ParallelProtocol]
