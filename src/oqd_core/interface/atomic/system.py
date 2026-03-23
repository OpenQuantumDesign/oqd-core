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
from typing import Annotated, List, Literal, Union

from oqd_compiler_infrastructure import TypeReflectBaseModel
from pydantic import (
    AfterValidator,
    NonNegativeFloat,
    NonNegativeInt,
)
from .expression import Expr

########################################################################################

__all__ = [
    "IonQubit",
    "IonRegister",
    "Declaration",
    "MyList",
    "Access",
    "Extract",
    "Identifier",
    "Statement",
    "IfElse",
    "While",
    "Break", 
    "Continue",
    "Beam",
    "Pulse",
    "ParallelProtocol",
]

########################################################################################


def is_halfint(v: float) -> bool:
    """
    Function that verifies a number is an integer or half-integer.

    Args:
        v: Number to verify.
    """
    if not (v * 2).is_integer():
        raise ValueError()
    return v


def _is_varname(value: str) -> str:
    if not value.isidentifier():
        raise ValueError(f"{value!r} is not a valid identifier")
    return value


Identifier = Annotated[str, AfterValidator(_is_varname)]

# ########################################################################################

class IonQubit(Expr):
    access: Access
    index: NonNegativeInt

class IonRegister(Expr):
    size: NonNegativeInt

class Access(Expr):
    name: Identifier

class MyList(Expr):
    values: List[Expr]
    
class Declaration(TypeReflectBaseModel):
    name: Identifier
    value: Expr

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
    duration: Expr
    target: Expr
    beam: Beam
    measured: Expr

class Beam(TypeReflectBaseModel):
    """
    Class representing a referenced optical channel/beam for the trapped-ion device.

    Attributes:
        rabi: Rabi frequency of the referenced transition driven by the beam.
        phase: Phase relative to the ion's clock.
        polarization: Polarization of the beam.
        wavevector: Wavevector of the beam.
    """
    frequency: Expr
    rabi: Expr
    phase: Expr
    polarization: Expr
    wavevector: Expr
    

class IfElse(TypeReflectBaseModel):
    """
    Class representing a conditional branch in the analog circuit
    """
    condition: Expr
    then_branch: List[Statement] = []
    else_branch: List[Statement] = []
    
    
class While(TypeReflectBaseModel):
    """
    Class representing a while loop in the analog circuit
    """
    condition : Expr
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


Statement = Union[Declaration, IfElse, While, Break, Continue, Pulse, ParallelProtocol]
