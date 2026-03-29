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
from pydantic import (
    Discriminator,
    Tag
)
from .expr import AtomicExprSubtypes, Identifier

########################################################################################

__all__ = [
    "Pulse",
    "Declaration",
    "ParallelProtocol",
    "IfElse",
    "While",
    "Break", 
    "Continue",
]


########################################################################################

class Declaration(TypeReflectBaseModel):
    name: Identifier
    value: AtomicExprSubtypes


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
    duration: AtomicExprSubtypes
    target: AtomicExprSubtypes
    beam: AtomicExprSubtypes
    measured: AtomicExprSubtypes
    

class IfElse(TypeReflectBaseModel):
    """
    Class representing a conditional branch in the analog circuit
    """
    condition: AtomicExprSubtypes
    then_branch: List[Statement] = []
    else_branch: List[Statement] = []
    
    
class While(TypeReflectBaseModel):
    """
    Class representing a while loop in the analog circuit
    """
    condition: AtomicExprSubtypes
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


Statement = Annotated[
    Union[
        Annotated[Declaration, Tag("Declaration")],
        Annotated[Pulse, Tag("Pulse")],
        Annotated[ParallelProtocol, Tag("ParallelProtocol")],
        Annotated[IfElse, Tag("IfElse")],
        Annotated[While, Tag("While")],
        Annotated[Break, Tag("Break")],
        Annotated[Continue, Tag("Continue")],
    ],
    Discriminator(lambda v: v["class_"] if isinstance(v, dict) else getattr(v, "class_")),
]
