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

from typing import Annotated, List, Union

from oqd_compiler_infrastructure import TypeReflectBaseModel
from pydantic import Discriminator, Tag

from .expr import AtomicExprSubtypes, Identifier

########################################################################################

__all__ = [
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
    pulses: List[Statement]



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

########################################################################################


"""
Union of classes
"""


def _Statement_discriminator(value):
    if isinstance(value, dict):
        class_ = value["class_"]
    else:
        class_ = getattr(value, "class_")

    if class_ not in ["Declaration", "IfElse", "While", "Break", "Continue", "ParallelProtocol"]:
        class_ = "AtomicExpr"

    return class_


Statement = Annotated[
    Union[
        Annotated[Declaration, Tag("Declaration")],
        Annotated[IfElse, Tag("IfElse")],
        Annotated[While, Tag("While")],
        Annotated[Break, Tag("Break")],
        Annotated[Continue, Tag("Continue")],
        Annotated[ParallelProtocol, Tag("ParallelProtocol")],
        Annotated[AtomicExprSubtypes, Tag("AtomicExpr")],
    ],
    Discriminator(discriminator=_Statement_discriminator),
]
