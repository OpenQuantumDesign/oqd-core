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

from oqd_core.interface.analog.expr import AnalogExprSubtypes, Identifier

########################################################################################
__all__ = [
    "Evolve",
    "Measure",
    "Initialize",
    "Declaration",
    "IfElse",
    "While",
    "Break",
    "Continue",
]

########################################################################################


class Declaration(TypeReflectBaseModel):
    name: Identifier
    value: AnalogExprSubtypes


class Evolve(TypeReflectBaseModel):
    """
    Class representing an evolution by an analog gate in the analog circuit

    Attributes:
        hamiltonian (Expr): Function to evolve by
        duration (Expr): Duration of the evolution
        targets (Expr): Indices and Quantum objects on which to apply the Hamiltonian
    """

    hamiltonian: AnalogExprSubtypes
    duration: AnalogExprSubtypes
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

    condition: AnalogExprSubtypes
    then_branch: List[Statement] = []
    else_branch: List[Statement] = []


class While(TypeReflectBaseModel):
    """
    Class representing a while loop in the analog circuit
    """

    condition: AnalogExprSubtypes
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

Statement = Annotated[
    Union[
        Annotated[Declaration, Tag("Declaration")],
        Annotated[Measure, Tag("Measure")],
        Annotated[Evolve, Tag("Evolve")],
        Annotated[Initialize, Tag("Initialize")],
        Annotated[IfElse, Tag("IfElse")],
        Annotated[While, Tag("While")],
        Annotated[Break, Tag("Break")],
        Annotated[Continue, Tag("Continue")],
    ],
    Discriminator("class_"),
]
