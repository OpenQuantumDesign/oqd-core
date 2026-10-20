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
from pydantic import Discriminator

from oqd_core.interface.analog.expr import AnalogExpr, CastAnalogExpr, Identifier

########################################################################################

__all__ = [
    "Declaration",
    "SCF",
    "IfElse",
    "While",
    "Break",
    "Continue",
]

########################################################################################


class Statement(TypeReflectBaseModel): ...


class AbstractStatement: ...


class SCF(AbstractStatement): ...


########################################################################################


class Declaration(Statement):
    name: Identifier
    value: CastAnalogExpr


class IfElse(Statement, SCF):
    """
    Class representing a conditional branch in the analog circuit
    """

    condition: CastAnalogExpr
    then_branch: List[StatementSubtypes] = []
    else_branch: List[StatementSubtypes] = []


class While(Statement, SCF):
    """
    Class representing a while loop in the analog circuit
    """

    condition: CastAnalogExpr
    body: List[StatementSubtypes] = []


class Break(Statement, SCF):
    """
    Class representing a break statement to exit the innermost loop
    """

    pass


class Continue(Statement, SCF):
    """
    Class representing a continue statement to jump to the next loop iteration
    """

    pass


########################################################################################


"""
Union of classes
"""


StatementSubtypes = Annotated[
    Union[tuple(AnalogExpr.__subclasses__() + Statement.__subclasses__())],
    Discriminator(discriminator="class_"),
]
