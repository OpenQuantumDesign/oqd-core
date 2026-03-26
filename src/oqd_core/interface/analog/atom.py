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

from typing import Annotated, Union

from oqd_compiler_infrastructure import (
    TypeReflectBaseModel,
)
from pydantic import (
    AfterValidator,
    Discriminator,
    Tag,
)

########################################################################################

__all__ = [
    "Atom",
    "Access",
    "MathNum",
    "MathVar",
    "MathImag",
    "Bool",
    "PauliI",
    "PauliX",
    "PauliY",
    "PauliZ",
    "Ladder",
    "Creation",
    "Annihilation",
    "Identity",
]

########################################################################################


def _is_varname(value: str) -> str:
    if not value.isidentifier():
        raise ValueError(f"{value!r} is not a valid identifier")
    return value


Identifier = Annotated[str, AfterValidator(_is_varname)]


class Access(TypeReflectBaseModel):
    name: Identifier


########################################################################################


class MathTerminal(TypeReflectBaseModel): ...


class MathVar(MathTerminal):
    """
    Class representing a variable in a [`MathExpr`][oqd_core.interface.math.MathExpr]

    Examples:
        >>> MathVar("t")

    """

    name: MathVarName


class MathNum(MathTerminal):
    """
    Class representing a number in a [`MathExpr`][oqd_core.interface.math.MathExpr]
    """

    value: Union[int, float]


class MathImag(MathTerminal):
    """
    Class representing the imaginary unit in a [`MathExpr`][oqd_core.interface.math.MathExpr] abstract syntax tree (AST)
    """

    pass


def _is_mathvarname(value: str) -> str:
    if not value.startswith("#") or len(value) < 2 or not value[1:].isidentifier():
        raise ValueError(
            "MathVar variable must start with a '#', followed by a valid identifier"
        )
    return value


MathVarName = Annotated[str, AfterValidator(_is_mathvarname)]

########################################################################################


class Bool(TypeReflectBaseModel):
    value: bool


########################################################################################


class OperatorTerminal(TypeReflectBaseModel): ...


########################################################################################


class Pauli(OperatorTerminal):
    """
    Class representing a Pauli operator
    """

    pass


class PauliI(Pauli):
    """
    Class for the Pauli I operator
    """

    pass


class PauliX(Pauli):
    """
    Class for the Pauli X operator
    """

    pass


class PauliY(Pauli):
    """
    Class for the Pauli Y operator
    """

    pass


class PauliZ(Pauli):
    """
    Class for the Pauli Z operator
    """

    pass


########################################################################################


class Ladder(OperatorTerminal):
    """
    Class representing a ladder operator in Fock space
    """

    pass


class Creation(Ladder):
    """
    Class for the Creation operator in Fock space
    """

    pass


class Annihilation(Ladder):
    """
    Class for the Annihilation operator in Fock space
    """

    pass


class Identity(Ladder):
    """
    Class for the Identity operator in Fock space
    """

    pass


########################################################################################

Atom = Annotated[
    Union[
        Annotated[Bool, Tag("Bool")],
        Annotated[MathVar, Tag("MathVar")],
        Annotated[MathNum, Tag("MathNum")],
        Annotated[MathImag, Tag("MathImag")],
        Annotated[PauliX, Tag("PauliX")],
        Annotated[PauliY, Tag("PauliY")],
        Annotated[PauliZ, Tag("PauliZ")],
        Annotated[PauliI, Tag("PauliI")],
        Annotated[Annihilation, Tag("Annihilation")],
        Annotated[Creation, Tag("Creation")],
        Annotated[Identity, Tag("Identity")],
        Annotated[Access, Tag("Access")],
    ],
    Discriminator("class_"),
]
