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

from typing import Annotated, Any, List, Literal, Union

import numpy as np
from oqd_compiler_infrastructure import (
    TypeReflectBaseModel,
)
from pydantic import (
    AfterValidator,
    BeforeValidator,
    Discriminator,
    NonNegativeInt,
    Tag,
    model_validator,
)

########################################################################################

__all__ = [
    "AnalogExpr",
    "CastAnalogExpr",
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
    "QuantumRegister",
    "ModeRegister",
    "MathExpr",
    "MathFunc",
    "MathBinaryOp",
    "MathAdd",
    "MathSub",
    "MathMul",
    "MathDiv",
    "MathPow",
    "BoolAnd",
    "BoolOr",
    "BoolNot",
    "BoolEq",
    "BoolNotEq",
    "BoolLessThan",
    "BoolLessThanEq",
    "BoolGreaterThan",
    "BoolGreaterThanEq",
    "BoolExpr",
    "OperatorAdd",
    "OperatorSub",
    "OperatorMul",
    "OperatorKron",
    "AnalogList",
    "AnalogListExtract",
    "QuantumBit",
    "QuantumMode",
]

########################################################################################


class AnalogExpr(TypeReflectBaseModel):
    @classmethod
    def cast(cls, value: Any):
        if isinstance(value, dict):
            return value
        if isinstance(value, AnalogExpr):
            return value
        if isinstance(value, (int, float)):
            value = MathNum(value=value)
            return value
        if isinstance(value, (complex, np.complex128)):
            value = MathNum(value=value.real) + MathImag() * value.imag
            return value
        if isinstance(value, str) and value.startswith("#"):
            return MathVar(name=value)
        if isinstance(value, str):
            return Access(name=value)

        raise TypeError

    def __neg__(self):
        return MathMul(expr1=MathNum(value=-1), expr2=self)

    def __pos__(self):
        return self

    def __add__(self, other):
        return MathAdd(expr1=self, expr2=other)

    def __sub__(self, other):
        return MathSub(expr1=self, expr2=other)

    def __mul__(self, other):
        return MathMul(expr1=self, expr2=other)

    def __truediv__(self, other):
        return MathDiv(expr1=self, expr2=other)

    def __pow__(self, other):
        return MathPow(expr1=self, expr2=other)

    def __radd__(self, other):
        other = MathExpr.cast(other)
        return other + self

    def __rsub__(self, other):
        other = MathExpr.cast(other)
        return other - self

    def __rmul__(self, other):
        other = MathExpr.cast(other)
        return other * self

    def __rpow__(self, other):
        other = MathExpr.cast(other)
        return other**self

    def __rtruediv__(self, other):
        other = MathExpr.cast(other)
        return other / self


class MathExpr(AnalogExpr): ...


class BoolExpr(AnalogExpr): ...


class OperatorExpr(AnalogExpr): ...


########################################################################################


def _is_varname(value: str) -> str:
    if not value.isidentifier():
        raise ValueError(f"{value!r} is not a valid identifier")
    return value


Identifier = Annotated[str, AfterValidator(_is_varname)]


class Access(AnalogExpr):
    name: Identifier


########################################################################################


class MathTerminal(MathExpr): ...


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


class Bool(BoolExpr):
    value: bool


########################################################################################


class OperatorTerminal(OperatorExpr): ...


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


class Register(AnalogExpr):
    pass


class QuantumRegister(Register):
    size: NonNegativeInt


class ModeRegister(Register):
    size: NonNegativeInt


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
        Annotated[QuantumRegister, Tag("QuantumRegister")],
        Annotated[ModeRegister, Tag("ModeRegister")],
    ],
    Discriminator("class_"),
]


########################################################################################


SupportedFuncNames = Literal[
    "abs",
    "sin",
    "cos",
    "tan",
    "exp",
    "log",
    "sinh",
    "cosh",
    "tanh",
    "atan",
    "acos",
    "asin",
    "atanh",
    "asinh",
    "acosh",
    "heaviside",
    "conj",
    "real",
    "imag",
    "atan2",
]
"""
List of supported functions
"""


class MathFunc(AnalogExpr):
    """
    Class representing a named function applied to a [`MathExpr`][oqd_core.interface.math.MathExpr] abstract syntax tree (AST)

    Attributes:
        func (SupportedFuncNames): Named function to apply
        expr (Union[CastMathExpr, List[CastMathExpr]]): Arguments of the named function
    """

    func: SupportedFuncNames
    expr: Annotated[
        Union[
            Annotated[CastAnalogExpr, Tag("expr")],
            Annotated[List[CastAnalogExpr], Tag("list")],
        ],
        Discriminator(lambda v: "list" if isinstance(v, list) else "expr"),
    ]

    @model_validator(mode="before")
    @classmethod
    def args_validate(cls, data):
        if data["func"] in [
            "abs",
            "sin",
            "cos",
            "tan",
            "exp",
            "log",
            "sinh",
            "cosh",
            "tanh",
            "atan",
            "acos",
            "asin",
            "atanh",
            "asinh",
            "acosh",
            "heaviside",
            "conj",
            "real",
            "imag",
        ]:
            if isinstance(data["expr"], list):
                assert (
                    len(data["expr"]) == 1
                ), "Attempted to apply unary function on multiple arguments"
                data["expr"] = data["expr"][0]

        if data["func"] in [
            "atan2",
        ]:
            assert (
                isinstance(data["expr"], list) and len(data["expr"]) == 2
            ), "Attempted to apply binary function with incorrect number of arguments"

        return data


########################################################################################


class MathBinaryOp(MathExpr):
    """
    Class representing binary operations on [`MathExprs`][oqd_core.interface.math.MathExpr] abstract syntax tree (AST)
    """

    pass


class MathAdd(MathBinaryOp):
    """
    Class representing the addition of [`MathExprs`][oqd_core.interface.analog.operator.Operator]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.analog.operator.Operator]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.analog.operator.Operator]
    """

    expr1: CastAnalogExpr
    expr2: CastAnalogExpr


class MathSub(MathBinaryOp):
    """
    Class representing the subtraction of [`MathExprs`][oqd_core.interface.math.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
    """

    expr1: CastAnalogExpr
    expr2: CastAnalogExpr


class MathMul(MathBinaryOp):
    """
    Class representing the multiplication of [`MathExprs`][oqd_core.interface.math.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
    """

    expr1: CastAnalogExpr
    expr2: CastAnalogExpr


class MathDiv(MathBinaryOp):
    """
    Class representing the division of [`MathExprs`][oqd_core.interface.math.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
    """

    expr1: CastAnalogExpr
    expr2: CastAnalogExpr


class MathPow(MathBinaryOp):
    """
    Class representing the exponentiation of [`MathExprs`][oqd_core.interface.math.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
    """

    expr1: CastAnalogExpr
    expr2: CastAnalogExpr


########################################################################################


class BoolUnaryOp(BoolExpr):
    """
    Class representing binary operations on [`BoolExprs`][oqd_core.interface.bool.BoolExpr] abstract syntax tree (AST)
    """

    pass


class BoolBinaryOp(BoolExpr):
    """
    Class representing binary operations on [`BoolExprs`][oqd_core.interface.bool.BoolExpr] abstract syntax tree (AST)
    """

    pass


class ComparisonOp(BoolExpr):
    """
    Class representing binary operations on [`BoolExprs`][oqd_core.interface.bool.BoolExpr] abstract syntax tree (AST)
    """

    pass


class BoolNot(BoolUnaryOp):
    expr: CastAnalogExpr


class BoolAnd(BoolBinaryOp):
    expr1: CastAnalogExpr
    expr2: CastAnalogExpr


class BoolOr(BoolBinaryOp):
    expr1: CastAnalogExpr
    expr2: CastAnalogExpr


class BoolEq(ComparisonOp):
    expr1: CastAnalogExpr
    expr2: CastAnalogExpr


class BoolNotEq(ComparisonOp):
    expr1: CastAnalogExpr
    expr2: CastAnalogExpr


class BoolLessThan(ComparisonOp):
    expr1: CastAnalogExpr
    expr2: CastAnalogExpr


class BoolLessThanEq(ComparisonOp):
    expr1: CastAnalogExpr
    expr2: CastAnalogExpr


class BoolGreaterThan(ComparisonOp):
    expr1: CastAnalogExpr
    expr2: CastAnalogExpr


class BoolGreaterThanEq(ComparisonOp):
    expr1: CastAnalogExpr
    expr2: CastAnalogExpr


########################################################################################


class OperatorBinaryOp(OperatorExpr):
    """
    Class representing binary operations on [`Operators`][oqd_core.interface.analog.operator.Operator]
    """

    pass


class OperatorAdd(OperatorBinaryOp):
    """
    Class representing the addition of [`Operators`][oqd_core.interface.analog.operator.Operator]

    Attributes:
        op1 (Operator): Left hand side [`Operator`][oqd_core.interface.analog.operator.Operator]
        op2 (Operator): Right hand side [`Operator`][oqd_core.interface.analog.operator.Operator]
    """

    op1: CastAnalogExpr
    op2: CastAnalogExpr


class OperatorSub(OperatorBinaryOp):
    """
    Class representing the subtraction of [`Operators`][oqd_core.interface.analog.operator.Operator]

    Attributes:
        op1 (Operator): Left hand side [`Operator`][oqd_core.interface.analog.operator.Operator]
        op2 (Operator): Right hand side [`Operator`][oqd_core.interface.analog.operator.Operator]
    """

    op1: CastAnalogExpr
    op2: CastAnalogExpr


class OperatorMul(OperatorBinaryOp):
    """
    Class representing the multiplication of [`Operators`][oqd_core.interface.analog.operator.Operator]

    Attributes:
        op1 (Operator): Left hand side [`Operator`][oqd_core.interface.analog.operator.Operator]
        op2 (Operator): Right hand side [`Operator`][oqd_core.interface.analog.operator.Operator]
    """

    op1: CastAnalogExpr
    op2: CastAnalogExpr


class OperatorKron(OperatorBinaryOp):
    """
    Class representing the tensor product of [`Operators`][oqd_core.interface.analog.operator.Operator]

    Attributes:
        op1 (Operator): Left hand side [`Operator`][oqd_core.interface.analog.operator.Operator]
        op2 (Operator): Right hand side [`Operator`][oqd_core.interface.analog.operator.Operator]
    """

    op1: CastAnalogExpr
    op2: CastAnalogExpr


########################################################################################


class AnalogList(AnalogExpr):
    values: List[CastAnalogExpr]


class AnalogListExtract(TypeReflectBaseModel):
    access: Access
    index: NonNegativeInt


class QuantumBit(TypeReflectBaseModel):
    access: Access
    index: NonNegativeInt


class QuantumMode(TypeReflectBaseModel):
    access: Access
    index: NonNegativeInt


########################################################################################

AnalogExprSubtypes = Annotated[
    Union[
        Atom,
        Annotated[BoolAnd, Tag("BoolAnd")],
        Annotated[BoolOr, Tag("BoolOr")],
        Annotated[BoolNot, Tag("BoolNot")],
        Annotated[BoolEq, Tag("BoolEq")],
        Annotated[BoolNotEq, Tag("BoolNotEq")],
        Annotated[BoolLessThan, Tag("BoolLessThan")],
        Annotated[BoolLessThanEq, Tag("BoolLessThanEq")],
        Annotated[BoolGreaterThan, Tag("BoolGreaterThan")],
        Annotated[BoolGreaterThanEq, Tag("BoolGreaterThanEq")],
        Annotated[MathFunc, Tag("MathFunc")],
        Annotated[MathAdd, Tag("MathAdd")],
        Annotated[MathSub, Tag("MathSub")],
        Annotated[MathMul, Tag("MathMul")],
        Annotated[MathDiv, Tag("MathDiv")],
        Annotated[MathPow, Tag("MathPow")],
        Annotated[OperatorAdd, Tag("OperatorAdd")],
        Annotated[OperatorSub, Tag("OperatorSub")],
        Annotated[OperatorMul, Tag("OperatorMul")],
        Annotated[OperatorKron, Tag("OperatorKron")],
        Annotated[QuantumBit, Tag("QuantumBit")],
        Annotated[QuantumMode, Tag("QuantumMode")],
        Annotated[AnalogList, Tag("AnalogList")],
        Annotated[AnalogListExtract, Tag("AnalogListExtract")],
    ],
    Discriminator("class_"),
]

CastAnalogExpr = Annotated[AnalogExprSubtypes, BeforeValidator(AnalogExpr.cast)]
