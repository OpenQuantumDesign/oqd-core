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

from oqd_compiler_infrastructure import (
    TypeReflectBaseModel,
)
from pydantic import (
    Discriminator,
    NonNegativeInt,
    Tag,
    model_validator,
)

from .atom import Access, Atom

########################################################################################

__all__ = [
    "AnalogExpr",
    "AnalogExprSubtypes",
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
    def cast(self): ...


class MathExpr(AnalogExpr): ...


class BoolExpr(AnalogExpr): ...


class OperatorExpr(AnalogExpr): ...


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
            Annotated[AnalogExprSubtypes, Tag("expr")],
            Annotated[List[AnalogExprSubtypes], Tag("list")],
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

    expr1: AnalogExprSubtypes
    expr2: AnalogExprSubtypes


class MathSub(MathBinaryOp):
    """
    Class representing the subtraction of [`MathExprs`][oqd_core.interface.math.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
    """

    expr1: AnalogExprSubtypes
    expr2: AnalogExprSubtypes


class MathMul(MathBinaryOp):
    """
    Class representing the multiplication of [`MathExprs`][oqd_core.interface.math.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
    """

    expr1: AnalogExprSubtypes
    expr2: AnalogExprSubtypes


class MathDiv(MathBinaryOp):
    """
    Class representing the division of [`MathExprs`][oqd_core.interface.math.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
    """

    expr1: AnalogExprSubtypes
    expr2: AnalogExprSubtypes


class MathPow(MathBinaryOp):
    """
    Class representing the exponentiation of [`MathExprs`][oqd_core.interface.math.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
    """

    expr1: AnalogExprSubtypes
    expr2: AnalogExprSubtypes


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
    expr: AnalogExprSubtypes


class BoolAnd(BoolBinaryOp):
    expr1: AnalogExprSubtypes
    expr2: AnalogExprSubtypes


class BoolOr(BoolBinaryOp):
    expr1: AnalogExprSubtypes
    expr2: AnalogExprSubtypes


class BoolEq(ComparisonOp):
    expr1: AnalogExprSubtypes
    expr2: AnalogExprSubtypes


class BoolNotEq(ComparisonOp):
    expr1: AnalogExprSubtypes
    expr2: AnalogExprSubtypes


class BoolLessThan(ComparisonOp):
    expr1: AnalogExprSubtypes
    expr2: AnalogExprSubtypes


class BoolLessThanEq(ComparisonOp):
    expr1: AnalogExprSubtypes
    expr2: AnalogExprSubtypes


class BoolGreaterThan(ComparisonOp):
    expr1: AnalogExprSubtypes
    expr2: AnalogExprSubtypes


class BoolGreaterThanEq(ComparisonOp):
    expr1: AnalogExprSubtypes
    expr2: AnalogExprSubtypes


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

    op1: AnalogExprSubtypes
    op2: AnalogExprSubtypes


class OperatorSub(OperatorBinaryOp):
    """
    Class representing the subtraction of [`Operators`][oqd_core.interface.analog.operator.Operator]

    Attributes:
        op1 (Operator): Left hand side [`Operator`][oqd_core.interface.analog.operator.Operator]
        op2 (Operator): Right hand side [`Operator`][oqd_core.interface.analog.operator.Operator]
    """

    op1: AnalogExprSubtypes
    op2: AnalogExprSubtypes


class OperatorMul(OperatorBinaryOp):
    """
    Class representing the multiplication of [`Operators`][oqd_core.interface.analog.operator.Operator]

    Attributes:
        op1 (Operator): Left hand side [`Operator`][oqd_core.interface.analog.operator.Operator]
        op2 (Operator): Right hand side [`Operator`][oqd_core.interface.analog.operator.Operator]
    """

    op1: AnalogExprSubtypes
    op2: AnalogExprSubtypes


class OperatorKron(OperatorBinaryOp):
    """
    Class representing the tensor product of [`Operators`][oqd_core.interface.analog.operator.Operator]

    Attributes:
        op1 (Operator): Left hand side [`Operator`][oqd_core.interface.analog.operator.Operator]
        op2 (Operator): Right hand side [`Operator`][oqd_core.interface.analog.operator.Operator]
    """

    op1: AnalogExprSubtypes
    op2: AnalogExprSubtypes


########################################################################################


class AnalogList(AnalogExpr):
    values: List[AnalogExprSubtypes]


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
