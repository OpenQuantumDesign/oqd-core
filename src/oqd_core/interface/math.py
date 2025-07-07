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

import ast
from typing import Annotated, Any, List, Literal, Union

import numpy as np
from oqd_compiler_infrastructure import (
    ConversionRule,
    Post,
    RewriteRule,
    TypeReflectBaseModel,
)
from pydantic import (
    AfterValidator,
    BeforeValidator,
    Discriminator,
    Tag,
    model_validator,
)

########################################################################################

__all__ = [
    "MathExpr",
    "MathTerminal",
    "MathStr",
    "MathNum",
    "MathVar",
    "MathImag",
    "MathFunc",
    "MathBinaryOp",
    "MathAdd",
    "MathSub",
    "MathMul",
    "MathDiv",
    "MathPow",
    "MathExprSubtypes",
    "ConstantMathExpr",
    "CastMathExpr",
]

########################################################################################


class MathExpr(TypeReflectBaseModel):
    """
    Class representing the abstract syntax tree (AST) for a mathematical expression
    """

    @classmethod
    def cast(cls, value: Any):
        if isinstance(value, dict):
            return value
        if isinstance(value, MathExpr):
            return value
        if isinstance(value, (int, float)):
            value = MathNum(value=value)
            return value
        if isinstance(value, (complex, np.complex128)):
            value = MathNum(value=value.real) + MathImag() * value.imag
            return value
        if isinstance(value, str):
            raise TypeError(
                "Tried to cast a string to MathExpr. "
                + f'Wrap your string ("{value}") with MathStr(string="{value}").'
            )
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
        try:
            return MathMul(expr1=self, expr2=other)
        except TypeError:  # make sure this is the right error to catch
            return other * self

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


########################################################################################


class MathTerminal(MathExpr):
    """
    Class representing a terminal in the [`MathExpr`][oqd_core.interface.math.MathExpr] abstract syntax tree (AST)
    """

    pass


class MathVar(MathTerminal):
    """
    Class representing a variable in a [`MathExpr`][oqd_core.interface.math.MathExpr]

    Examples:
        >>> MathVar("t")

    """

    name: VarName


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


def _is_varname(value: str) -> str:
    if not value.isidentifier():
        raise ValueError
    return value


VarName = Annotated[str, AfterValidator(_is_varname)]


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
    "atan2",
]
"""
List of supported functions
"""


class MathFunc(MathExpr):
    """
    Class representing a named function applied to a [`MathExpr`][oqd_core.interface.math.MathExpr] abstract syntax tree (AST)

    Attributes:
        func (SupportedFuncNames): Named function to apply
        expr (Union[CastMathExpr, List[CastMathExpr]]): Arguments of the named function
    """

    func: SupportedFuncNames
    expr: Annotated[
        Union[
            Annotated[CastMathExpr, Tag("MathExpr")],
            Annotated[List[CastMathExpr], Tag("list")],
        ],
        Discriminator(lambda v: "list" if isinstance(v, list) else "MathExpr"),
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

    expr1: CastMathExpr
    expr2: CastMathExpr


class MathSub(MathBinaryOp):
    """
    Class representing the subtraction of [`MathExprs`][oqd_core.interface.math.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
    """

    expr1: CastMathExpr
    expr2: CastMathExpr


class MathMul(MathBinaryOp):
    """
    Class representing the multiplication of [`MathExprs`][oqd_core.interface.math.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
    """

    expr1: CastMathExpr
    expr2: CastMathExpr


class MathDiv(MathBinaryOp):
    """
    Class representing the division of [`MathExprs`][oqd_core.interface.math.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
    """

    expr1: CastMathExpr
    expr2: CastMathExpr


class MathPow(MathBinaryOp):
    """
    Class representing the exponentiation of [`MathExprs`][oqd_core.interface.math.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
    """

    expr1: CastMathExpr
    expr2: CastMathExpr


########################################################################################

MathExprSubtypes = Annotated[
    Union[
        Annotated[MathNum, Tag("MathNum")],
        Annotated[MathVar, Tag("MathVar")],
        Annotated[MathImag, Tag("MathImag")],
        Annotated[MathFunc, Tag("MathFunc")],
        Annotated[MathAdd, Tag("MathAdd")],
        Annotated[MathSub, Tag("MathSub")],
        Annotated[MathMul, Tag("MathMul")],
        Annotated[MathDiv, Tag("MathDiv")],
        Annotated[MathPow, Tag("MathPow")],
    ],
    Discriminator(
        lambda v: v["class_"] if isinstance(v, dict) else getattr(v, "class_")
    ),
]
"""
Alias for the union of concrete MathExpr subtypes
"""

CastMathExpr = Annotated[MathExprSubtypes, BeforeValidator(MathExpr.cast)]
"""
Annotated type that cast typical numeric python types to MathExpr
"""

########################################################################################


class _MathExprIsConstant(RewriteRule):
    def map_MathExpr(self, model):
        if getattr(self, "isconstant", None) is None:
            self.isconstant = True

    def map_MathVar(self, model):
        self.isconstant = False


def _isconstant(model):
    constant_analysis = _MathExprIsConstant()

    Post(constant_analysis)(model)

    if constant_analysis.isconstant:
        return model

    raise ValueError("MathExpr is not a constant")


ConstantMathExpr = Annotated[
    CastMathExpr,
    AfterValidator(_isconstant),
]
"""
Annotated type for constant MathExpr
"""


########################################################################################


class _AST_to_MathExpr(ConversionRule):
    def generic_map(self, model: Any, operands):
        raise TypeError

    def map_Module(self, model: ast.Module, operands):
        if len(model.body) == 1:
            return self(model.body[0])
        raise TypeError

    def map_Expr(self, model: ast.Expr, operands):
        return self(model.value)

    def map_Constant(self, model: ast.Constant, operands):
        return MathExpr.cast(model.value)

    def map_Name(self, model: ast.Name, operands):
        return MathVar(name=model.id)

    def map_BinOp(self, model: ast.BinOp, operands):
        if isinstance(model.op, ast.Add):
            return MathAdd(expr1=self(model.left), expr2=self(model.right))
        if isinstance(model.op, ast.Sub):
            return MathSub(expr1=self(model.left), expr2=self(model.right))
        if isinstance(model.op, ast.Mult):
            return MathMul(expr1=self(model.left), expr2=self(model.right))
        if isinstance(model.op, ast.Div):
            return MathDiv(expr1=self(model.left), expr2=self(model.right))
        if isinstance(model.op, ast.Pow):
            return MathPow(expr1=self(model.left), expr2=self(model.right))
        raise TypeError

    def map_UnaryOp(self, model: ast.UnaryOp, operands):
        if isinstance(model.op, ast.USub):
            return -self(model.operand)
        if isinstance(model.op, ast.UAdd):
            return self(model.operand)
        raise TypeError

    def map_Call(self, model: ast.Call, operands):
        return MathFunc(
            func=model.func.id,
            expr=[self(arg) for arg in model.args],
        )


def MathStr(*, string):
    return _AST_to_MathExpr()(ast.parse(string))
