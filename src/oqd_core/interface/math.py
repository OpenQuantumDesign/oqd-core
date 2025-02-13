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
from typing import Annotated, Any, Literal, Union

import numpy as np
from oqd_compiler_infrastructure import ConversionRule, TypeReflectBaseModel
from pydantic import AfterValidator, BeforeValidator

########################################################################################

__all__ = [
    "MathExpr",
    "MathTerminal",
    "MathStr",
    "MathNum",
    "MathVar",
    "MathImag",
    "MathUnaryOp",
    "MathFunc",
    "MathBinaryOp",
    "MathAdd",
    "MathSub",
    "MathMul",
    "MathDiv",
    "MathPow",
    "MathExprSubtypes",
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

    pass


########################################################################################


def is_varname(value: str) -> str:
    if not value.isidentifier():
        raise ValueError
    return value


VarName = Annotated[str, AfterValidator(is_varname)]
Functions = Literal[
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
]


########################################################################################


class AST_to_MathExpr(ConversionRule):
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
        if len(model.args) == 1:
            return MathFunc(func=model.func.id, expr=self(model.args[0]))
        raise TypeError


def MathStr(*, string):
    return AST_to_MathExpr()(ast.parse(string))


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


class MathUnaryOp(MathExpr):
    """
    Class representing a unary operations on a [`MathExpr`][oqd_core.interface.math.MathExpr] abstract syntax tree (AST)
    """

    pass


class MathFunc(MathUnaryOp):
    """
    Class representing a named function applied to a [`MathExpr`][oqd_core.interface.math.MathExpr] abstract syntax tree (AST)

    Attributes:
        func (Literal["sin", "cos", "tan", "exp", "log", "sinh", "cosh", "tanh"]): Named function to apply
        expr (MathExpr): Argument of the named function
    """

    func: Functions
    expr: CastMathExpr


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

MathExprSubtypes = Union[
    MathNum,
    MathVar,
    MathImag,
    MathFunc,
    MathAdd,
    MathSub,
    MathMul,
    MathDiv,
    MathPow,
]
"""
Alias for the union of concrete MathExpr subtypes
"""

CastMathExpr = Annotated[MathExprSubtypes, BeforeValidator(MathExpr.cast)]
