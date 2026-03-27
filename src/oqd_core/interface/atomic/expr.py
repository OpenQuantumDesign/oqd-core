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
from oqd_compiler_infrastructure import TypeReflectBaseModel
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
    "AtomicExpr",
    "CastAtomicExpr",
    "Atom",
    "Access",
    "MathNum",
    "MathVar",
    "MathImag",
    "Bool",
    "IonQubit",
    "IonRegister",
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
    "AtomicList",
    "AtomicListExtract",
    "Beam",
]

########################################################################################

class AtomicExpr(TypeReflectBaseModel):
    @classmethod
    def cast(cls, value: Any):
        if isinstance(value, dict):
            return value
        if isinstance(value, AtomicExpr):
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


class MathExpr(AtomicExpr): ...


class BoolExpr(AtomicExpr): ...

########################################################################################


def _is_varname(value: str) -> str:
    if not value.isidentifier():
        raise ValueError(f"{value!r} is not a valid identifier")
    return value


Identifier = Annotated[str, AfterValidator(_is_varname)]


class Access(AtomicExpr):
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

class IonRegister(AtomicExpr):
    size: NonNegativeInt

class Beam(AtomicExpr):
    """
    Class representing a referenced optical channel/beam for the trapped-ion device.

    Attributes:
        rabi: Rabi frequency of the referenced transition driven by the beam.
        phase: Phase relative to the ion's clock.
        polarization: Polarization of the beam.
        wavevector: Wavevector of the beam.
    """
    frequency: CastAtomicExpr
    rabi: CastAtomicExpr
    phase: CastAtomicExpr
    polarization: CastAtomicExpr
    wavevector:CastAtomicExpr


########################################################################################

Atom = Union[Bool, MathVar, MathNum, MathImag, Access, IonRegister, Beam]

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


class MathFunc(AtomicExpr):
    """
    Class representing a named function applied to a [`MathExpr`][oqd_core.interface.math.MathExpr] abstract syntax tree (AST)

    Attributes:
        func (SupportedFuncNames): Named function to apply
        expr (Union[CastMathExpr, List[CastMathExpr]]): Arguments of the named function
    """

    func: SupportedFuncNames
    expr: Annotated[
        Union[
            Annotated[CastAtomicExpr, Tag("expr")],
            Annotated[List[CastAtomicExpr], Tag("list")],
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
    Class representing the addition of [`MathExprs`][oqd_core.interface.Atomic.operator.Operator]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.Atomic.operator.Operator]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.Atomic.operator.Operator]
    """

    expr1: CastAtomicExpr
    expr2: CastAtomicExpr


class MathSub(MathBinaryOp):
    """
    Class representing the subtraction of [`MathExprs`][oqd_core.interface.math.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
    """

    expr1: CastAtomicExpr
    expr2: CastAtomicExpr


class MathMul(MathBinaryOp):
    """
    Class representing the multiplication of [`MathExprs`][oqd_core.interface.math.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
    """

    expr1: CastAtomicExpr
    expr2: CastAtomicExpr


class MathDiv(MathBinaryOp):
    """
    Class representing the division of [`MathExprs`][oqd_core.interface.math.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
    """

    expr1: CastAtomicExpr
    expr2: CastAtomicExpr


class MathPow(MathBinaryOp):
    """
    Class representing the exponentiation of [`MathExprs`][oqd_core.interface.math.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.math.MathExpr]
    """

    expr1: CastAtomicExpr
    expr2: CastAtomicExpr


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
    expr: CastAtomicExpr


class BoolAnd(BoolBinaryOp):
    expr1: CastAtomicExpr
    expr2: CastAtomicExpr


class BoolOr(BoolBinaryOp):
    expr1: CastAtomicExpr
    expr2: CastAtomicExpr


class BoolEq(ComparisonOp):
    expr1: CastAtomicExpr
    expr2: CastAtomicExpr


class BoolNotEq(ComparisonOp):
    expr1: CastAtomicExpr
    expr2: CastAtomicExpr


class BoolLessThan(ComparisonOp):
    expr1: CastAtomicExpr
    expr2: CastAtomicExpr


class BoolLessThanEq(ComparisonOp):
    expr1: CastAtomicExpr
    expr2: CastAtomicExpr


class BoolGreaterThan(ComparisonOp):
    expr1: CastAtomicExpr
    expr2: CastAtomicExpr


class BoolGreaterThanEq(ComparisonOp):
    expr1: CastAtomicExpr
    expr2: CastAtomicExpr


########################################################################################


class AtomicList(AtomicExpr):
    values: List[CastAtomicExpr]


class AtomicListExtract(AtomicExpr):
    access: Access
    index: NonNegativeInt
    
    
class IonQubit(AtomicExpr):
    access: Access
    index: NonNegativeInt


########################################################################################

AtomicExprSubtypes = Annotated[
    Union[
        Annotated[Beam, Tag("Beam")],
        Annotated[Bool, Tag("Bool")],
        Annotated[MathVar, Tag("MathVar")],
        Annotated[MathNum, Tag("MathNum")],
        Annotated[MathImag, Tag("MathImag")],
        Annotated[Access, Tag("Access")],
        Annotated[IonRegister, Tag("IonRegister")],
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
        Annotated[IonQubit, Tag("IonQubit")],
        Annotated[AtomicList, Tag("AtomicList")],
        Annotated[AtomicListExtract, Tag("AtomicListExtract")],
    ],
    Discriminator(lambda v: v["class_"] if isinstance(v, dict) else getattr(v, "class_")),
]

CastAtomicExpr = Annotated[AtomicExprSubtypes, BeforeValidator(AtomicExpr.cast)]


