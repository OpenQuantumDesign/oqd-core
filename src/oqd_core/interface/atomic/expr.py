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

from oqd_compiler_infrastructure import TypeReflectBaseModel, VisitableBaseModel
from pydantic import AfterValidator, BaseModel, BeforeValidator, Discriminator
from typing_extensions import TypeAliasType

########################################################################################

__all__ = [
    "AtomicExpr",
    "CastAtomicExpr",
    "Access",
    "AtomicList",
    "Extract",
    "Constant",
    "RuntimeVar",
    "Neg",
    "Pos",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Not",
    "And",
    "Or",
    "Xor",
    "Eq",
    "Neq",
    "Lt",
    "Leq",
    "Gt",
    "Geq",
    "BuiltinCall",
    "IonRegister",
    "Beam",
    "Pulse",
]

########################################################################################

class AtomicExpr(TypeReflectBaseModel):
    @classmethod
    def cast(cls, value: Any):
        match value:
            case str() if (
                value.startswith("#") and len(value) > 1 and value[1:].isidentifier()
            ):
                return RuntimeVar(name=value)
            case str() if value.isidentifier():
                return Access(name=value)
            case int() | float() | bool():
                return Constant(value=value)
            case complex():
                return Constant(value=Complex(real=value.real, imag=value.imag))
            case dict() | AtomicExpr():
                return value
            case Complex():
                return Constant(value=value)
            case _:
                raise ValueError("Failed to cast value to AtomicExpr")


class AbstractAtomicExpr: ...


########################################################################################


def _is_varname(value: str) -> str:
    if not value.isidentifier():
        raise ValueError(f"{value!r} is not a valid identifier")
    return value


Identifier = Annotated[str, AfterValidator(_is_varname)]


class Access(AtomicExpr):
    name: Identifier


########################################################################################

class CollectionExpr(AbstractAtomicExpr): ...


class IndexingExpr(AbstractAtomicExpr): ...


class AtomicList(CollectionExpr, AtomicExpr):
    values: List[CastAtomicExpr]


class Extract(IndexingExpr, AtomicExpr):
    value: CastAtomicExpr
    index: CastAtomicExpr


########################################################################################

class Complex(VisitableBaseModel):
    real: float
    imag: float


class Constant(AtomicExpr):
    value: Union[int, float, Complex, RuntimeVar, bool]


def _is_runtimevar(value: str) -> str:
    if not value.startswith("#") or len(value) < 2 or not value[1:].isidentifier():
        raise ValueError(
            "MathVar variable must start with a '#', followed by a valid identifier"
        )
    return value


RuntimeVarName = Annotated[str, AfterValidator(_is_runtimevar)]


class RuntimeVar(AtomicExpr):
    name: RuntimeVarName


########################################################################################


class UnaryOp(AbstractAtomicExpr):
    expr: CastAtomicExpr


ListCastAtomicExpr = TypeAliasType("ListCastAtomicExpr", "List[CastAtomicExpr]")


class BinaryOp(BaseModel, AbstractAtomicExpr):
    """
    Class representing binary operations on [`MathExprs`][oqd_core.interface.atomic.expr.MathExpr] abstract syntax tree (AST)
    """

    exprs: ListCastAtomicExpr


class ArithOp(AbstractAtomicExpr): ...


class BoolOp(AbstractAtomicExpr): ...


class Neg(UnaryOp, ArithOp, AtomicExpr): ...


class Pos(UnaryOp, ArithOp, AtomicExpr): ...


class Add(BinaryOp, ArithOp, AtomicExpr):
    """
    Class representing the addition of [`MathExprs`][oqd_core.interface.atomic.expr.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.atomic.expr.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.atomic.expr.MathExpr]
    """


class Sub(BinaryOp, ArithOp, AtomicExpr):
    """
    Class representing the subtraction of [`MathExprs`][oqd_core.interface.atomic.expr.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.atomic.expr.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.atomic.expr.MathExpr]
    """


class Mul(BinaryOp, ArithOp, AtomicExpr):
    """
    Class representing the multiplication of [`MathExprs`][oqd_core.interface.atomic.expr.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.atomic.expr.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.atomic.expr.MathExpr]
    """


class Div(BinaryOp, ArithOp, AtomicExpr):
    """
    Class representing the division of [`MathExprs`][oqd_core.interface.atomic.expr.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.atomic.expr.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.atomic.expr.MathExpr]
    """


class Pow(BinaryOp, ArithOp, AtomicExpr):
    """
    Class representing the exponentiation of [`MathExprs`][oqd_core.interface.atomic.expr.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.atomic.expr.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.atomic.expr.MathExpr]
    """


########################################################################################

class ComparisonOp(AbstractAtomicExpr):
    """
    Class representing binary operations on [`BoolExprs`][oqd_core.interface.atomic.expr.BoolExpr] abstract syntax tree (AST)
    """


class Not(UnaryOp, BoolOp, AtomicExpr): ...


class And(BinaryOp, BoolOp, AtomicExpr): ...


class Or(BinaryOp, BoolOp, AtomicExpr): ...


class Xor(BinaryOp, BoolOp, AtomicExpr): ...


class Eq(BinaryOp, BoolOp, ComparisonOp, AtomicExpr): ...


class Neq(BinaryOp, BoolOp, ComparisonOp, AtomicExpr): ...


class Lt(BinaryOp, BoolOp, ComparisonOp, AtomicExpr): ...


class Leq(BinaryOp, BoolOp, ComparisonOp, AtomicExpr): ...


class Gt(BinaryOp, BoolOp, ComparisonOp, AtomicExpr): ...


class Geq(BinaryOp, BoolOp, ComparisonOp, AtomicExpr): ...


########################################################################################

SupportedBuiltinFuncs = Literal[
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
    "round",
    "range",
    "flatten",
    "print",
    "len",
    "_pos",
    "_neg",
    "_add",
    "_sub",
    "_mul",
    "_div",
    "_pow",
    "_eq",
    "_neq",
    "_lt",
    "_leq",
    "_gt",
    "_geq",
    "_and",
    "_or",
    "_xor",
    "_not",
]
"""
List of supported functions
"""


class BuiltinCall(AtomicExpr):
    """
    Class representing a named function applied to a [`MathExpr`][oqd_core.interface.atomic.expr.MathExpr] abstract syntax tree (AST)

    Attributes:
        func (SupportedFuncNames): Named function to apply
        expr (Union[CastMathExpr, List[CastMathExpr]]): Arguments of the named function
    """

    func: SupportedBuiltinFuncs
    args: List[CastAtomicExpr]

########################################################################################

class RegisterExpr(AbstractAtomicExpr): ...

class IonRegister(CollectionExpr, RegisterExpr, AtomicExpr):
    size: CastAtomicExpr

########################################################################################

class Beam(AtomicExpr):
    """
    Class representing a referenced optical channel/beam for the trapped-ion device.

    Attributes:
        frequency: frequency of the beam.
        rabi: Rabi frequency of the referenced transition driven by the beam.
        phase: Phase relative to the ion's clock.
        polarization: Polarization of the beam.
        wavevector: Wavevector of the beam.
    """

    frequency: CastAtomicExpr
    rabi: CastAtomicExpr
    phase: CastAtomicExpr
    polarization: CastAtomicExpr
    wavevector: CastAtomicExpr


class Pulse(AtomicExpr):
    """
    Class representing the application of the beam for some duration.

    Attributes:
        beam: Optical channel/beam to turn on.
        duration: Period of time to turn the optical channel on for.
        target: Target ion of the beam.
        measured: Boolean that tracks if the pulse has been measured.
    """

    beam: CastAtomicExpr
    duration: CastAtomicExpr
    target: CastAtomicExpr
    measured: CastAtomicExpr


########################################################################################


AtomicExprSubtypes = Annotated[
    Union[tuple(AtomicExpr.__subclasses__())],
    Discriminator(discriminator="class_"),
]


CastAtomicExpr = Annotated[AtomicExprSubtypes, BeforeValidator(AtomicExpr.cast)]
