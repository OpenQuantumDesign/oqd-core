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
from pydantic import (
    AfterValidator,
    BeforeValidator,
    Discriminator,
)

########################################################################################

__all__ = [
    "AnalogExpr",
    "CastAnalogExpr",
    "Access",
    "AnalogList",
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
    "Kron",
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
    "PauliI",
    "PauliX",
    "PauliY",
    "PauliZ",
    "Creation",
    "Annihilation",
    "Identity",
    "QuantumRegister",
    "ModeRegister",
    "Evolve",
    "Measure",
    "Initialize",
]

########################################################################################


class AnalogExpr(TypeReflectBaseModel):
    @classmethod
    def cast(cls, value: Any):
        match value:
            case str() if value in DEFAULT_OP_MAPPING.keys():
                return DEFAULT_OP_MAPPING[value]()
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
            case dict() | AnalogExpr():
                return value
            case Complex():
                return Constant(value=value)
            case _:
                raise ValueError("Failed to cast value to AnalogExpr")

    def __neg__(self):
        return Neg(expr=self)

    def __pos__(self):
        return Pos(expr=self)

    def __add__(self, other):
        return Add(expr1=self, expr2=other)

    def __sub__(self, other):
        return Sub(expr1=self, expr2=other)

    def __mul__(self, other):
        return Mul(expr1=self, expr2=other)

    def __truediv__(self, other):
        return Div(expr1=self, expr2=other)

    def __pow__(self, other):
        return Pow(expr1=self, expr2=other)

    def __matmul__(self, other):
        return Kron(expr1=self, expr2=other)

    def __not__(self, other):
        return Not(expr1=self, expr2=other)

    def __or__(self, other):
        return Or(expr1=self, expr2=other)

    def __and__(self, other):
        return And(expr1=self, expr2=other)

    def __eq__(self, other):
        return Eq(expr1=self, expr2=other)

    def __neq__(self, other):
        return Neq(expr1=self, expr2=other)

    def __lt__(self, other):
        return Lt(expr1=self, expr2=other)

    def __leq__(self, other):
        return Leq(expr1=self, expr2=other)

    def __gt__(self, other):
        return Gt(expr1=self, expr2=other)

    def __geq__(self, other):
        return Geq(expr1=self, expr2=other)

    def __radd__(self, other):
        return AnalogExpr.cast(self).__add__(other)

    def __rsub__(self, other):
        return AnalogExpr.cast(self).__sub__(other)

    def __rmul__(self, other):
        return AnalogExpr.cast(self).__mul__(other)

    def __rtruediv__(self, other):
        return AnalogExpr.cast(self).______truediv__(other)

    def __rpow__(self, other):
        return AnalogExpr.cast(self).__pow__(other)

    def __rmatmul__(self, other):
        return AnalogExpr.cast(self)._____matmul__(other)

    def __rnot__(self, other):
        return AnalogExpr.cast(self).__not__(other)

    def __ror__(self, other):
        return AnalogExpr.cast(self).__or__(other)

    def __rand__(self, other):
        return AnalogExpr.cast(self).__and__(other)

    def __req__(self, other):
        return AnalogExpr.cast(self).__eq__(other)

    def __rneq__(self, other):
        return AnalogExpr.cast(self).__neq__(other)

    def __rlt__(self, other):
        return AnalogExpr.cast(self).__lt__(other)

    def __rleq__(self, other):
        return AnalogExpr.cast(self).__leq__(other)

    def __rgt__(self, other):
        return AnalogExpr.cast(self).__gt__(other)

    def __rgeq__(self, other):
        return AnalogExpr.cast(self).__geq__(other)


class AbstractAnalogExpr: ...


########################################################################################


def _is_varname(value: str) -> str:
    if not value.isidentifier():
        raise ValueError(f"{value!r} is not a valid identifier")
    return value


Identifier = Annotated[str, AfterValidator(_is_varname)]


class Access(AnalogExpr):
    name: Identifier


########################################################################################


class CollectionExpr(AbstractAnalogExpr): ...


class IndexingExpr(AbstractAnalogExpr): ...


class AnalogList(CollectionExpr, AnalogExpr):
    values: List[CastAnalogExpr]


class Extract(IndexingExpr, AnalogExpr):
    access: Access
    index: CastAnalogExpr


########################################################################################


class Complex(VisitableBaseModel):
    real: float
    imag: float


class Constant(AnalogExpr):
    value: Union[int, float, Complex, RuntimeVar, bool]


def _is_runtimevar(value: str) -> str:
    if not value.startswith("#") or len(value) < 2 or not value[1:].isidentifier():
        raise ValueError(
            "MathVar variable must start with a '#', followed by a valid identifier"
        )
    return value


RuntimeVarName = Annotated[str, AfterValidator(_is_runtimevar)]


class RuntimeVar(AnalogExpr):
    name: RuntimeVarName


########################################################################################


class UnaryOp(AbstractAnalogExpr):
    expr: CastAnalogExpr


class BinaryOp(AbstractAnalogExpr):
    """
    Class representing binary operations on [`MathExprs`][oqd_core.interface.analog.expr.MathExpr] abstract syntax tree (AST)
    """

    expr1: CastAnalogExpr
    expr2: CastAnalogExpr


class ArithOp(AbstractAnalogExpr): ...


class BoolOp(AbstractAnalogExpr): ...


class Neg(UnaryOp, ArithOp, AnalogExpr): ...


class Pos(UnaryOp, ArithOp, AnalogExpr): ...


class Add(BinaryOp, ArithOp, AnalogExpr):
    """
    Class representing the addition of [`MathExprs`][oqd_core.interface.analog.expr.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.analog.expr.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.analog.expr.MathExpr]
    """


class Sub(BinaryOp, ArithOp, AnalogExpr):
    """
    Class representing the subtraction of [`MathExprs`][oqd_core.interface.analog.expr.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.analog.expr.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.analog.expr.MathExpr]
    """


class Mul(BinaryOp, ArithOp, AnalogExpr):
    """
    Class representing the multiplication of [`MathExprs`][oqd_core.interface.analog.expr.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.analog.expr.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.analog.expr.MathExpr]
    """


class Div(BinaryOp, ArithOp, AnalogExpr):
    """
    Class representing the division of [`MathExprs`][oqd_core.interface.analog.expr.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.analog.expr.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.analog.expr.MathExpr]
    """


class Pow(BinaryOp, ArithOp, AnalogExpr):
    """
    Class representing the exponentiation of [`MathExprs`][oqd_core.interface.analog.expr.MathExpr]

    Attributes:
        expr1 (MathExpr): Left hand side [`MathExpr`][oqd_core.interface.analog.expr.MathExpr]
        expr2 (MathExpr): Right hand side [`MathExpr`][oqd_core.interface.analog.expr.MathExpr]
    """


class Kron(BinaryOp, ArithOp, AnalogExpr):
    """
    Class representing the tensor product of [`Operators`][oqd_core.interface.analog.expr.OperatorExpr]

    Attributes:
        op1 (Operator): Left hand side [`Operator`][oqd_core.interface.analog.expr.OperatorExpr]
        op2 (Operator): Right hand side [`Operator`][oqd_core.interface.analog.expr.OperatorExpr]
    """


########################################################################################


class ComparisonOp(AbstractAnalogExpr):
    """
    Class representing binary operations on [`BoolExprs`][oqd_core.interface.analog.expr.BoolExpr] abstract syntax tree (AST)
    """


class Not(UnaryOp, BoolOp, AnalogExpr): ...


class And(BinaryOp, BoolOp, AnalogExpr): ...


class Or(BinaryOp, BoolOp, AnalogExpr): ...


class Xor(BinaryOp, BoolOp, AnalogExpr): ...


class Eq(BinaryOp, BoolOp, ComparisonOp, AnalogExpr): ...


class Neq(BinaryOp, BoolOp, ComparisonOp, AnalogExpr): ...


class Lt(BinaryOp, BoolOp, ComparisonOp, AnalogExpr): ...


class Leq(BinaryOp, BoolOp, ComparisonOp, AnalogExpr): ...


class Gt(BinaryOp, BoolOp, ComparisonOp, AnalogExpr): ...


class Geq(BinaryOp, BoolOp, ComparisonOp, AnalogExpr): ...


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
]
"""
List of supported functions
"""


class BuiltinCall(AnalogExpr):
    """
    Class representing a named function applied to a [`MathExpr`][oqd_core.interface.analog.expr.MathExpr] abstract syntax tree (AST)

    Attributes:
        func (SupportedFuncNames): Named function to apply
        expr (Union[CastMathExpr, List[CastMathExpr]]): Arguments of the named function
    """

    func: SupportedBuiltinFuncs
    args: List[CastAnalogExpr]


########################################################################################


class QuantumExpr(AbstractAnalogExpr): ...


class Operator(QuantumExpr, AbstractAnalogExpr): ...


class Pauli(Operator, AbstractAnalogExpr):
    """
    Class representing a Pauli operator
    """

    level1: CastAnalogExpr = Constant(value=0)
    level2: CastAnalogExpr = Constant(value=1)


class PauliI(Pauli, AnalogExpr):
    """
    Class for the Pauli I operator
    """


class PauliX(Pauli, AnalogExpr):
    """
    Class for the Pauli X operator
    """


class PauliY(Pauli, AnalogExpr):
    """
    Class for the Pauli Y operator
    """


class PauliZ(Pauli, AnalogExpr):
    """
    Class for the Pauli Z operator
    """


class Ladder(Operator, AbstractAnalogExpr):
    """
    Class representing a ladder operator in Fock space
    """


class Creation(Ladder, AnalogExpr):
    """
    Class for the Creation operator in Fock space
    """


class Annihilation(Ladder, AnalogExpr):
    """
    Class for the Annihilation operator in Fock space
    """


class Identity(Ladder, AnalogExpr):
    """
    Class for the Identity operator in Fock space
    """


DEFAULT_OP_MAPPING = {
    "%X": PauliX,
    "%Y": PauliY,
    "%Z": PauliZ,
    "%I": PauliI,
    "%A": Annihilation,
    "%C": Creation,
    "%J": Identity,
}

########################################################################################


class RegisterExpr(AbstractAnalogExpr): ...


class QuantumRegister(CollectionExpr, RegisterExpr, AnalogExpr):
    size: CastAnalogExpr


class ModeRegister(CollectionExpr, RegisterExpr, AnalogExpr):
    size: CastAnalogExpr


class Evolve(QuantumExpr, AnalogExpr):
    """
    Class representing an evolution by an analog gate in the analog circuit

    Attributes:
        hamiltonian (Expr): Function to evolve by
        duration (Expr): Duration of the evolution
        targets (Expr): Indices and Quantum objects on which to apply the Hamiltonian
    """

    hamiltonian: CastAnalogExpr
    duration: CastAnalogExpr
    targets: CastAnalogExpr


class Measure(QuantumExpr, AnalogExpr):
    """
    Class representing a measurement in the analog circuit
    """

    targets: CastAnalogExpr


class Initialize(QuantumExpr, AnalogExpr):
    """
    Class representing a initialization in the analog circuit
    """

    targets: CastAnalogExpr


########################################################################################

AnalogExprSubtypes = Annotated[
    Union[tuple(AnalogExpr.__subclasses__())],
    Discriminator(discriminator="class_"),
]


CastAnalogExpr = Annotated[AnalogExprSubtypes, BeforeValidator(AnalogExpr.cast)]
