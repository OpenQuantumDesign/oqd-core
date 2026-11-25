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
from pydantic import AfterValidator, BaseModel, BeforeValidator, Discriminator, Field
from typing_extensions import TypeAliasType

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
    value: CastAnalogExpr
    index: CastAnalogExpr


########################################################################################


class Complex(VisitableBaseModel):
    real: float
    imag: float

    @classmethod
    def cast(cls, value):
        match value:
            case Complex():
                return value
            case complex():
                return Complex(real=value.real, imag=value.imag)
            case float() | int():
                return Complex(real=value.real, imag=value.imag)
            case _:
                raise TypeError(
                    f"Unsupported type ({value.__class__.__name__}) for casting to Complex"
                )

    @property
    def value(self):
        return self.real + 1j * self.imag

    def __pos__(self):
        return self

    def __neg__(self):
        return self.cast(-self.value)

    def __add__(self, other):
        return self.cast(self.value + self.cast(other).value)

    def __sub__(self, other):
        return self.cast(self.value - self.cast(other).value)

    def __mul__(self, other):
        return self.cast(self.value * self.cast(other).value)

    def __truediv__(self, other):
        return self.cast(self.value / self.cast(other).value)

    def __pow__(self, other):
        return self.cast(self.value ** self.cast(other).value)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self.cast(self.cast(other).value - self.value)

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return self.cast(self.cast(other).value / self.value)

    def __rpow__(self, other):
        return self.cast(self.cast(other).value ** self.value)


class Constant(AnalogExpr):
    value: Union[int, float, Complex, RuntimeVar, bool]


def _is_runtimevar(value: str) -> str:
    if not value.startswith("#") or len(value) < 2 or not value[1:].isidentifier():
        raise ValueError(
            "RuntimeVar variable must start with a '#', followed by a valid identifier"
        )
    return value


RuntimeVarName = Annotated[str, AfterValidator(_is_runtimevar)]


class RuntimeVar(AnalogExpr):
    name: RuntimeVarName


########################################################################################


class UnaryOp(AbstractAnalogExpr):
    expr: CastAnalogExpr


ListCastAnalogExpr = TypeAliasType("ListCastAnalogExpr", "List[CastAnalogExpr]")


class BinaryOp(BaseModel, AbstractAnalogExpr):
    """
    Class representing binary operations on [`AnalogExprs`][oqd_core.interface.analog.expr.AnalogExpr] abstract syntax tree (AST)
    """

    exprs: ListCastAnalogExpr


class ArithOp(AbstractAnalogExpr): ...


class BoolOp(AbstractAnalogExpr): ...


class Neg(UnaryOp, ArithOp, AnalogExpr): ...


class Pos(UnaryOp, ArithOp, AnalogExpr): ...


class Add(BinaryOp, ArithOp, AnalogExpr): ...


class Sub(BinaryOp, ArithOp, AnalogExpr): ...


class Mul(BinaryOp, ArithOp, AnalogExpr): ...


class Div(BinaryOp, ArithOp, AnalogExpr): ...


class Pow(BinaryOp, ArithOp, AnalogExpr): ...


class Kron(BinaryOp, ArithOp, AnalogExpr): ...


########################################################################################


class ComparisonOp(AbstractAnalogExpr): ...


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
    "min",
    "max",
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
    "commutator",
    "acommutator",
    "dag",
    "_pos",
    "_neg",
    "_add",
    "_sub",
    "_mul",
    "_div",
    "_pow",
    "_kron",
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


class BuiltinCall(AnalogExpr):
    """
    Class representing a named function applied to an [`AnalogExpr`][oqd_core.interface.analog.expr.AnalogExpr] abstract syntax tree (AST) node.

    Attributes:
        func (SupportedFuncNames): Named function to apply
        args (List[CastAnalogExpr]): Arguments of the named function
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
    dims: CastAnalogExpr = Constant(value=2)
    target: CastAnalogExpr = Constant(value=0)


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

    dims: CastAnalogExpr = Constant(value=-1)
    target: CastAnalogExpr = Constant(value=0)

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
    dim: CastAnalogExpr = Constant(value=2)


class ModeRegister(CollectionExpr, RegisterExpr, AnalogExpr):
    size: CastAnalogExpr


class Evolve(QuantumExpr, AnalogExpr):
    """
    Class representing an evolution by an analog gate in the analog circuit

    Attributes:
        hamiltonian (CastAnalogExpr): Function to evolve by
        duration (CastAnalogExpr): Duration of the evolution
        targets (CastAnalogExpr): Indices and Quantum objects on which to apply the Hamiltonian
    """

    hamiltonian: CastAnalogExpr
    jumps: CastAnalogExpr = Field(default_factory=lambda: AnalogList(values=[]))
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
