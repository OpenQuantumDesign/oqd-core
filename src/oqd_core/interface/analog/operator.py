# Copyright 2024 Open Quantum Design

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

from typing import Union
from oqd_compiler_infrastructure import TypeReflectBaseModel

########################################################################################

from oqd_core.interface.math import (
    MathExpr,
    MathExprSubtypes,
    MathImag,
    MathMul,
    MathNum,
)

########################################################################################

__all__ = [
    "Operator",
    "OperatorTerminal",
    "Pauli",
    "PauliI",
    "PauliX",
    "PauliY",
    "PauliZ",
    "PauliPlus",
    "PauliMinus",
    "Ladder",
    "Creation",
    "Annihilation",
    "Identity",
    "OperatorBinaryOp",
    "OperatorAdd",
    "OperatorSub",
    "OperatorMul",
    "OperatorScalarMul",
    "OperatorKron",
    "OperatorSubtypes",
]


########################################################################################


class Operator(TypeReflectBaseModel):
    """
    Class representing the abstract syntax tree (AST) for a quantum operator
    """

    def __neg__(self):
        return OperatorScalarMul(op=self, expr=MathNum(value=-1))

    def __pos__(self):
        return self

    def __add__(self, other):
        return OperatorAdd(op1=self, op2=other)

    def __sub__(self, other):
        return OperatorSub(op1=self, op2=other)

    def __matmul__(self, other):
        if isinstance(other, MathExpr):
            raise TypeError(
                "Tried Kron product between Operator and MathExpr. "
                + "Scalar multiplication of MathExpr and Operator should be bracketed when perfoming Kron product."
            )
        return OperatorKron(op1=self, op2=other)

    def __mul__(self, other):
        if isinstance(other, Operator):
            return OperatorMul(op1=self, op2=other)
        else:
            other = MathExpr.cast(other)
            return OperatorScalarMul(op=self, expr=other)

    def __rmul__(self, other):
        other = MathExpr.cast(other)
        return self * other

    pass


########################################################################################


class OperatorTerminal(Operator):
    """
    Class representing a terminal in the [`Operator`][oqd_core.interface.analog.operator.Operator] abstract syntax tree (AST)
    """

    pass


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


def PauliPlus():
    """
    Function that constructs the Pauli + operator
    """
    return OperatorAdd(
        op1=PauliX(),
        op2=OperatorScalarMul(
            op=PauliY(), expr=MathMul(expr1=MathImag(), expr2=MathNum(value=1))
        ),
    )


def PauliMinus():
    """
    Function that constructs the Pauli - operator
    """
    return OperatorAdd(
        op1=PauliX(),
        op2=OperatorScalarMul(
            op=PauliY(), expr=MathMul(expr1=MathImag(), expr2=MathNum(value=-1))
        ),
    )


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


class OperatorScalarMul(Operator):
    """
    Class representing scalar multiplication of an [`Operator`][oqd_core.interface.analog.operator.Operator] and a
    [`MathExpr`][oqd_core.interface.math.MathExpr]

    Attributes:
        op (Operator): [`Operator`][oqd_core.interface.analog.operator.Operator] to multiply
        expr (MathExpr): [`MathExpr`][oqd_core.interface.math.MathExpr] to multiply by
    """

    op: OperatorSubtypes
    expr: MathExprSubtypes


class OperatorBinaryOp(Operator):
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

    op1: OperatorSubtypes
    op2: OperatorSubtypes


class OperatorSub(OperatorBinaryOp):
    """
    Class representing the subtraction of [`Operators`][oqd_core.interface.analog.operator.Operator]

    Attributes:
        op1 (Operator): Left hand side [`Operator`][oqd_core.interface.analog.operator.Operator]
        op2 (Operator): Right hand side [`Operator`][oqd_core.interface.analog.operator.Operator]
    """

    op1: OperatorSubtypes
    op2: OperatorSubtypes


class OperatorMul(OperatorBinaryOp):
    """
    Class representing the multiplication of [`Operators`][oqd_core.interface.analog.operator.Operator]

    Attributes:
        op1 (Operator): Left hand side [`Operator`][oqd_core.interface.analog.operator.Operator]
        op2 (Operator): Right hand side [`Operator`][oqd_core.interface.analog.operator.Operator]
    """

    op1: OperatorSubtypes
    op2: OperatorSubtypes


class OperatorKron(OperatorBinaryOp):
    """
    Class representing the tensor product of [`Operators`][oqd_core.interface.analog.operator.Operator]

    Attributes:
        op1 (Operator): Left hand side [`Operator`][oqd_core.interface.analog.operator.Operator]
        op2 (Operator): Right hand side [`Operator`][oqd_core.interface.analog.operator.Operator]
    """

    op1: OperatorSubtypes
    op2: OperatorSubtypes


########################################################################################

OperatorSubtypes = Union[
    PauliI,
    PauliX,
    PauliY,
    PauliZ,
    Creation,
    Annihilation,
    Identity,
    OperatorAdd,
    OperatorSub,
    OperatorMul,
    OperatorScalarMul,
    OperatorKron,
]
"""
Alias for the union of concrete Operator subtypes
"""
