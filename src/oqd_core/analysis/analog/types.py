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

from typing import Generic, TypeVar, Union, _GenericAlias

from oqd_compiler_infrastructure.lattice import (
    LatticeBase,
    LatticeTop,
)

from oqd_core.analysis.utils import all_subclasses

########################################################################################


class AnalogTypeError(TypeError):
    """Type Error class for Analog."""

    pass


########################################################################################


class TLatticeTop(LatticeTop): ...


class TLatticeBottom(TLatticeTop): ...


class TAnalog(LatticeTop): ...


LatticeValueTypeVar = TypeVar("LatticeValueTypeVar", bound=TLatticeTop)


class TList(TAnalog, Generic[LatticeValueTypeVar]): ...


class TScalar(TAnalog): ...


class TComplex(TScalar): ...


class TFloat(TComplex): ...


class TInt(TFloat): ...


class TBool(TAnalog): ...


class TOp(TAnalog): ...


class TQReg(TAnalog): ...


class TQRegElem(TQReg): ...


class TNull(TAnalog): ...


TLatticeValue = Union[all_subclasses(TLatticeTop)]
TypeEnv = dict[str, TLatticeValue]


def get_type_name(value: TLatticeValue):
    if issubclass(type(value), _GenericAlias):
        return f"{value.__name__[1:]}[{','.join(map(get_type_name, value.__args__))}]"

    return value.__name__[1:]


def isTList(value: TLatticeValue):
    if issubclass(type(value), _GenericAlias) and value.__origin__ is TList:
        return True
    return False


class AnalogTypeLattice(LatticeBase[TLatticeValue]):
    """Type lattice for analog expressions."""

    def top(self):
        return TLatticeTop

    def bottom(self):
        return TLatticeBottom

    def leq(self, t1: TLatticeValue, t2: TLatticeValue) -> bool:
        if t1 is TLatticeBottom:
            return True
        if isTList(t1) and isTList(t2):
            return self.leq(t1.__args__[0], t2.__args__[0])
        if isTList(t1) or isTList(t2):
            return False
        return super().leq(t1, t2)

    def join(self, t1: TLatticeValue, t2: TLatticeValue) -> TLatticeValue:
        if self.leq(t1, t2):
            return t2
        if self.leq(t2, t1):
            return t1
        if isTList(t1) and isTList(t2):
            return TList[self.join(t1.__args__[0], t2.__args__[0])]
        if isTList(t1) or isTList(t2):
            return TAnalog
        return super().join(t1, t2)

    def meet(self, t1: TLatticeValue, t2: TLatticeValue) -> TLatticeValue:
        if self.leq(t1, t2):
            return t1
        if self.leq(t2, t1):
            return t2
        if isTList(t1) and isTList(t2):
            return TList[self.meet(t1.__args__[0], t2.__args__[0])]
        return super().meet(t1, t2)


########################################################################################


SUPPORTED_FUNC_SIGNATURES = {
    "BoolNot": [((TBool,), TBool)],
    "BoolEq": [((TScalar, TScalar), TBool)],
    "BoolNotEq": [((TScalar, TScalar), TBool)],
    "BoolLessThan": [((TScalar, TScalar), TBool)],
    "BoolLessThanEq": [((TScalar, TScalar), TBool)],
    "BoolGreaterThan": [((TScalar, TScalar), TBool)],
    "BoolGreaterThanEq": [((TScalar, TScalar), TBool)],
    "MathAdd": [
        ((TInt, TInt), TInt),
        ((TFloat, TFloat), TFloat),
        ((TComplex, TComplex), TComplex),
    ],
    "MathSub": [
        ((TInt, TInt), TInt),
        ((TFloat, TFloat), TFloat),
        ((TComplex, TComplex), TComplex),
    ],
    "MathMul": [
        ((TInt, TInt), TInt),
        ((TFloat, TFloat), TFloat),
        ((TComplex, TComplex), TComplex),
        ((TScalar, TOp), TOp),
        ((TOp, TScalar), TOp),
    ],
    "MathDiv": [
        ((TInt, TInt), TFloat),
        ((TFloat, TFloat), TFloat),
        ((TComplex, TComplex), TComplex),
    ],
    "MathPow": [
        ((TInt, TInt), TInt),
        ((TFloat, TFloat), TFloat),
        ((TComplex, TComplex), TComplex),
    ],
    "Evolve": [
        ((TOp, TFloat, TQReg), TNull),
        ((TOp, TFloat, TList[TQRegElem]), TNull),
    ],
    "Initialize": [
        ((TQReg,), TNull),
        ((TList[TQRegElem],), TNull),
    ],
    "Measure": [
        ((TQReg,), TList[TInt]),
        ((TList[TQRegElem],), TList[TInt]),
    ],
    "abs": [((TInt,), TInt), ((TFloat,), TFloat), ((TComplex,), TFloat)],
    "sin": [((TFloat,), TFloat), ((TComplex,), TComplex)],
    "cos": [((TFloat,), TFloat), ((TComplex,), TComplex)],
    "tan": [((TFloat,), TFloat), ((TComplex,), TComplex)],
    "exp": [((TFloat,), TFloat), ((TComplex,), TComplex)],
    "log": [((TFloat,), TFloat), ((TComplex,), TComplex)],
    "sinh": [((TFloat,), TFloat), ((TComplex,), TComplex)],
    "cosh": [((TFloat,), TFloat), ((TComplex,), TComplex)],
    "tanh": [((TFloat,), TFloat), ((TComplex,), TComplex)],
    "atan": [((TFloat,), TFloat), ((TComplex,), TComplex)],
    "acos": [((TFloat,), TFloat), ((TComplex,), TComplex)],
    "asin": [((TFloat,), TFloat), ((TComplex,), TComplex)],
    "atanh": [((TFloat,), TFloat), ((TComplex,), TComplex)],
    "asinh": [((TFloat,), TFloat), ((TComplex,), TComplex)],
    "acosh": [((TFloat,), TFloat), ((TComplex,), TComplex)],
    "heaviside": [((TFloat,), TInt), ((TInt), TInt)],
    "conj": [((TComplex,), TComplex)],
    "real": [((TComplex,), TFloat)],
    "imag": [((TComplex,), TFloat)],
    "atan2": [((TFloat, TFloat), TFloat), ((TComplex, TComplex), TComplex)],
}
