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

from typing import Generic, Type, TypeVar, _GenericAlias

from oqd_compiler_infrastructure.lattice import (
    LatticeBase,
    LatticeTop,
    MapLatticeValue,
)

########################################################################################


class AtomicTypeError(TypeError):
    """Type Error class for Atomic."""

    pass

########################################################################################

class TAtomic(LatticeTop): ...


class TInvalid(TAtomic): ...


LatticeValueTypeVar = TypeVar("LatticeValueTypeVar", bound=TAtomic)


class TList(TAtomic, Generic[LatticeValueTypeVar]): ...


class TInt(TAtomic): ...


class TFloat(TInt): ...


class TComplex(TFloat): ...


class TBool(TAtomic): ...


class TIRegElem(TAtomic): ...


class TIReg(TAtomic): ...


class TNull(TAtomic): ...


class TBeam(TAtomic): ...


class TPulse(TAtomic): ...


TLatticeValue = Type[TAtomic]

TypeEnv = MapLatticeValue[TLatticeValue]

########################################################################################


def get_type_name(value: TLatticeValue):
    if issubclass(type(value), _GenericAlias):
        return f"{value.__name__}[{','.join(map(get_type_name, value.__args__))}]"
    return value.__name__


def isTList(value: TLatticeValue):
    if issubclass(type(value), _GenericAlias) and value.__origin__ is TList:
        return True
    return False


########################################################################################

class AtomicTypeLattice(LatticeBase[TLatticeValue]):
    """Type lattice for atomic expressions."""

    # TODO: Add test for AtomicTypeLattice especially with TList interaction with other LatticeValues

    def top(self):
        return TAtomic

    def bottom(self):
        return TInvalid

    def leq(self, t1: TLatticeValue, t2: TLatticeValue) -> bool:
        if isTList(t1) and isTList(t2):
            return self.leq(t1.__args__[0], t2.__args__[0])

        t1 = TList if isTList(t1) else t1
        t2 = TList if isTList(t2) else t2

        return super().leq(t1, t2)

    def join(self, t1: TLatticeValue, t2: TLatticeValue) -> TLatticeValue:
        if isTList(t1) and isTList(t2):
            return TList[self.join(t1.__args__[0], t2.__args__[0])]

        if isTList(t1) and super().leq(t2, TList):
            return t1

        if isTList(t2) and super().leq(t1, TList):
            return t2

        return super().join(t1, t2)

    def meet(self, t1: TLatticeValue, t2: TLatticeValue) -> TLatticeValue:
        if isTList(t1) and isTList(t2):
            return TList[self.meet(t1.__args__[0], t2.__args__[0])]

        if isTList(t1) and super().leq(TList, t2):
            return t1

        if isTList(t2) and super().leq(TList, t1):
            return t2

        return super().meet(t1, t2)


########################################################################################


VariableType = TypeVar("VariableType", bound=TAtomic)


ATOMIC_SUPPORTED_FUNC_SIGNATURES = {
    "Not": [((TBool,), TBool)],
    "And": [((TBool, TBool), TBool)],
    "Xor": [((TBool, TBool), TBool)],
    "Or": [((TBool, TBool), TBool)],
    "Eq": [((VariableType, VariableType), TBool)],
    "Neq": [((VariableType, VariableType), TBool)],
    "Lt": [((TFloat, TFloat), TBool)],
    "Leq": [((TFloat, TFloat), TBool)],
    "Gt": [((TFloat, TFloat), TBool)],
    "Geq": [((TFloat, TFloat), TBool)],
    "Neg": [
        ((TInt,), TInt),
        ((TFloat,), TFloat),
        ((TComplex,), TComplex),
    ],
    "Pos": [
        ((TInt,), TInt),
        ((TFloat,), TFloat),
        ((TComplex,), TComplex),
    ],
    "Add": [
        ((TInt, TInt), TInt),
        ((TFloat, TFloat), TFloat),
        ((TComplex, TComplex), TComplex),
    ],
    "Sub": [
        ((TInt, TInt), TInt),
        ((TFloat, TFloat), TFloat),
        ((TComplex, TComplex), TComplex),
    ],
    "Mul": [
        ((TInt, TInt), TInt),
        ((TFloat, TFloat), TFloat),
        ((TComplex, TComplex), TComplex),
    ],
    "Div": [
        ((TInt, TInt), TFloat),
        ((TFloat, TFloat), TFloat),
        ((TComplex, TComplex), TComplex),
    ],
    "Pow": [
        ((TInt, TInt), TInt),
        ((TFloat, TFloat), TFloat),
        ((TComplex, TComplex), TComplex),
    ],
    "Beam": [((TFloat, TFloat, TFloat, TList[TFloat], TList[TFloat]), TBeam)],
    "Pulse": [
        ((TBeam, TFloat, TIReg,), TPulse),
        ((TBeam, TFloat, TIRegElem,), TPulse),
        ((TBeam, TFloat, TList[TIRegElem],), TPulse),
        ((TBeam, TFloat, TIReg, TBool), TPulse),
        ((TBeam, TFloat, TIRegElem, TBool), TPulse),
        ((TBeam, TFloat, TList[TIRegElem], TBool), TPulse),
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
    "round": [((TInt,), TInt), ((TFloat,), TInt)],
    "len": [((TList[VariableType],), TInt)],
    "range": [
        ((TInt,), TList[TInt]),
        ((TFloat,), TList[TFloat]),
        ((TInt, TInt), TList[TInt]),
        ((TFloat, TFloat), TList[TFloat]),
        ((TInt, TInt, TInt), TList[TInt]),
        ((TFloat, TFloat, TFloat), TList[TFloat]),
    ],
    "IonRegister": [((TInt,), TIReg)],
    "Extract": [
        ((TIReg, TInt), TIRegElem),
        ((TList[VariableType], TInt), VariableType),
    ],
    "flatten": [((TList[TList[VariableType]],), TList[VariableType])],
    "print": [((VariableType,), TNull)],
}
