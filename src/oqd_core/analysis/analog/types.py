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

from typing import Union

from oqd_compiler_infrastructure.lattice import (
    LatticeBase,
    LatticeBottom,
    LatticeTop,
)
from pydantic import BaseModel, ConfigDict

########################################################################################


class AnalogTypeError(TypeError):
    """Type Error class for Analog."""

    pass


class TList(LatticeTop, BaseModel):
    """Lattice value representing a list."""

    model_config = ConfigDict(frozen=True)
    elem: TLatticeValue


TLatticeValue = Union[TList, type[LatticeTop]]
TypeEnv = dict[str, TLatticeValue]


def type_name(t: TLatticeValue) -> str:
    """Format a lattice value into a readable type name for error messages."""
    if isinstance(t, TList):
        return f"TList[{type_name(t.elem)}]"
    if isinstance(t, type) and issubclass(t, LatticeTop):
        return t.__name__
    return str(t)


class TAnalog(LatticeTop): ...


class TScalar(TAnalog): ...


class TComplex(TScalar): ...


class TFloat(TComplex): ...


class TInt(TFloat): ...


class TBool(TAnalog): ...


class TOp(TAnalog): ...


class TTarget(TAnalog): ...


class TTargetRef(TTarget): ...


class TQReg(TTarget): ...


class TMReg(TTarget): ...


class TQRef(TTargetRef): ...


class TMRef(TTargetRef): ...


class TNull(TAnalog): ...


class AnalogTypeLattice(LatticeBase[TLatticeValue]):
    """Type lattice for analog expressions."""

    def leq(self, t1: TLatticeValue, t2: TLatticeValue) -> bool:
        if t1 is LatticeBottom:
            return True
        if isinstance(t1, TList) and isinstance(t2, TList):
            return self.leq(t1.elem, t2.elem)
        if isinstance(t1, TList) or isinstance(t2, TList):
            return False
        return super().leq(t1, t2)

    def join(self, t1: TLatticeValue, t2: TLatticeValue) -> TLatticeValue:
        if self.leq(t1, t2):
            return t2
        if self.leq(t2, t1):
            return t1
        if isinstance(t1, TList) and isinstance(t2, TList):
            return TList(elem=self.join(t1.elem, t2.elem))
        if isinstance(t1, TList) or isinstance(t2, TList):
            return TAnalog
        return super().join(t1, t2)

    def meet(self, t1: TLatticeValue, t2: TLatticeValue) -> TLatticeValue:
        if self.leq(t1, t2):
            return t1
        if self.leq(t2, t1):
            return t2
        if isinstance(t1, TList) and isinstance(t2, TList):
            return TList(elem=self.meet(t1.elem, t2.elem))
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
        ((TOp, TFloat, TTargetRef), TNull),
        ((TOp, TFloat, TTarget), TNull),
        ((TOp, TFloat, TList(elem=TTargetRef)), TNull),
    ],
    "Initialize": [
        ((TTargetRef,), TNull),
        ((TTarget,), TNull),
        ((TList(elem=TTargetRef),), TNull),
    ],
    "Measure": [
        ((TTargetRef,), TList(elem=TInt)),
        ((TTarget,), TList(elem=TInt)),
        ((TList(elem=TTargetRef),), TList(elem=TInt)),
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
