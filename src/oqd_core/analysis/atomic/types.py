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

from oqd_core.interface.atomic import (
    BoolAnd,
    BoolGreaterThan,
    BoolGreaterThanEq,
    BoolLessThan,
    BoolLessThanEq,
    BoolOr,
    MathAdd,
    MathDiv,
    MathMul,
    MathPow,
    MathSub,
)

########################################################################################


class AtomicTypeError(TypeError):
    """Type Error class for Atomic."""

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


class TAtomic(LatticeTop):
    pass


class TScalar(TAtomic):
    pass


class TBool(TAtomic):
    pass


class TBeam(TAtomic):
    pass


class TPulse(TAtomic):
    pass


class TTarget(TAtomic):
    pass


class TTargetRef(TTarget):
    pass


class TIonReg(TTarget):
    pass


class TIonRef(TTargetRef):
    pass


class AtomicTypeLattice(LatticeBase[TLatticeValue]):
    """Type lattice for atomic expressions."""

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
            return TAtomic
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


# Binary expression signature table: node -> ((left_type, right_type), output_type)
BIN_SIG_TABLE = {
    MathAdd: ((TScalar, TScalar), TScalar),
    MathSub: ((TScalar, TScalar), TScalar),
    MathMul: ((TScalar, TScalar), TScalar),
    MathDiv: ((TScalar, TScalar), TScalar),
    MathPow: ((TScalar, TScalar), TScalar),
    BoolAnd: ((TBool, TBool), TBool),
    BoolOr: ((TBool, TBool), TBool),
    BoolLessThan: ((TScalar, TScalar), TBool),
    BoolLessThanEq: ((TScalar, TScalar), TBool),
    BoolGreaterThan: ((TScalar, TScalar), TBool),
    BoolGreaterThanEq: ((TScalar, TScalar), TBool),
}
