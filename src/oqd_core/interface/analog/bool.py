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

from typing import Union
from oqd_compiler_infrastructure import VisitableBaseModel
########################################################################################
from .register import ClassicalRef, ClassicalRegister

########################################################################################
__all__ = [
    "BoolExpr",
    "RegisterNonZero",
    "BitEquals",
    "BoolAnd",
    "BoolOr",
    "BoolNot",
    "BoolRef",
    "SSAValBool",
    "BoolExprSubtypes",
]

########################################################################################

class BoolExpr(VisitableBaseModel):
    pass

class RegisterNonZero(BoolExpr):
    creg: Union[ClassicalRegister, ClassicalRef]

class BitEquals(BoolExpr):
    creg: Union[ClassicalRegister, ClassicalRef]
    index: int
    value: int
    
class BoolAnd(BoolExpr):
    left: "BoolExprSubtypes"
    right: "BoolExprSubtypes"

class BoolOr(BoolExpr):
    left: "BoolExprSubtypes"
    right: "BoolExprSubtypes"
    
class BoolNot(BoolExpr):
    expr: "BoolExprSubtypes"

class BoolRef(BoolExpr):
    name: str

class SSAValBool(BoolExpr):
    name: str

BoolExprSubtypes = Union[
    RegisterNonZero,
    BitEquals,
    BoolAnd,
    BoolOr,
    BoolNot,
    BoolRef,
    SSAValBool,
]

