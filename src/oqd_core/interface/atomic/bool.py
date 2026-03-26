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
from typing import Any, Annotated, Union
from pydantic import BeforeValidator, Discriminator, Tag
from .expression import Expr, Access
from .math import CastMathExpr

########################################################################################
__all__ = [
    "BoolAnd",
    "BoolOr",
    "BoolNot",
    "BoolEq",
    "BoolNotEq",
    "BoolLessThan",
    "BoolLessThanEq",
    "BoolGreaterThan",
    "BoolGreaterThanEq",
    "BoolTrue",
    "BoolFalse",
    "BoolExprSubtypes",
    "BoolExpr",
    "CastBool",
]

########################################################################################


class BoolExpr(Expr):
    @classmethod
    def cast(cls, value: Any):
        if isinstance(value, dict):
            return value
        if isinstance(value, BoolExpr):
            return value
        if isinstance(value, Access):
            return value
        if value is True:
            return BoolTrue()
        if value is False:
            return BoolFalse()
        raise TypeError


CastBool = Annotated[Expr, BeforeValidator(BoolExpr.cast)]

class BoolAnd(BoolExpr):
    left: CastBool
    right: CastBool

class BoolOr(BoolExpr):
    left: CastBool
    right: CastBool
    
class BoolNot(BoolExpr):
    expr: CastBool

class BoolEq(BoolExpr):
    left: ComparisonOperand
    right: ComparisonOperand

class BoolNotEq(BoolExpr):
    left: ComparisonOperand
    right: ComparisonOperand

class BoolLessThan(BoolExpr):
    left: ComparisonOperand
    right: ComparisonOperand
    
class BoolLessThanEq(BoolExpr):
    left: ComparisonOperand
    right: ComparisonOperand

class BoolGreaterThan(BoolExpr):
    left: ComparisonOperand
    right: ComparisonOperand

class BoolGreaterThanEq(BoolExpr):
    left: ComparisonOperand
    right: ComparisonOperand

class BoolTrue(BoolExpr):
    pass

class BoolFalse(BoolExpr):
    pass


ComparisonOperand = Union[CastMathExpr, CastBool, Access, BoolTrue, BoolFalse]

BoolExprSubtypes = Annotated[
    Union[
        Annotated[BoolAnd, Tag("BoolAnd")],
        Annotated[BoolOr, Tag("BoolOr")],
        Annotated[BoolNot, Tag("BoolNot")],
        Annotated[BoolEq, Tag("BoolEq")],
        Annotated[BoolNotEq, Tag("BoolNotEq")],
        Annotated[BoolLessThan, Tag("BoolLessThan")],
        Annotated[BoolLessThanEq, Tag("BoolLessThanEq")],
        Annotated[BoolGreaterThan, Tag("BoolGreaterThan")],
        Annotated[BoolGreaterThanEq, Tag("BoolGreaterThanEq")],
        Annotated[BoolTrue, Tag("BoolTrue")],
        Annotated[BoolFalse, Tag("BoolFalse")],
    ],
    Discriminator(
        lambda v: v["class_"] if isinstance(v, dict) else getattr(v, "class_")
    ),
]

