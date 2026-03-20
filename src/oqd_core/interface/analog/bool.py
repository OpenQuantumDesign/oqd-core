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
from .expression import Expr

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
]

########################################################################################
    
class BoolAnd(Expr):
    left: Expr
    right: Expr

class BoolOr(Expr):
    left: Expr
    right: Expr
    
class BoolNot(Expr):
    expr: Expr

class BoolEq(Expr):
    left: Expr
    right: Expr

class BoolNotEq(Expr):
    left: Expr
    right: Expr

class BoolLessThan(Expr):
    left: Expr
    right: Expr
    
class BoolLessThanEq(Expr):
    left: Expr
    right: Expr

class BoolGreaterThan(Expr):
    left: Expr
    right: Expr

class BoolGreaterThanEq(Expr):
    left: Expr
    right: Expr

class BoolTrue(Expr):
    pass

class BoolFalse(Expr):
    pass

