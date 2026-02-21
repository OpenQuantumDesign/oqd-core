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

from typing import List, Literal, Union

from oqd_compiler_infrastructure import TypeReflectBaseModel, VisitableBaseModel

########################################################################################
from oqd_core.interface.math import MathExprSubtypes, SSAValMath
from .bool import BoolExprSubtypes, SSAValBool
from .operation import Evolve, Initialize, Measure, Declaration
from .register import QuantumRegister, ClassicalRegister

########################################################################################

__all__ = [
    "SSADefBool",
    "SSADefMath",
    "Terminator",
    "Branch",
    "CondBranch",
    "Exit",
    "Block",
    "AnalogCircuitSSA",
]

########################################################################################

class SSADefBool(VisitableBaseModel):
    name: str
    expr: BoolExprSubtypes

class SSADefMath(VisitableBaseModel):
    name: str
    expr: MathExprSubtypes

class Terminator(VisitableBaseModel):
    pass
    
class Branch(Terminator):
    target: str
    args: List[Union[MathExprSubtypes, BoolExprSubtypes]] = []

class CondBranch(Terminator):
    condition: BoolExprSubtypes
    true_target: str
    true_args: List[Union[MathExprSubtypes, BoolExprSubtypes]] = []
    false_target: str
    false_args: List[Union[MathExprSubtypes, BoolExprSubtypes]] = []

class Exit(Terminator):
    pass

BlockBodyItem = Union[SSADefBool, SSADefMath, Evolve, Measure, Initialize]

class Block(VisitableBaseModel):
    label: str
    args: List[str]
    body: List[BlockBodyItem]
    terminator: Terminator

class AnalogCircuitSSA(VisitableBaseModel):
    qreg: List["QuantumRegister"] = []
    creg: List["ClassicalRegister"] = []
    declarations: List["Declaration"] = []
    blocks: List[Block] = []
    