# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union
from oqd_compiler_infrastructure import VisitableBaseModel
from oqd_core.interface.math import MathExprSubtypes

########################################################################################
from .operator import OperatorSubtypes
from .register import QuantumRef, ClassicalRef
from .bool import BoolExprSubtypes

########################################################################################
__all__ = [
    "QuantumDeclaration",
    "ClassicalDeclaration",
    "AliasDeclaration",
    "OperatorDeclaration",
    "BoolDeclaration",
    "MathExprDeclaration"
]

########################################################################################

class QuantumDeclaration(VisitableBaseModel):
    name: str
    size: int

class ClassicalDeclaration(VisitableBaseModel):
    name: str
    size: int

class AliasDeclaration(VisitableBaseModel):
    name: str
    target: Union[QuantumRef, ClassicalRef]
    begin: Optional[int] = None
    end: Optional[int] = None

class OperatorDeclaration(VisitableBaseModel):
    name: str
    operator: OperatorSubtypes

class BoolDeclaration(VisitableBaseModel):
    name: str
    expr: BoolExprSubtypes

class MathExprDeclaration(VisitableBaseModel):
    name: str
    expr: MathExprSubtypes
