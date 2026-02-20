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

from typing import List, Optional

from oqd_compiler_infrastructure import VisitableBaseModel
from pydantic import ValidationInfo, conint, field_validator

########################################################################################

__all__ = [
    "QuantumBit",
    "ClassicalBit",
    "QuantumRegister",
    "ClassicalRegister",
    "QuantumRef",
    "ClassicalRef"
]

########################################################################################

class QuantumRef(VisitableBaseModel):
    name: str
    index: Optional[conint(ge=0)] = None

class ClassicalRef(VisitableBaseModel):
    name: str
    index: Optional[conint(ge=0)] = None

class QuantumBit(VisitableBaseModel):
    id: str
    index: conint(ge=0)  # a non-negative integer


class ClassicalBit(VisitableBaseModel):
    id: str
    index: conint(ge=0)  # a non-negative integer


class QuantumRegister(VisitableBaseModel):
    id: str = "q"
    reg: List[QuantumBit]

    @field_validator("reg", mode="before")
    @classmethod
    def convert_reg(cls, v, info: ValidationInfo):
        if isinstance(v, int):
            id = info.data.get("id")
            v = [QuantumBit(id=id, index=i) for i in range(v)]
        return v

    # def __getitem__(self, item):
    #     return QuantumRegister(id=self.id, reg=[self.reg[item]])

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            return QuantumRegister(id=self.id, reg=self.reg[start:stop:step])
        else:
            return self.reg[key]


class ClassicalRegister(VisitableBaseModel):
    id: str = "c"
    reg: List[ClassicalBit]

    @field_validator("reg", mode="before")
    @classmethod
    def convert_reg(cls, v, info: ValidationInfo):
        if isinstance(v, int):
            id = info.data.get("id")
            v = [ClassicalBit(id=id, index=i) for i in range(v)]
        return v

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            return ClassicalRegister(id=self.id, reg=self.reg[start:stop:step])
        else:
            return self.reg[key]
