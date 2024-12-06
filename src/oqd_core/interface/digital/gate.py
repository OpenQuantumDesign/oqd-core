# Copyright 2024 Open Quantum Design

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Union, Optional

from oqd_compiler_infrastructure import VisitableBaseModel

########################################################################################

from .register import (
    QuantumRegister,
    ClassicalRegister,
    QuantumBit,
    ClassicalBit,
)

########################################################################################

__all__ = ["GateParameters", "Gate"]

########################################################################################


class GateParameters(VisitableBaseModel):
    vals: List[Union[int, float]] = []


class Gate(VisitableBaseModel):
    name: str
    qreg: Optional[Union[QuantumRegister, QuantumBit]] = None
    creg: Optional[Union[ClassicalRegister, ClassicalBit]] = None
    params: Optional[GateParameters] = None

    @property
    def qasm(self):
        _str = f"{self.name} "
        if isinstance(self.qreg, QuantumBit):
            _str += f"{self.qreg.id}[{self.qreg.index}]"
        elif isinstance(self.qreg, QuantumRegister):
            _str += ",".join([f"{reg.id}[{reg.index}]" for reg in self.qreg.reg])
        _str += ";\n"
        return _str


def H(**kwargs):
    return Gate(name="h", **kwargs)


def CNOT(**kwargs):
    return Gate(name="cx", **kwargs)
