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

from typing import List, Union

from oqd_compiler_infrastructure import VisitableBaseModel
from pydantic import ConfigDict, field_validator, model_validator

########################################################################################
from .gate import Gate
from .register import ClassicalRegister, QuantumRegister
from .statement import Statement
from .declaration import QuantumDeclaration, ClassicalDeclaration, AliasDeclaration

########################################################################################

__all__ = [
    "DigitalCircuit",
]

########################################################################################

Declaration = Union[QuantumDeclaration, ClassicalDeclaration, AliasDeclaration]

class DigitalCircuit(VisitableBaseModel):
    model_config = ConfigDict(extra="forbid")

    qreg: List[QuantumRegister] = []
    creg: List[ClassicalRegister] = []

    declarations: List[Declaration] = []
    sequence: List[Union[Gate, Statement]] = []

    @field_validator("creg", mode="before")
    @classmethod
    def convert_creg(cls, v):
        if isinstance(v, int):
            v = [ClassicalRegister(reg=v)]
        elif isinstance(v, ClassicalRegister):
            v = [v]
        return v

    @field_validator("qreg", mode="before")
    @classmethod
    def convert_qreg(cls, v):
        if isinstance(v, int):
            v = [QuantumRegister(reg=v)]
        elif isinstance(v, QuantumRegister):
            v = [v]
        return v

    @model_validator(mode="after")
    @classmethod
    def validate_ids(cls, data):
        ids = []
        for creg in data.creg:
            ids.append(creg.id)
        for qreg in data.qreg:
            ids.append(qreg.id)

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Found multiple registers with the same identifier, register identifier must be unique"
            )

        return data

    def add(self, op: Union[Gate, Statement]):
        self.sequence.append(op)

    @property
    def qasm(self):
        version = "2.0"
        header = 'include "qelib1.inc";'

        qasm_str = f"OPENQASM {version};\n"
        qasm_str += f"{header};\n"

        for qreg in self.qreg:
            qasm_str += f"qreg {qreg.id}[{len(qreg.reg)}];\n"

        for creg in self.creg:
            qasm_str += f"creg {creg.id}[{len(creg.reg)}];\n"

        for decl in self.declarations:
            if isinstance(decl, QuantumDeclaration):
                qasm_str += f"qreg {decl.name}[{decl.size}];\n"
            elif isinstance(decl, ClassicalDeclaration):
                qasm_str += f"creg {decl.name}[{decl.size}];\n"
            elif isinstance(decl, AliasDeclaration):
                qasm_str += f"let {decl.name} = {decl.target.name}[{decl.begin}:{decl.end}];\n"

        for op in self.sequence:
            if isinstance(op, Gate):
                qasm_str += op.qasm
            elif isinstance(op, Statement):
                qasm_str += op.qasm

        return qasm_str
