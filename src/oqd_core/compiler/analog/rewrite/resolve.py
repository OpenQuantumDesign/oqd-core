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

from oqd_compiler_infrastructure import RewriteRule


########################################################################################
from oqd_core.interface.analog.register import (
    ClassicalRegister,
    QuantumRegister,
)

########################################################################################

__all__ = [
    "BuildAnalogLookup",
    "ResolveAnalogQuantumRef",
    "ResolveAnalogClassicalRef",
    "VerifyNoUnresolvedAnalogRefs",
]

########################################################################################


class BuildAnalogLookup(RewriteRule):
    """
    Builds a lookup table for analog declarations, mapping variable names to concrete registers or operators

    Args:
        model (AnalogCircuit): The rule acts on [`AnalogCircuit`][oqd_core.interface.analog.operation.AnalogCircuit] objects.

    Returns:
        model (AnalogCircuit)

    Example:
        QuantumDeclaration(name="q", size=5) => lookup["q"] = QuantumRegister(id="q", reg=5)
    """

    def __init__(self):
        super().__init__()
        self.lookup = {}

    def map_QuantumDeclaration(self, model):
        if model.name in self.lookup:
            raise ValueError(f"Duplicate declaration: {model.name}")
        self.lookup[model.name] = QuantumRegister(id=model.name, reg=model.size)

    def map_ClassicalDeclaration(self, model):
        if model.name in self.lookup:
            raise ValueError(f"Duplicate declaration: {model.name}")
        self.lookup[model.name] = ClassicalRegister(id=model.name, reg=model.size)

    def map_AliasDeclaration(self, model):
        target = self.lookup.get(model.target.name)
        if target is None:
            raise ValueError(f"Undefined reference in alias: {model.target.name}")
        self.lookup[model.name] = target[model.begin : model.end]
    
    def map_OperatorDeclaration(self, model):
        if model.name in self.lookup:
            raise ValueError(f"Duplicate declaration: {model.name}")
        self.lookup[model.name] = model.operator


class ResolveAnalogQuantumRef(RewriteRule):
    """
    Resolves QuantumRef to QuantumBit or QuantumRegister
    """
    def __init__(self, lookup):
        super().__init__()
        self.lookup = lookup

    def map_QuantumRef(self, model):
        reg = self.lookup.get(model.name)
        if reg is None:
            raise ValueError(f"Undefined quantum reference: {model.name}")
        if model.index is not None:
            return reg[model.index]
        return reg


class ResolveAnalogClassicalRef(RewriteRule):
    """
    Resolves ClassicalRef to ClassicalBit or ClassicalRegister
    """
    def __init__(self, lookup):
        super().__init__()
        self.lookup = lookup

    def map_ClassicalRef(self, model):
        reg = self.lookup.get(model.name)
        if reg is None:
            raise ValueError(f"Undefined classical reference: {model.name}")
        if model.index is not None:
            return reg[model.index]
        return reg

class VerifyNoUnresolvedAnalogRefs(RewriteRule):
    def map_QuantumRef(self, model):
        raise ValueError(f"Unresolved quantum reference: {model.name}")

    def map_ClassicalRef(self, model):
        raise ValueError(f"Unresolved classical reference: {model.name}")
