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
from oqd_core.interface.digital.register import (
    ClassicalRegister,
    QuantumRegister,
)

########################################################################################

__all__ = [
    "BuildLookup",
]

########################################################################################


class BuildLookup(RewriteRule):
    """
    Builds a lookup table from declarations, mapping variable names to concrete registers.

    Args:
        model (DigitalCircuit): The rule acts on [`DigitalCircuit`][oqd_core.interface.digital.circuit.DigitalCircuit] objects.

    Returns:
        model (DigitalCircuit)

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
