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

__all__ = [
    "ResolveQuantumRef",
    "ResolveClassicalRef",
    "VerifyNoUnresolvedRefs",
]

########################################################################################


class ResolveQuantumRef(RewriteRule):
    """
    Resolves [`QuantumRef`][oqd_core.interface.digital.register.QuantumRef] nodes to concrete
    [`QuantumBit`][oqd_core.interface.digital.register.QuantumBit] or
    [`QuantumRegister`][oqd_core.interface.digital.register.QuantumRegister].

    Args:
        model (DigitalCircuit): The rule acts on [`DigitalCircuit`][oqd_core.interface.digital.circuit.DigitalCircuit] objects.

    Returns:
        model (DigitalCircuit): All QuantumRef nodes are replaced with concrete registers.

    Assumptions:
        Lookup table has been built via [`BuildLookup`][oqd_core.compiler.digital.analysis.BuildLookup].
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


class ResolveClassicalRef(RewriteRule):
    """
    Resolves [`ClassicalRef`][oqd_core.interface.digital.register.ClassicalRef] nodes to concrete
    [`ClassicalBit`][oqd_core.interface.digital.register.ClassicalBit] or
    [`ClassicalRegister`][oqd_core.interface.digital.register.ClassicalRegister].

    Args:
        model (DigitalCircuit): The rule acts on [`DigitalCircuit`][oqd_core.interface.digital.circuit.DigitalCircuit] objects.

    Returns:
        model (DigitalCircuit): All ClassicalRef nodes are replaced with concrete registers.

    Assumptions:
        Lookup table has been built via [`BuildLookup`][oqd_core.compiler.digital.analysis.BuildLookup].
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


class VerifyNoUnresolvedRefs(RewriteRule):
    """
    Verifies that no unresolved [`QuantumRef`][oqd_core.interface.digital.register.QuantumRef] or
    [`ClassicalRef`][oqd_core.interface.digital.register.ClassicalRef] remain in the circuit.

    Args:
        model (DigitalCircuit): The rule acts on [`DigitalCircuit`][oqd_core.interface.digital.circuit.DigitalCircuit] objects.

    Returns:
        model (DigitalCircuit): unchanged

    Assumptions:
        [`ResolveQuantumRef`][oqd_core.compiler.digital.rules.ResolveQuantumRef],
        [`ResolveClassicalRef`][oqd_core.compiler.digital.rules.ResolveClassicalRef]

    Example:
        QuantumRef(name="q", index=0) => raises ValueError
    """

    def map_QuantumRef(self, model):
        raise ValueError(f"Unresolved quantum reference: {model.name}")

    def map_ClassicalRef(self, model):
        raise ValueError(f"Unresolved classical reference: {model.name}")
