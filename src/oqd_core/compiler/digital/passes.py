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

from oqd_compiler_infrastructure import Post

########################################################################################
from oqd_core.compiler.digital.analysis import BuildLookup
from oqd_core.compiler.digital.rules import (
    ResolveClassicalRef,
    ResolveQuantumRef,
    VerifyNoUnresolvedRefs,
)

########################################################################################

__all__ = [
    "resolve_declarations",
]

########################################################################################


def resolve_declarations(model):
    """
    This pass resolves all symbolic references in a [`DigitalCircuit`][oqd_core.interface.digital.circuit.DigitalCircuit].

    Builds a lookup table from declarations, substitutes all
    [`QuantumRef`][oqd_core.interface.digital.register.QuantumRef] and
    [`ClassicalRef`][oqd_core.interface.digital.register.ClassicalRef] with concrete registers,
    and verifies no unresolved references remain.

    Args:
        model (DigitalCircuit): Circuit with symbolic references.

    Returns:
        model (DigitalCircuit): Circuit with all references resolved to concrete registers.

    Assumptions:
        None
    """
    lookup_builder = BuildLookup()
    Post(lookup_builder)(model)

    resolved_sequence = Post(ResolveQuantumRef(lookup_builder.lookup))(model.sequence)
    resolved_sequence = Post(ResolveClassicalRef(lookup_builder.lookup))(
        resolved_sequence
    )

    Post(VerifyNoUnresolvedRefs())(resolved_sequence)

    return model.__class__(
        qreg=model.qreg,
        creg=model.creg,
        declarations=model.declarations,
        sequence=resolved_sequence,
    )
