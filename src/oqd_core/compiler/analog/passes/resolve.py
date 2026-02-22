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
from oqd_core.compiler.analog.rewrite.resolve import (
    BuildAnalogLookup,
    ResolveAnalogClassicalRef,
    ResolveAnalogQuantumRef,
    ResolveAnalogBoolRef,
    ResolveAnalogMathRef,
    VerifyNoUnresolvedAnalogRefs,
)
from oqd_core.interface.analog import AnalogCircuitSSA
from oqd_core.compiler.analog.rewrite.ssa import resolve_analog_ssa

########################################################################################

__all__ = [
    "resolve_analog_declarations",
    "resolve_analog",
]

########################################################################################

def resolve_analog_declarations(model):
    """
    This pass resolves all references in AnalogCircuit
    """
    lookup_builder = BuildAnalogLookup()
    Post(lookup_builder)(model)
    
    resolved_sequence = Post(ResolveAnalogBoolRef(lookup_builder.lookup))(model.sequence)
    resolved_sequence = Post(ResolveAnalogMathRef(lookup_builder.lookup))(resolved_sequence)
    resolved_sequence = Post(ResolveAnalogQuantumRef(lookup_builder.lookup))(resolved_sequence)
    resolved_sequence = Post(ResolveAnalogClassicalRef(lookup_builder.lookup))(resolved_sequence)

    Post(VerifyNoUnresolvedAnalogRefs())(resolved_sequence)

    return model.__class__(
        qreg=model.qreg,
        creg=model.creg,
        declarations=model.declarations,
        sequence=resolved_sequence,
        n_qreg=model.n_qreg,
        n_qmode=model.n_qmode,
    )

def resolve_analog(model):
    """
    Resolves references in analog circuits. Dispatches on circuit type.
    - AnalogCircuit: tree form (sequence, IfElse, While)
    - AnalogCircuitSSA: block form
    """
    if isinstance(model, AnalogCircuitSSA):
        return resolve_analog_ssa(model)
    return resolve_analog_declarations(model)
