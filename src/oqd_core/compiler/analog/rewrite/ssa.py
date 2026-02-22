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

from oqd_core.interface.analog import AnalogCircuitSSA
from oqd_core.compiler.analog.rewrite.resolve import (
    BuildAnalogLookup,
    ResolveAnalogBoolRef,
    ResolveAnalogClassicalRef,
    ResolveAnalogMathRef,
    ResolveAnalogQuantumRef,
    VerifyNoUnresolvedAnalogRefs,
)
########################################################################################

__all__ = ["resolve_analog_ssa"]


########################################################################################

def resolve_analog_ssa(model):
    """
    This pass resolves all references in AnalogCircuitSSA Blocks
    """
    lookup_builder = BuildAnalogLookup()
    Post(lookup_builder)(model)

    resolved_blocks = []
    for block in model.blocks:
        resolved_body = Post(ResolveAnalogBoolRef(lookup_builder.lookup))(block.body)
        resolved_body = Post(ResolveAnalogMathRef(lookup_builder.lookup))(resolved_body)
        resolved_body = Post(ResolveAnalogQuantumRef(lookup_builder.lookup))(resolved_body)
        resolved_body = Post(ResolveAnalogClassicalRef(lookup_builder.lookup))(resolved_body)

        resolved_term = Post(ResolveAnalogBoolRef(lookup_builder.lookup))(block.terminator)
        resolved_term = Post(ResolveAnalogMathRef(lookup_builder.lookup))(resolved_term)
        resolved_term = Post(ResolveAnalogQuantumRef(lookup_builder.lookup))(resolved_term)
        resolved_term = Post(ResolveAnalogClassicalRef(lookup_builder.lookup))(resolved_term)

        resolved_blocks.append(
            block.__class__(
                label=block.label,
                args=block.args,
                body=resolved_body,
                terminator=resolved_term,
            )
        )

    Post(VerifyNoUnresolvedAnalogRefs())(resolved_blocks)

    return model.__class__(
        qreg=model.qreg,
        creg=model.creg,
        declarations=model.declarations,
        blocks=resolved_blocks,
    )