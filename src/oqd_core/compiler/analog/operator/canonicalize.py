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


from oqd_compiler_infrastructure import Chain, FixedPoint, Pre, Post
from oqd_core.compiler.analog.rewrite.canonicalize import (
    GatherMathExpr, GatherPauli, NormalOrder, OperatorDistribute,
    PauliAlgebra, ProperOrder, PruneIdentity, PruneZeros, ScaleTerms, SortedOrder,
)
from oqd_core.compiler.analog.verify.canonicalize import (
    CanVerGatherMathExpr, CanVerGatherPauli, CanVerNormalOrder,
    CanVerOperatorDistribute, CanVerPauliAlgebra, CanVerProperOrder,
    CanVerPruneIdentity, CanVerScaleTerm, CanVerSortedOrder,
)
from oqd_core.compiler.analog.operator.dim import operator_dim
from oqd_core.compiler.analog.error import AnalogCompilerError
from oqd_core.compiler.analog.math.passes import canonicalize_math_expr
from oqd_core.interface.analog.expr import OperatorExpr, Access

########################################################################################

__all__ = [
    "resolve_operator_expr",
    "canonicalize_operator_expr",
]

########################################################################################

dist_chain = Chain(
    FixedPoint(Post(OperatorDistribute())),
    FixedPoint(Post(GatherMathExpr())),
    FixedPoint(Post(OperatorDistribute())),
)

pauli_chain = Chain(
    FixedPoint(Post(PauliAlgebra())),
    FixedPoint(Post(GatherMathExpr())),
    FixedPoint(Post(PauliAlgebra())),
)

normal_order_chain = Chain(
    FixedPoint(Post(NormalOrder())),
    FixedPoint(Post(OperatorDistribute())),
    FixedPoint(Post(GatherMathExpr())),
    FixedPoint(Post(ProperOrder())),
    FixedPoint(Post(NormalOrder())),
)

scale_terms_chain = Chain(
    FixedPoint(Pre(ScaleTerms())),
    FixedPoint(Post(GatherMathExpr())),
)

verify_canonicalization = Chain(
    Post(CanVerOperatorDistribute()),
    Post(CanVerGatherMathExpr()),
    Post(CanVerProperOrder()),
    Post(CanVerPauliAlgebra()),
    Post(CanVerGatherPauli()),
    Post(CanVerNormalOrder()),
    Post(CanVerPruneIdentity()),
    Post(CanVerSortedOrder()),
    Pre(CanVerScaleTerm()),
)

def verify_operator_dim(expr):
    operator_dim(expr)
    return expr

def resolve_operator_expr(expr, symbols: dict):
    if isinstance(expr, Access):
        if expr.name not in symbols:
            raise AnalogCompilerError(f"Undefined access: {expr.name}")
        expr = symbols[expr.name]
        if isinstance(expr, Access):
            return resolve_operator_expr(expr, symbols)
        if not isinstance(expr, OperatorExpr):
            raise AnalogCompilerError(f" Access {expr.name} is not an operator.")
    return expr
    
def canonicalize_operator_expr(model):
    return Chain(
        FixedPoint(dist_chain),
        FixedPoint(Post(ProperOrder())),
        FixedPoint(pauli_chain),
        FixedPoint(Post(GatherPauli())),
        FixedPoint(normal_order_chain),
        FixedPoint(Post(PruneIdentity())),
        FixedPoint(scale_terms_chain),
        FixedPoint(Post(SortedOrder())),
        verify_operator_dim,
        canonicalize_math_expr,
        FixedPoint(Post(PruneZeros())),
        verify_canonicalization,
    )(model=model)


