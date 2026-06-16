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

from oqd_compiler_infrastructure import Chain, FixedPoint, In, Post, Pre

########################################################################################
from oqd_core.compiler.analog.rewrite.canonicalize import (
    GatherMathExpr,
    GatherPauli,
    NormalOrder,
    OperatorDistribute,
    PauliAlgebra,
    ProperOrder,
    PruneIdentity,
    PruneZeros,
    ScaleTerms,
    SortedOrder,
)
from oqd_core.compiler.analog.verify.canonicalize import (
    CanVerGatherMathExpr,
    CanVerGatherPauli,
    CanVerNormalOrder,
    CanVerOperatorDistribute,
    CanVerPauliAlgebra,
    CanVerProperOrder,
    CanVerPruneIdentity,
    CanVerScaleTerm,
    CanVerSortedOrder,
)
from oqd_core.compiler.analog.verify.operator import VerifyHilbertSpaceDim
from oqd_core.compiler.analog.error import CanonicalFormError
from oqd_core.compiler.analog.math.passes import canonicalize_math_expr
from oqd_core.interface.analog import AnalogCircuit, Declaration, Evolve, IfElse, While
from oqd_core.interface.analog.expr import OperatorExpr, Access
from oqd_core.interface.analog.statement import Statement

########################################################################################

__all__ = [
    "analog_operator_canonicalization",
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


def resolve_operator_expr(expr, symbols: dict):
    if isinstance(expr, Access):
        if expr.name not in symbols:
            raise CanonicalFormError(f"Undefined access: {expr.name}")
        expr = symbols[expr.name]
        if isinstance(expr, Access):
            return resolve_operator_expr(expr, symbols)
        if not isinstance(expr, OperatorExpr):
            raise CanonicalFormError(f" Access {expr.name} is not an operator.")
    return expr
    

def canonicalize_stmt(stmt: Statement, symbols: dict) -> Statement:
    if isinstance(stmt, Declaration):
        if isinstance(stmt.value, OperatorExpr):
            stmt.value = analog_operator_canonicalization(stmt.value)
        symbols[stmt.name] = stmt.value
    elif isinstance(stmt, Evolve):
        resolved = resolve_operator_expr(stmt.hamiltonian, symbols)
        stmt.hamiltonian = analog_operator_canonicalization(resolved)
    elif isinstance(stmt, IfElse):
        stmt.then_branch = [canonicalize_stmt(s, symbols) for s in stmt.then_branch]
        stmt.else_branch = [canonicalize_stmt(s, symbols) for s in stmt.else_branch]
    elif isinstance(stmt, While):
        stmt.body = [canonicalize_stmt(s, symbols) for s in stmt.body]
    return stmt


def analog_operator_canonicalization(model):
    """
    This pass runs canonicalization chain for Operators with a verifies for canonicalization.

    Args:
        model (VisitableBaseModel):

    Returns:
        model (VisitableBaseModel):  [`Operator`][oqd_core.interface.analog.operator.Operator] of Analog level are in canonical form

    Assumptions:
        None

    Example:
        - for model = X@(Y + Z), output is 1*(X@Y) + 1 * (X@Z)
        - for model = [`AnalogGate`][oqd_core.interface.analog.operations.AnalogGate](hamiltonian = (A * J)@X), output is
            [`AnalogGate`][oqd_core.interface.analog.operations.AnalogGate](hamiltonian = 1 * (X@A))
            (where A = Annhiliation(), J = Identity() [Ladder])

    Acknowledgement:
        This code was inspired by [Liang.jl](https://github.com/Roger-luo/Liang.jl/blob/main/src/canonicalize/entry.jl#L8).
    """
    
    if isinstance(model, AnalogCircuit):
        symbols = {}
        model.statements = [canonicalize_stmt(s, symbols) for s in model.statements]
        return model
    
    return Chain(
        FixedPoint(dist_chain),
        FixedPoint(Post(ProperOrder())),
        FixedPoint(pauli_chain),
        FixedPoint(Post(GatherPauli())),
        In(VerifyHilbertSpaceDim(), reverse=True),
        FixedPoint(normal_order_chain),
        FixedPoint(Post(PruneIdentity())),
        FixedPoint(scale_terms_chain),
        FixedPoint(Post(SortedOrder())),
        canonicalize_math_expr,
        FixedPoint(Post(PruneZeros())),
        verify_canonicalization,
    )(model=model)
