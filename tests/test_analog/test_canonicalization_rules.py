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

import pytest
from oqd_compiler_infrastructure import RewriteRule, WalkBase, Post, Pre, FixedPoint
from oqd_core.interface.analog import (
    Annihilation,
    Creation,
    Identity,
    Operator,
    PauliI,
    PauliX,
    PauliY,
    PauliZ,
)
from oqd_core.compiler.analog.rewrite.canonicalize import (
    GatherMathExpr,
    GatherPauli,
    NormalOrder,
    OperatorDistribute,
    PauliAlgebra,
    ProperOrder,
    PruneIdentity,
    ScaleTerms,
    SortedOrder,
)
from oqd_core.interface.math import MathStr

X, Y, Z, PI, A, C, LI = (
    PauliX(),
    PauliY(),
    PauliZ(),
    PauliI(),
    Annihilation(),
    Creation(),
    Identity(),
)


def canonicalize_operator(
    operator: Operator, rule: RewriteRule, walk_method: WalkBase = Post
):
    return FixedPoint(walk_method(rule))(operator)


@pytest.fixture
def operator_distribute_rule():
    return OperatorDistribute()


def test_operator_distribute_simple(operator_distribute_rule):
    op = X @ (X + Y)
    expected = X @ X + X @ Y
    assert canonicalize_operator(operator=op, rule=operator_distribute_rule) == expected


def test_operator_distribute_no_effect(operator_distribute_rule):
    op = 2 * (X @ Y) * (3 * (PI @ PI))
    expected = 2 * (X @ Y) * (3 * (PI @ PI))
    assert canonicalize_operator(operator=op, rule=operator_distribute_rule) == expected


@pytest.fixture
def gather_math_expr_rule():
    return GatherMathExpr()


def test_gather_math_expr_simple(gather_math_expr_rule):
    op = X @ (3 * Y) + (2 * X) @ Z
    expected = 3 * (X @ Y) + 2 * (X @ Z)
    assert canonicalize_operator(operator=op, rule=gather_math_expr_rule) == expected


def test_gather_math_expr_complicated(gather_math_expr_rule):
    op = X @ (3 * Y * (3 * Z)) @ (10 * PI) + (2 * X) @ Z @ (5 * Y)
    expected = (MathStr(string="3*3*10")) * ((X @ (Y * Z)) @ PI) + MathStr(
        string="2*5"
    ) * (X @ Z @ Y)
    assert canonicalize_operator(operator=op, rule=gather_math_expr_rule) == expected


def test_gather_math_expr_no_effect(gather_math_expr_rule):
    op = X @ (2 * X + 3 * Y)
    expected = X @ (2 * X + 3 * Y)
    assert canonicalize_operator(operator=op, rule=gather_math_expr_rule) == expected


@pytest.fixture
def proper_order_rule():
    return ProperOrder()


def test_proper_order_simple(proper_order_rule):
    op = X @ (Y @ (PI @ Z))
    expected = ((X @ Y) @ PI) @ Z
    assert canonicalize_operator(operator=op, rule=proper_order_rule) == expected


def test_proper_order_complicated(proper_order_rule):
    op = X @ (Y @ Z) + 3 * (Z @ (Y @ PI))
    expected = (X @ Y) @ Z + 3 * ((Z @ Y) @ PI)
    assert canonicalize_operator(operator=op, rule=proper_order_rule) == expected


@pytest.fixture
def pauli_algebra_rule():
    return PauliAlgebra()


def test_simple_pauli(pauli_algebra_rule):
    """Simple test"""
    op = X * X + Y * Y + Z * PI
    expected = PI + PI + Z
    assert canonicalize_operator(operator=op, rule=pauli_algebra_rule) == expected


def test_nested_multiplications(pauli_algebra_rule):
    """Nested Multiplication test"""
    op = X * X + Y * Y * Z * PI
    expected = PI + Z
    assert canonicalize_operator(operator=op, rule=pauli_algebra_rule) == expected


@pytest.mark.xfail(strict=True)
def test_nested_multiplications_complicated(pauli_algebra_rule):
    """Complicated Nested Multiplication test fails as we need GatherMathExpr after PauliAlgebra"""
    op = Z * X * X * Y * Y * Z * PI
    expected = PI + Z
    assert (
        canonicalize_operator(operator=op, rule=pauli_algebra_rule) == expected
    )  # check fails


@pytest.fixture
def pauli_gather_rule():
    return GatherPauli()


def test_simple_pauli_gather(pauli_gather_rule):
    """Simple test"""
    op = X @ A @ Y
    expected = X @ Y @ A
    assert canonicalize_operator(operator=op, rule=pauli_gather_rule) == expected


def test_complicated_pauli_gather(pauli_gather_rule):
    """Complicated test"""
    op = X @ A @ Y + (A * A * C) @ Y @ Z + X @ Y @ Z
    expected = X @ Y @ A + Y @ Z @ (A * A * C) + X @ Y @ Z
    assert canonicalize_operator(operator=op, rule=pauli_gather_rule) == expected


@pytest.fixture
def normal_order_rule():
    return NormalOrder()


def test_normal_order_simple(normal_order_rule):
    """Simple test"""
    op = (A * C) @ (A * C)
    expected = (C * A + LI) @ (C * A + LI)
    assert canonicalize_operator(operator=op, rule=normal_order_rule) == expected


def test_normal_order_simple_fail(normal_order_rule):
    """Simple test fail as distribution required"""
    op = 1 * (A * A * C)
    expected = 1 * (C * A * A + LI * A + A * LI)
    assert canonicalize_operator(operator=op, rule=normal_order_rule) != expected


def test_normal_order_inside_pauli(normal_order_rule):
    """Simple test with Pauli"""
    op = X @ (A * C) @ Y
    expected = X @ (C * A + LI) @ Y
    assert canonicalize_operator(operator=op, rule=normal_order_rule) == expected


@pytest.fixture
def prune_identity_rule():
    return PruneIdentity()


def test_prune_identity_simple(prune_identity_rule):
    """Simple Test"""
    op = A * LI * C * LI * LI
    expected = A * C
    assert canonicalize_operator(operator=op, rule=prune_identity_rule) == expected


def test_prune_identity_simple_nested(prune_identity_rule):
    """Simple nested"""
    op = (A * LI * C * A * LI * C) @ LI @ (A * LI)
    expected = (A * C * A * C) @ LI @ (A)
    assert canonicalize_operator(operator=op, rule=prune_identity_rule) == expected


@pytest.fixture
def sorted_order_rule():
    return SortedOrder()


def test_sorted_order_simple(sorted_order_rule):
    """Simple Test"""
    op = X @ Y + X @ Z + PI @ Z
    expected = PI @ Z + X @ Y + X @ Z
    assert canonicalize_operator(operator=op, rule=sorted_order_rule) == expected


def test_sorted_order_terminals(sorted_order_rule):
    """Simple Test with terminals"""
    op = X + Z + Y + PI
    expected = PI + X + Y + Z
    assert canonicalize_operator(operator=op, rule=sorted_order_rule) == expected


def test_sorted_order_ladder(sorted_order_rule):
    """Simple Test with ladder"""
    op = X @ (C * A * A) + X @ (C * A)
    expected = X @ (C * A) + X @ (C * A * A)
    assert canonicalize_operator(operator=op, rule=sorted_order_rule) == expected


@pytest.fixture
def scale_terms_rule():
    return ScaleTerms()


def test_scale_terms_simple(scale_terms_rule):
    """Simple test"""
    op = X @ (Y @ Z) + (Z @ (Y @ PI))
    expected = MathStr(string="1") * (X @ (Y @ Z)) + MathStr(string="1") * (
        Z @ (Y @ PI)
    )
    assert (
        canonicalize_operator(operator=op, rule=scale_terms_rule, walk_method=Pre)
        == expected
    )


def test_scale_terms_single_term(scale_terms_rule):
    """Single term sorted order"""
    op = X @ (Y @ Z)
    expected = MathStr(string="1") * (X @ (Y @ Z))
    assert (
        canonicalize_operator(operator=op, rule=scale_terms_rule, walk_method=Pre)
        == expected
    )


def test_scale_terms_terminals(scale_terms_rule):
    """Terminal term sorted order"""
    op = X + Y + Z + PI
    expected = (
        MathStr(string="1") * X
        + MathStr(string="1") * Y
        + MathStr(string="1") * Z
        + MathStr(string="1") * PI
    )
    assert (
        canonicalize_operator(operator=op, rule=scale_terms_rule, walk_method=Pre)
        == expected
    )
