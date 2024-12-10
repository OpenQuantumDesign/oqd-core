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

from oqd_compiler_infrastructure import In
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
from oqd_core.compiler.analog.analysis import TermIndex

X, Y, Z, PI, A, C, LI = (
    PauliX(),
    PauliY(),
    PauliZ(),
    PauliI(),
    Annihilation(),
    Creation(),
    Identity(),
)


def compute_term_index(operator: Operator):
    analysis = In(TermIndex())
    analysis(model=operator)
    return analysis.children[0].term_idx


def test_simple_single_term_pauli():
    """Simple test single term pauli"""
    op = X + Y + Z + PI
    expected = [[1], [2], [3], [0]]
    assert compute_term_index(operator=op) == expected


def test_simple_single_term_ladder():
    """Simple test single term ladder"""
    op = A + C + LI
    expected = [[(1, 0)], [(1, 1)], [(0, 0)]]
    assert compute_term_index(operator=op) == expected


def test_kron_add():
    """Simple kron add"""
    op = X @ Y + Z @ X + X @ X + Y @ Z
    expected = [[1, 2], [3, 1], [1, 1], [2, 3]]
    assert compute_term_index(operator=op) == expected


def test_kron_single():
    """Simple kron single term"""
    op = X @ Y @ Z @ PI
    expected = [[1, 2, 3, 0]]
    assert compute_term_index(operator=op) == expected


def test_kron_single_ladder():
    """Simple kron single term with ladder"""
    op = C @ (A * LI)
    expected = [[(1, 1), (1, 0)]]
    assert compute_term_index(operator=op) == expected


def test_kron_single_complicated():
    """Complicated kron single term"""
    op = X @ Y @ X @ X @ (A * C * A) @ PI @ C
    expected = [[1, 2, 1, 1, (3, 1), 0, (1, 1)]]
    assert compute_term_index(operator=op) == expected


def test_kron_add_complicated():
    """Complicated kron add"""
    op = X @ Y @ (C * LI * A * C) + X @ X + (A * C * A) @ PI @ C + Y + Z
    expected = [[1, 2, (3, 2)], [1, 1], [(3, 1), 0, (1, 1)], [2], [3]]
    assert compute_term_index(operator=op) == expected


def test_kron_add_complicated_with_scalar():
    """Complicated kron add with scalar mul"""
    op = C @ (C * LI * A * C) + 3 * (X @ X) + A * C * A + 3 * Y + 2 * Z + C + LI
    expected = [[(1, 1), (3, 2)], [1, 1], [(3, 1)], [2], [3], [(1, 1)], [(0, 0)]]
    assert compute_term_index(operator=op) == expected
