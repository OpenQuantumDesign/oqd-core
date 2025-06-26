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

########################################################################################

import pytest

from oqd_core.compiler.math.passes import (
    canonicalize_math_expr,
    evaluate_math_expr,
    simplify_math_expr,
)
from oqd_core.interface.math import MathImag, MathNum, MathStr, MathVar

########################################################################################


class TestEvaluate:
    @pytest.mark.parametrize(
        ("expr", "target"),
        [
            [MathStr(string="1"), 1],
            [MathStr(string="1+2"), 3],
            [MathStr(string="1-2"), -1],
            [MathStr(string="1*-1"), -1],
            [MathStr(string="1*2"), 2],
            [MathStr(string="1/2"), 0.5],
            [MathStr(string="2**3"), 8],
            [MathStr(string="1+2*3-(5/5)**6"), 6],
            [MathStr(string="cos(0)"), 1],
            [MathStr(string="heaviside(-2)"), 0],
            [MathStr(string="heaviside(cos(0))"), 1],
        ],
    )
    def test_evaluate(self, expr, target):
        assert evaluate_math_expr(expr) == target


class TestSimplify:
    @pytest.mark.parametrize(
        ("expr", "target"),
        [
            [MathStr(string="1"), MathNum(value="1")],
            [MathStr(string="1+2"), MathNum(value="3")],
            [MathStr(string="1-2"), MathNum(value="-1")],
            [MathStr(string="1*-1"), MathNum(value="-1")],
            [MathStr(string="1*2"), MathNum(value="2")],
            [MathStr(string="1/2"), MathNum(value="0.5")],
            [MathStr(string="2**3"), MathNum(value="8")],
            [MathStr(string="1+2*3-(5/5)**6"), MathNum(value="6")],
            [MathStr(string="cos(0)"), MathNum(value="1")],
            [MathStr(string="heaviside(-2)"), MathNum(value="0")],
            [MathStr(string="heaviside(cos(0))"), MathNum(value="1")],
            [MathStr(string="cos(t)"), MathStr(string="cos(t)")],
            [MathStr(string="heaviside(cos(t))"), MathStr(string="heaviside(cos(t))")],
            [MathStr(string="1 * t * 2"), MathStr(string="1 * t * 2")],
            [MathStr(string="1*t*t*t*t*t*2"), MathStr(string="1*t*t*t*t*t*2")],
            [MathStr(string="1j"), MathImag()],
            [
                MathStr(string="1j * t * 2"),
                MathImag() * MathVar(name="t") * MathNum(value=2),
            ],
        ],
    )
    def test_simplify(self, expr, target):
        assert simplify_math_expr(expr) == target


class TestCanonicalizeMath:
    @pytest.mark.parametrize(
        ("expr", "target"),
        [
            [MathStr(string="1"), MathNum(value="1")],
            [MathStr(string="1+2"), MathNum(value="3")],
            [MathStr(string="1-2"), MathNum(value="-1")],
            [MathStr(string="1*-1"), MathNum(value="-1")],
            [MathStr(string="1*2"), MathNum(value="2")],
            [MathStr(string="1/2"), MathNum(value="0.5")],
            [MathStr(string="2**3"), MathNum(value="8")],
            [MathStr(string="1+2*3-(5/5)**6"), MathNum(value="6")],
            [MathStr(string="cos(0)"), MathNum(value="1")],
            [MathStr(string="heaviside(-2)"), MathNum(value="0")],
            [MathStr(string="heaviside(cos(0))"), MathNum(value="1")],
            [MathStr(string="cos(t)"), MathStr(string="cos(t)")],
            [MathStr(string="heaviside(cos(t))"), MathStr(string="heaviside(cos(t))")],
            [MathStr(string="1 * t * 2"), MathStr(string="2 * t")],
            [MathStr(string="1*t*t*t*t*t*2"), MathStr(string="2*t*t*t*t*t")],
            [MathStr(string="1*t*2*3"), MathStr(string="6*t")],
            [MathStr(string="1j"), MathImag()],
            [
                MathStr(string="1j * 2"),
                MathImag() * MathNum(value=2),
            ],
            [
                MathStr(string="1j * 2 + 3"),
                MathNum(value=3) + MathImag() * MathNum(value=2),
            ],
            [
                MathStr(string="1j * t * 2"),
                MathImag() * MathNum(value=2) * MathVar(name="t"),
            ],
            [MathStr(string="0**0"), MathNum(value=1)],
            [MathStr(string="(cos(0)-cos(0))**sin(0)"), MathNum(value=1)],
        ],
    )
    def test_canonicalize(self, expr, target):
        assert canonicalize_math_expr(expr) == target
