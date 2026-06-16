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

import numpy as np
import pytest

########################################################################################

from oqd_core.compiler.analog.math.passes import (
    canonicalize_math_expr,
    evaluate_math_expr,
    simplify_math_expr,
)
from oqd_core.frontend.analog.AnalogCircuitAST import parse_analog
from oqd_core.interface.analog import (
    Access,
    Declaration,
    MathAdd,
    MathFunc,
    MathImag,
    MathMul,
    MathNum,
    MathVar,
    MathFunc,
)
from helpers import parse_math

########################################################################################


class TestEvaluate:
    @pytest.mark.parametrize(
        ("source", "target"),
        [
            ("1", 1),
            ("1+2", 3),
            ("1-2", -1),
            ("1*-1", -1),
            ("1*2", 2),
            ("1/2", 0.5),
            ("2^3", 8),
            ("1+2*3-(5/5)^6", 6),
            ("imag(0)", 0),
            ("imag(1j)", 1),
            ("real(1j)", 0),
            ("real(-1)", -1),
            ("cos(0)", 1),
            ("heaviside(-2)", 0),
            ("heaviside(cos(0))", 1),
            ("atan2(0, 1)", 0),
            ("atan2(1, 0)", np.pi / 2),
            ("atan2(1, -1)", 3 * np.pi / 4),
            ("atan2(-1, -1)", -3 * np.pi / 4),
        ],
    )
    def test_evaluate(self, source, target):
        assert evaluate_math_expr(parse_math(source)) == target
    @pytest.mark.parametrize(
        "source, expected",
        [
            ("3+5", 8),
            ("3.02+5.01", 8.03),
            ("3-5", -2),
            ("-3.02+5.01", 1.99),
            ("3*5", 15),
            ("15/2", 7.5),
            ("3^2.01", 9.10),
            ("sin(0.25)", 0.2474),
            ("tan(0.205)", 0.208),
            ("2*3 + 5*(1j)", 6 + 5j),
            ("1+2*3 + 9 - 0.1 + 7*(2+3*5+(10/3))", 158.233),
            ("sin(exp(2))", 0.894),
        ],
    )

    def test_evaluate_approx(self, source, expected):
        assert pytest.approx(
            evaluate_math_expr(parse_math(source)), 0.001
        ) == expected

    def test_access_raises(self):
        expr = MathAdd(expr1=Access(name="s"), expr2=MathNum(value=1))
        with pytest.raises(TypeError, match="Access"):
            evaluate_math_expr(expr)


class TestSimplify:
    @pytest.mark.parametrize(
        ("source", "target"),
        [
            ("1", MathNum(value=1)),
            ("1+2", MathNum(value=3)),
            ("1-2", MathNum(value=-1)),
            ("1*-1", MathNum(value=-1)),
            ("1*2", MathNum(value=2)),
            ("1/2", MathNum(value=0.5)),
            ("2^3", MathNum(value=8)),
            ("1+2*3-(5/5)^6", MathNum(value=6)),
            ("cos(0)", MathFunc(func="cos", expr=MathNum(value=0))),
            ("heaviside(-2)", MathFunc(func="heaviside", expr=MathNum(value=-2))),
            (
                "heaviside(cos(0))",
                MathFunc(
                    func="heaviside",
                    expr=MathFunc(func="cos", expr=MathNum(value=0)),
                ),
            ),
            ("cos(#t)", "cos(#t)"),
            ("heaviside(cos(#t))", "heaviside(cos(#t))"),
            ("1 * #t * 2", "1 * #t * 2"),
            ("1j", MathImag()),
            (
                "1j * #t * 2",
                MathImag() * MathVar(name="#t") * MathNum(value=2),
            ),
        ],
    )
    def test_simplify(self, source, target):
        assert simplify_math_expr(parse_math(source)) == parse_math(target)


class TestCanonicalizeMath:
    @pytest.mark.parametrize(
        ("source", "target"),
        [
            ("1", MathNum(value=1)),
            ("1+2", MathNum(value=3)),
            ("1-2", MathNum(value=-1)),
            ("1*-1", MathNum(value=-1)),
            ("1*2", MathNum(value=2)),
            ("1/2", MathNum(value=0.5)),
            ("2^3", MathNum(value=8)),
            ("1+2*3-(5/5)^6", MathNum(value=6)),
            ("cos(0)", MathFunc(func="cos", expr=MathNum(value=0))),
            ("heaviside(-2)", MathFunc(func="heaviside", expr=MathNum(value=-2))),
            (
                "heaviside(cos(0))",
                MathFunc(
                    func="heaviside",
                    expr=MathFunc(func="cos", expr=MathNum(value=0)),
                ),
            ),
            ("cos(#t)", "cos(#t)"),
            ("heaviside(cos(#t))", "heaviside(cos(#t))"),
            ("1 * #t * 2", "2 * #t"),
            ("1j", MathImag()),
            ("1j * 2", MathImag() * MathNum(value=2)),
            (
                "1j * 2 + 3",
                MathNum(value=3) + MathImag() * MathNum(value=2),
            ),
            (
                "1j * #t * 2",
                MathImag() * MathNum(value=2) * MathVar(name="#t"),
            ),
            ("0^0", MathNum(value=1)),
            ("(cos(0)-cos(0))^sin(0)", MathNum(value=0)),
        ],
    )
    def test_canonicalize(self, source, target):
        assert canonicalize_math_expr(parse_math(source)) == parse_math(target)


class TestMathFunc:
    @pytest.mark.parametrize(
        ("func", "expr"),
        [
            *[(fn, MathNum(value=0)) for fn in [
                "abs", "sin", "cos", "tan", "exp", "log", "sinh", "cosh",
                "tanh", "atan", "acos", "asin", "atanh", "asinh", "acosh",
                "heaviside", "conj",
            ]],
            *[(fn, [MathNum(value=0)]) for fn in [
                "abs", "sin", "cos", "tan", "exp", "log", "sinh", "cosh",
                "tanh", "atan", "acos", "asin", "atanh", "asinh", "acosh",
                "heaviside", "conj",
            ]],
        ],
    )
    def test_unary_function(self, func, expr):
        MathFunc(func=func, expr=expr)
    @pytest.mark.xfail
    @pytest.mark.parametrize(
        ("func", "expr"),
        [
            (fn, [MathNum(value=0), MathNum(value=1)])
            for fn in [
                "abs", "sin", "cos", "tan", "exp", "log", "sinh", "cosh",
                "tanh", "atan", "acos", "asin", "atanh", "asinh", "acosh",
                "heaviside", "conj",
            ]
        ],
    )

    def test_xfail_unary_with_two_args(self, func, expr):
        MathFunc(func=func, expr=expr)
    @pytest.mark.parametrize(
        ("func", "expr"),
        [(fn, [MathNum(value=0), MathNum(value=1)]) for fn in ["atan2"]],
    )

    def test_binary_function(self, func, expr):
        MathFunc(func=func, expr=expr)



class TestParseMath:
    @pytest.mark.parametrize(
        "source",
        [
            "1", "1.0", "1e10", "#t", "cos(0)", "sin(#t)",
            *[
                f"{fn}(w * #t - k * x + phi)"
                for fn in [
                    "abs", "sin", "cos", "tan", "exp", "log", "sinh", "cosh",
                    "tanh", "atan", "acos", "asin", "atanh", "asinh", "acosh",
                    "heaviside", "conj",
                ]
            ],
            "atan2(#imag, #real)",
        ],
    )
    def test_parse(self, source):
        parse_math(source)
