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

from oqd_core.interface.math import MathFunc, MathStr

########################################################################################


class TestUnaryFunctions:
    @pytest.mark.parametrize(
        ("func", "expr"),
        [
            *[
                (fn, 0)
                for fn in [
                    "abs",
                    "sin",
                    "cos",
                    "tan",
                    "exp",
                    "log",
                    "sinh",
                    "cosh",
                    "tanh",
                    "atan",
                    "acos",
                    "asin",
                    "atanh",
                    "asinh",
                    "acosh",
                    "heaviside",
                    "conj",
                ]
            ],
            *[
                (fn, [0])
                for fn in [
                    "abs",
                    "sin",
                    "cos",
                    "tan",
                    "exp",
                    "log",
                    "sinh",
                    "cosh",
                    "tanh",
                    "atan",
                    "acos",
                    "asin",
                    "atanh",
                    "asinh",
                    "acosh",
                    "heaviside",
                    "conj",
                ]
            ],
        ],
    )
    def test_binary_function(self, func, expr):
        MathFunc(func=func, expr=expr)

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        ("func", "expr"),
        [
            (fn, [0, 1])
            for fn in [
                "abs",
                "sin",
                "cos",
                "tan",
                "exp",
                "log",
                "sinh",
                "cosh",
                "tanh",
                "atan",
                "acos",
                "asin",
                "atanh",
                "asinh",
                "acosh",
                "heaviside",
                "conj",
            ]
        ],
    )
    def test_xfail_unary_functions(self, func, expr):
        MathFunc(func=func, expr=expr)


class TestBinaryFunctions:
    @pytest.mark.parametrize(
        ("func", "expr"),
        [(fn, [0, 1]) for fn in ["atan2"]],
    )
    def test_binary_function(self, func, expr):
        MathFunc(func=func, expr=expr)

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        ("func", "expr"),
        [
            *[(fn, [0]) for fn in ["atan2"]],
            *[(fn, [0, 1, 2]) for fn in ["atan2"]],
        ],
    )
    def test_xfail_binary_functions(self, func, expr):
        MathFunc(func=func, expr=expr)


class TestMathStr:
    @pytest.mark.parametrize(
        ("string"),
        [
            "1",
            "1.0",
            "1e10",
            "t",
            "cos(0)",
            "sin(t)",
            *[
                f"{fn}(w * t - k * x + phi)"
                for fn in [
                    "abs",
                    "sin",
                    "cos",
                    "tan",
                    "exp",
                    "log",
                    "sinh",
                    "cosh",
                    "tanh",
                    "atan",
                    "acos",
                    "asin",
                    "atanh",
                    "asinh",
                    "acosh",
                    "heaviside",
                    "conj",
                ]
            ],
            *[f"{fn}(imag, real)" for fn in ["atan2"]],
        ],
    )
    def test_binary_function(self, string):
        MathStr(string=string)
