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
from typing import Union

from oqd_compiler_infrastructure import Post, WalkBase, ConversionRule, RewriteRule

########################################################################################

from oqd_core.compiler.math.rules import EvaluateMathExpr
from oqd_core.interface.math import MathStr

########################################################################################


def evaluate_math_expr(
    math_str: str,
    rule: Union[ConversionRule, RewriteRule] = EvaluateMathExpr(),
    walk_method: WalkBase = Post,
):
    return walk_method(rule)(MathStr(string=math_str))


@pytest.mark.parametrize(
    "math_str, expected",
    [
        ("3+5", 8),
        ("3.02+5.01", 8.03),
        ("3-5", -2),
        ("-3.02+5.01", 1.99),
        ("3*5", 15),
        ("15/2", 7.5),
        ("3**2.01", 9.10),
        ("sin(0.25)", 0.2474),
        ("tan(0.205)", 0.208),
        ("2*3 + 5*(1j)", 6 + 5j),
        ("1+2*3 + 9 - 0.1 + 7*(2+3*5+(10/3))", 158.233),
        ("sin(exp(2))", 0.894),
    ],
)
def test_math_expressions(math_str, expected):
    assert pytest.approx(evaluate_math_expr(math_str), 0.001) == expected
