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

from oqd_compiler_infrastructure import Chain, FixedPoint, Post

from oqd_core.compiler.math.rules import (
    DistributeMathExpr,
    EvaluateMathExpr,
    PartitionMathExpr,
    PrintMathExpr,
    ProperOrderMathExpr,
    PruneMathExpr,
    SimplifyMathExpr,
)

########################################################################################

__all__ = [
    "evaluate_math_expr",
    "simplify_math_expr",
    "print_math_expr",
    "canonicalize_math_expr",
]

########################################################################################

evaluate_math_expr = Post(EvaluateMathExpr())
"""
Pass for evaluating math expression
"""

simplify_math_expr = Post(SimplifyMathExpr())
"""
Pass for simplifying math expression
"""

print_math_expr = Post(PrintMathExpr())
"""
Pass for printing math expression
"""

canonicalize_math_expr = Chain(
    FixedPoint(
        Post(
            Chain(
                PruneMathExpr(),
                SimplifyMathExpr(),
                DistributeMathExpr(),
                ProperOrderMathExpr(),
            )
        )
    ),
    FixedPoint(Post(PartitionMathExpr())),
    simplify_math_expr,
)

"""
Pass for canonicalizing math expression
"""
