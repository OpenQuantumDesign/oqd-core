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

from typing import Union

from oqd_compiler_infrastructure import RewriteRule
from oqd_core.compiler.analog.operator.term_index import term_index
from oqd_core.compiler.analog.operator.dim import (
    coeff_and_op,
    is_scalar_mul,
    scalar_mul,
)
from oqd_core.interface.analog.expr import (
    Annihilation, Creation, Identity, Ladder,
    MathAdd, MathImag, MathMul, MathNum,
    OperatorAdd, OperatorKron, OperatorMul, OperatorSub,
    OperatorTerminal, Pauli, PauliI, PauliX, PauliY, PauliZ,
)

########################################################################################

__all__ = [
    "OperatorDistribute",
    "GatherMathExpr",
    "GatherPauli",
    "PruneIdentity",
    "PauliAlgebra",
    "NormalOrder",
    "ProperOrder",
    "ScaleTerms",
    "SortedOrder",
    "PruneZeros",
]

########################################################################################


class OperatorDistribute(RewriteRule):
    """
    RewriteRule which distributes operators of hamiltonians

    Args:
        model (VisitableBaseModel):

    Returns:
        model (VisitableBaseModel):

    Assumptions:
        [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr] (sometimes)

    Example:
        X@(Y+Z) => X@Y + X@Z
    """

    def map_OperatorMul(self, model: OperatorMul):
        if is_scalar_mul(model):
            return None
        
        if isinstance(model.op1, (OperatorAdd, OperatorSub)):
            return model.op1.__class__(
                op1=OperatorMul(op1=model.op1.op1, op2=model.op2),
                op2=OperatorMul(op1=model.op1.op2, op2=model.op2),
            )
        if isinstance(model.op2, (OperatorAdd, OperatorSub)):
            return model.op2.__class__(
                op1=OperatorMul(op1=model.op1, op2=model.op2.op1),
                op2=OperatorMul(op1=model.op1, op2=model.op2.op2),
            )
        if isinstance(model.op1, (OperatorKron)) and isinstance(
            model.op2, (OperatorKron)
        ):
            return OperatorKron(
                op1=OperatorMul(op1=model.op1.op1, op2=model.op2.op1),
                op2=OperatorMul(op1=model.op1.op2, op2=model.op2.op2),
            )
        return None

    def map_OperatorKron(self, model: OperatorKron):
        if isinstance(model.op1, (OperatorAdd, OperatorSub)):
            return model.op1.__class__(
                op1=OperatorKron(op1=model.op1.op1, op2=model.op2),
                op2=OperatorKron(op1=model.op1.op2, op2=model.op2),
            )
        if isinstance(model.op2, (OperatorAdd, OperatorSub)):
            return model.op2.__class__(
                op1=OperatorKron(op1=model.op1, op2=model.op2.op1),
                op2=OperatorKron(op1=model.op1, op2=model.op2.op2),
            )
        return None

    def map_OperatorSub(self, model: OperatorSub):
        return OperatorAdd(
            op1=model.op1,
            op2=scalar_mul(MathNum(value=-1), model.op2),
        )


class GatherMathExpr(RewriteRule):
    """
    Gathers the math expressions of  [`Operator`][oqd_core.interface.analog.expr.Operator] so that we have math_expr * ( [`Operator`][oqd_core.interface.analog.expr.Operator] without scalar multiplication)

    Args:
        model (VisitableBaseModel):

    Returns:
        model (VisitableBaseModel):

    Assumptions:
        [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute] (sometimes)

    Example:
        (1 * X) @ (2 * Y) => (1 * 2) => (1 * 2) * (X @ Y)
    """

    def map_OperatorMul(self, model: OperatorMul):
        if is_scalar_mul(model):
            c1, inner = coeff_and_op(model)
            if is_scalar_mul(inner):
                c2, op = coeff_and_op(inner)
                return scalar_mul(MathMul(expr1=c1, expr2=c2), op)
        return self._mulkron(model)

    def map_OperatorKron(self, model: OperatorKron):
        return self._mulkron(model)

    def _mulkron(self, model):
        c1, t1 = coeff_and_op(model.op1)
        c2, t2 = coeff_and_op(model.op2)
        if c1 is None or c2 is None:
            return None
        if not (is_scalar_mul(model.op1) or is_scalar_mul(model.op2)):
            return None
        coeff = MathMul(expr1=c1, expr2=c2)
        return scalar_mul(coeff, model.__class__(op1=t1, op2=t2))


class GatherPauli(RewriteRule):
    """
    Gathers ladders and paulis so that we have paulis and then ladders

    Args:
        model (VisitableBaseModel):

    Returns:
        model (VisitableBaseModel):

    Assumptions:
        [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr],
        [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute],
        [`ProperOrder`][oqd_core.compiler.analog.rewrite.canonicalize.ProperOrder]
        [`Operator`][oqd_core.interface.analog.expr.Operator]

    Example:
        X@A@Y => X@Y@A
    """

    def map_OperatorKron(self, model: OperatorKron):
        _, op1 = coeff_and_op(model.op1)
        if isinstance(model.op2, Pauli):
            if isinstance(op1, Ladder):
                return OperatorKron(
                    op1=model.op2,
                    op2=op1,
                )
            if isinstance(op1, OperatorMul) and isinstance(op1.op2, Ladder):
                return OperatorKron(
                    op1=model.op2,
                    op2=op1,
                )
            if isinstance(op1, OperatorKron) and isinstance(
                op1.op2, Union[Ladder, OperatorMul]
            ):
                return OperatorKron(
                    op1=OperatorKron(op1=op1.op1, op2=model.op2),
                    op2=model.op1.op2,
                )
        return None


class PruneIdentity(RewriteRule):
    """
    Removes unnecessary ladder Identities from operators

    Args:
        model (VisitableBaseModel):

    Returns:
        model (VisitableBaseModel):

    Assumptions:
        [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr],
        [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute],
        [`ProperOrder`][oqd_core.compiler.analog.rewrite.canonicalize.ProperOrder],
        [`GatherPauli`][oqd_core.compiler.analog.rewrite.canonicalize.GatherPauli],
        [`NormalOrder`][oqd_core.compiler.analog.rewrite.canonicalize.NormalOrder]

    Example:
        A*J => A
    """

    def map_OperatorMul(self, model: OperatorMul):
        if is_scalar_mul(model):
            return None
        if isinstance(model.op1, (Identity)):
            return model.op2
        if isinstance(model.op2, (Identity)):
            return model.op1
        return None


class PauliAlgebra(RewriteRule):
    """
    RewriteRule for Pauli algebra operations

    Args:
        model (VisitableBaseModel):

    Returns:
        model (VisitableBaseModel):

    Assumptions:
        [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr],
        [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute],
        [`ProperOrder`][oqd_core.compiler.analog.rewrite.canonicalize.ProperOrder]

    Example:
        X*Y => iZ
    """

    def map_OperatorMul(self, model: OperatorMul):
        if is_scalar_mul(model):
            return None
        if isinstance(model.op1, Pauli) and isinstance(model.op2, Pauli):
            if isinstance(model.op1, PauliI):
                return model.op2
            if isinstance(model.op2, PauliI):
                return model.op1
            if model.op1 == model.op2:
                return PauliI()
            if isinstance(model.op1, PauliX) and isinstance(model.op2, PauliY):
                return scalar_mul(MathImag(), PauliZ())
            if isinstance(model.op1, PauliY) and isinstance(model.op2, PauliZ):
                return scalar_mul(MathImag(), PauliX())
            if isinstance(model.op1, PauliZ) and isinstance(model.op2, PauliX):
                return scalar_mul(MathImag(), PauliY())
            return scalar_mul(
                MathNum(value=-1),
                OperatorMul(op1=model.op2, op2=model.op1),
            )
        return None


class NormalOrder(RewriteRule):
    """
    Arranges Ladder oeprators in normal order form

    Args:
        model (VisitableBaseModel):

    Returns:
        model (VisitableBaseModel):

    Assumptions:
        [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr],
        [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute],
        [`ProperOrder`][oqd_core.compiler.analog.rewrite.canonicalize.ProperOrder],
        [`GatherPauli`][oqd_core.compiler.analog.rewrite.canonicalize.GatherPauli]

    Example:
        A*C => C*A + J
    """

    def map_OperatorMul(self, model: OperatorMul):
        if is_scalar_mul(model):
            return None
        if isinstance(model.op2, Creation):
            if isinstance(model.op1, Annihilation):
                return OperatorAdd(
                    op1=OperatorMul(op1=model.op2, op2=model.op1), op2=Identity()
                )
            if isinstance(model.op1, Identity):
                return OperatorMul(op1=model.op2, op2=model.op1)
            if isinstance(model.op1, OperatorMul) and isinstance(
                model.op1.op2, (Annihilation, Identity)
            ):
                return OperatorMul(
                    op1=model.op1.op1,
                    op2=OperatorMul(op1=model.op1.op2, op2=model.op2),
                )
        return None


class ProperOrder(RewriteRule):
    """
    Converts expressions to proper order bracketing. Please see example for clarification.

    Args:
        model (VisitableBaseModel):

    Returns:
        model (VisitableBaseModel):

    Assumptions:
        [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr],
        [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute]

    Example:
        X @ (Y @ Z) =>  (X @ Y) @ Z
    """

    def map_OperatorAdd(self, model: OperatorAdd):
        return self._addmullkron(model=model)

    def map_OperatorMul(self, model: OperatorMul):
        if is_scalar_mul(model):
            return None
        return self._addmullkron(model=model)

    def map_OperatorKron(self, model: OperatorKron):
        return self._addmullkron(model=model)

    def _addmullkron(self, model: Union[OperatorAdd, OperatorMul, OperatorKron]):
        if isinstance(model.op2, model.__class__):
            return model.__class__(
                op1=model.__class__(op1=model.op1, op2=model.op2.op1),
                op2=model.op2.op2,
            )
        return model.__class__(op1=model.op1, op2=model.op2)


class ScaleTerms(RewriteRule):
    """
    Scales operators to ensure consistency

    Args:
        model (VisitableBaseModel):

    Returns:
        model (VisitableBaseModel):

    Assumptions:
        [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr],
        [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute],
        [`ProperOrder`][oqd_core.compiler.analog.rewrite.canonicalize.ProperOrder],
        [`GatherPauli`][oqd_core.compiler.analog.rewrite.canonicalize.GatherPauli],
        [`NormalOrder`][oqd_core.compiler.analog.rewrite.canonicalize.NormalOrder],
        [`PruneIdentity`][oqd_core.compiler.analog.rewrite.canonicalize.PruneIdentity]

    Note:
        - Requires [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr] right after application of [`ScaleTerms`][oqd_core.compiler.analog.rewrite.canonicalize.ScaleTerms]  for Post walk
        - [`SortedOrder`][oqd_core.compiler.analog.rewrite.canonicalize.SortedOrder] and  [`ScaleTerms`][oqd_core.compiler.analog.rewrite.canonicalize.ScaleTerms] can be run in either order

    Example:
        X + Y + 2*Z => 1*X + 1*Y + 2*Z
        X@Y => 1*(X@Y)
    """

    def __init__(self):
        super().__init__()
        self.op_add_root = False

    def map_Evolve(self, model):
        self.op_add_root = False

    def map_Expectation(self, model):
        self.op_add_root = False

    def map_OperatorAdd(self, model: OperatorAdd):
        self.op_add_root = True
        op1 = model.op1 if (is_scalar_mul(model.op1) or isinstance(model.op1, OperatorAdd)) else scalar_mul(MathNum(value=1), model.op1)
        op2 = model.op2 if (is_scalar_mul(model.op2) or isinstance(model.op2, OperatorAdd)) else scalar_mul(MathNum(value=1), model.op2)
        return OperatorAdd(op1=op1, op2=op2)
    
    def map_OperatorTerminal(self, model):
        if not self.op_add_root:
            self.op_add_root = True
            return scalar_mul(MathNum(value=1), model)
        
    def map_OperatorKron(self, model):
        if not self.op_add_root:
            self.op_add_root = True
            return scalar_mul(MathNum(value=1), model)
        
    def map_OperatorMul(self, model):
        if is_scalar_mul(model):
            return None
        if not self.op_add_root:
            self.op_add_root = True
            return scalar_mul(MathNum(value=1), model)


class SortedOrder(RewriteRule):
    """
    Sorts operators based on TermIndex and collects duplicate terms.
    Please see example for clarification

    Args:
        model (VisitableBaseModel):

    Returns:
        model (VisitableBaseModel)

    Assumptions:
        [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr],
        [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute],
        [`ProperOrder`][oqd_core.compiler.analog.rewrite.canonicalize.ProperOrder],
        [`GatherPauli`][oqd_core.compiler.analog.rewrite.canonicalize.GatherPauli],
        [`NormalOrder`][oqd_core.compiler.analog.rewrite.canonicalize.NormalOrder],
        [`PruneIdentity`][oqd_core.compiler.analog.rewrite.canonicalize.PruneIdentity]

    Note:
        - [`SortedOrder`][oqd_core.compiler.analog.rewrite.canonicalize.SortedOrder] and  [`ScaleTerms`][oqd_core.compiler.analog.rewrite.canonicalize.ScaleTerms] can be run in either order

    Example:
        (X@Y) + (X@I) => (X@I) + (X@Y)
        X + I + Z + Y => I + X + Y + Z
    """

    def map_OperatorAdd(self, model: OperatorAdd):
        if isinstance(model.op1, OperatorAdd):
            term1 = term_index(model.op1.op2)
            term2 = term_index(model.op2)

            if term1 == term2:
                expr1, _ = coeff_and_op(model.op1.op2)
                expr2, op_part = coeff_and_op(model.op2)
                return OperatorAdd(
                    op1=model.op1.op1,
                    op2=scalar_mul(
                        MathAdd(expr1=expr1, expr2=expr2),
                        op_part,
                    ),
                )

            elif term1 > term2:
                return OperatorAdd(
                    op1=OperatorAdd(op1=model.op1.op1, op2=model.op2),
                    op2=model.op1.op2,
                )

            elif term1 < term2:
                return OperatorAdd(op1=model.op1, op2=model.op2)

        else:
            term1 = term_index(model.op1)
            term2 = term_index(model.op2)

            if term1 == term2:
                expr1, _ = coeff_and_op(model.op1)
                expr2, op_part = coeff_and_op(model.op2)
                return scalar_mul(MathAdd(expr1=expr1, expr2=expr2), op_part)

            elif term1 > term2:
                return OperatorAdd(
                    op1=model.op2,
                    op2=model.op1,
                )

            elif term1 < term2:
                return OperatorAdd(op1=model.op1, op2=model.op2)


class PruneZeros(RewriteRule):
    """
    Removes operators multiplied by zero

    Args:
        model (VisitableBaseModel):

    Returns:
        model (VisitableBaseModel):

    Assumptions:
        [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr],
        [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute],
        [`ProperOrder`][oqd_core.compiler.analog.rewrite.canonicalize.ProperOrder],
        [`GatherPauli`][oqd_core.compiler.analog.rewrite.canonicalize.GatherPauli],
        [`NormalOrder`][oqd_core.compiler.analog.rewrite.canonicalize.NormalOrder]
        [`PruneIdentity`][oqd_core.compiler.analog.rewrite.canonicalize.PruneIdentity]

    """

    def map_OperatorAdd(self, model):
        c1, _ = coeff_and_op(model.op1)
        c2, _ = coeff_and_op(model.op2)
        if c1 == MathNum(value=0):
            return model.op2
        if c2 == MathNum(value=0):
            return model.op1
