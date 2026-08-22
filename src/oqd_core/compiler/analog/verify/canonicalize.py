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

from oqd_core.compiler.analog.error import AnalogCompilerError
from oqd_core.compiler.analog.operator.dim import is_scalar_mul, coeff_and_op
from oqd_core.compiler.analog.operator.term_index import term_index

########################################################################################
from oqd_core.interface.analog.expr import (
    Annihilation,
    Creation,
    Identity,
    Ladder,
    OperatorAdd,
    OperatorKron,
    OperatorMul,
    OperatorSub,
    OperatorTerminal,
    Pauli,
    Access,
)

########################################################################################

__all__ = [
    "CanVerPauliAlgebra",
    "CanVerGatherMathExpr",
    "CanVerOperatorDistribute",
    "CanVerProperOrder",
    "CanVerPruneIdentity",
    "CanVerGatherPauli",
    "CanVerNormalOrder",
    "CanVerSortedOrder",
    "CanVerScaleTerm",
]

########################################################################################


class CanVerPauliAlgebra(RewriteRule):
    """
    Checks whether there is any incomplete Pauli Algebra computation

    Args:
        model (VisitableBaseModel): The rule only verifies [`Operator`][oqd_core.interface.analog.expr.OperatorExpr] in Analog level

    Returns:
        model (VisitableBaseMode): unchanged

    Assumptions:
        [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr],
        [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute],
        [`ProperOrder`][oqd_core.compiler.analog.rewrite.canonicalize.ProperOrder]

    Example:
        - X@(Y*Z) => fail
        - X@Y => pass
    """

    def map_OperatorMul(self, model: OperatorMul):
        if is_scalar_mul(model):
            return
        if isinstance(model.op1, Pauli) and isinstance(model.op2, Pauli):
            raise AnalogCompilerError("Incomplete Pauli Algebra")
        elif isinstance(model.op1, Pauli) and isinstance(model.op2, Ladder):
            raise AnalogCompilerError("Incorrect Ladder and Pauli multiplication")
        elif isinstance(model.op1, Ladder) and isinstance(model.op2, Pauli):
            raise AnalogCompilerError("Incorrect Ladder and Pauli multiplication")
        pass


class CanVerGatherMathExpr(RewriteRule):
    """
    Checks whether all MathExpr have been gathered (i.e. basically checks whether
    there is any scalar multiplication within a term)

    Args:
        model (VisitableBaseModel): The rule only verifies [`Operator`][oqd_core.interface.analog.expr.OperatorExpr] in Analog level

    Returns:
        model (VisitableBaseMode): unchanged

    Assumptions:
        OperatorDistribute
        [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute]

    Example:
        - X@(1*Z) => fail
        - 1*(X@Z) => pass
    """

    def map_OperatorMul(self, model: OperatorMul):
        if is_scalar_mul(model):
            _, inner = coeff_and_op(model)
            if is_scalar_mul(inner):
                raise AnalogCompilerError(
                    "Incomplete scalar multiplications after GatherMathExpression"
                )
            return
        self._mulkron(model)

    def map_OperatorKron(self, model: OperatorKron):
        self._mulkron(model)

    def _mulkron(self, model: Union[OperatorMul, OperatorKron]):
        if is_scalar_mul(model.op1) or is_scalar_mul(model.op2):
            raise AnalogCompilerError("Incomplete Gather Math Expression")


class CanVerOperatorDistribute(RewriteRule):
    """
    Checks for incomplete distribution of Operators

    Args:
        model (VisitableBaseModel): The rule only verifies [`Operator`][oqd_core.interface.analog.expr.OperatorExpr] in Analog level

    Returns:
        model (VisitableBaseMode): unchanged

    Assumptions:
        None

    Example:
        - X@(Y+Z) => fail
        - X@Y + X@Z => pass
    """

    def __init__(self):
        super().__init__()
        self.allowed_ops = (
            OperatorTerminal,
            Ladder,
            OperatorMul,
            OperatorKron,
            Access,
        )

    def map_OperatorMul(self, model):
        return self._OperatorMulKron(model)

    def map_OperatorKron(self, model):
        return self._OperatorMulKron(model)

    def _OperatorMulKron(self, model: Union[OperatorMul, OperatorKron]):
        if (
            isinstance(model, OperatorMul)
            and not is_scalar_mul(model)
            and isinstance(model.op1, OperatorKron)
            and isinstance(model.op2, OperatorKron)
        ):
            raise AnalogCompilerError(
                "Incomplete Operator Distribution (multiplication of OperatorKron present)"
            )
        if is_scalar_mul(model):
            _, inner = coeff_and_op(model)
            if not isinstance(inner, self.allowed_ops):
                raise AnalogCompilerError(
                    "Scalar multiplication of operators not simplified fully"
                )
            return
        if not (
            isinstance(model.op1, self.allowed_ops)
            and isinstance(model.op2, self.allowed_ops)
        ):
            raise AnalogCompilerError("Incomplete Operator Distribution")

        pass

    def map_OperatorSub(self, model: OperatorSub):
        if isinstance(model, OperatorSub):
            raise AnalogCompilerError("Subtraction of terms present")
        pass


class CanVerProperOrder(RewriteRule):
    """
    Checks whether all Operators are ProperOrdered according to how they are bracketed
    Please see example for clarification

    Args:
        model (VisitableBaseModel): The rule only verifies [`Operator`][oqd_core.interface.analog.expr.OperatorExpr] in Analog level

    Returns:
        model (VisitableBaseMode): unchanged

    Assumptions:
        None

    Example:
        - X@(Y@Z) => fail
        - (X@Y)@Z => pass
    """

    def map_OperatorAdd(self, model: OperatorAdd):
        self._OperatorAddMulKron(model)
        pass

    def map_OperatorMul(self, model: OperatorMul):
        if is_scalar_mul(model):
            _, inner = coeff_and_op(model)
            if isinstance(inner, OperatorMul):
                raise AnalogCompilerError(
                    "Incorrect Proper Ordering (for scalar multiplication)"
                )
            return
        self._OperatorAddMulKron(model)

    def map_OperatorKron(self, model: OperatorKron):
        self._OperatorAddMulKron(model)
        pass

    def _OperatorAddMulKron(self, model: Union[OperatorAdd, OperatorMul, OperatorKron]):
        if isinstance(model.op2, model.__class__):
            raise AnalogCompilerError("Incorrect Proper Ordering")
        pass


class CanVerPruneIdentity(RewriteRule):
    """
    Checks if there is any ladder Identity present in ladder multiplication

    Args:
        model (VisitableBaseModel): The rule only verifies [`Operator`][oqd_core.interface.analog.expr.OperatorExpr] in Analog level

    Returns:
        model (VisitableBaseMode): unchanged

    Assumptions:
        OperatorDistribute
        [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute]

    Example:
        - A*J*C => fail
        - A*C => pass
    """

    def map_OperatorMul(self, model: OperatorMul):
        if is_scalar_mul(model):
            return
        if isinstance(model.op1, Identity) or isinstance(model.op2, Identity):
            raise AnalogCompilerError("Prune Identity is not complete")
        pass


class CanVerGatherPauli(RewriteRule):
    """
    Checks whether pauli and ladder have been separated.

    Args:
        model (VisitableBaseModel): The rule only verifies [`Operator`][oqd_core.interface.analog.expr.OperatorExpr] in Analog level

    Returns:
        model (VisitableBaseMode): unchanged

    Assumptions:
        [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr],
        [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute],
        [`ProperOrder`][oqd_core.compiler.analog.rewrite.canonicalize.ProperOrder],
        [`PauliAlgebra`][oqd_core.compiler.analog.rewrite.canonicalize.PauliAlgebra]

    Example:
        - X@A@Y => fail
        - X@Y@A => pass
    """

    def map_OperatorKron(self, model: OperatorKron):
        _, op1 = coeff_and_op(model.op1)
        if isinstance(model.op2, Pauli):
            if isinstance(op1, (Ladder, OperatorMul)):
                raise AnalogCompilerError("Incorrect GatherPauli")
            if isinstance(op1, OperatorKron):
                if isinstance(op1.op2, (Ladder, OperatorMul)):
                    raise AnalogCompilerError("Incorrect GatherPauli")
        pass


class CanVerNormalOrder(RewriteRule):
    """
    Checks whether the ladder operations are in normal order

    Args:
        model (VisitableBaseModel): The rule only verifies [`Operator`][oqd_core.interface.analog.expr.OperatorExpr] in Analog level

    Returns:
        model (VisitableBaseMode): unchanged

    Assumptions:
        OperatorDistribute, GatherMathExpr, ProperOrder, PauliAlgebra, PruneIdentity
        [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr],
        [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute],
        [`ProperOrder`][oqd_core.compiler.analog.rewrite.canonicalize.ProperOrder],
        [`PauliAlgebra`][oqd_core.compiler.analog.rewrite.canonicalize.PauliAlgebra],
        [`PruneIdentity`][oqd_core.compiler.analog.rewrite.canonicalize.PruneIdentity]


    Example:
        - A*C => fail
        - C*A => pass
    """

    def map_OperatorMul(self, model: OperatorMul):
        if is_scalar_mul(model):
            return
        if isinstance(model.op2, Creation):
            if isinstance(model.op1, Annihilation):
                raise AnalogCompilerError("Incorrect NormalOrder")
            if isinstance(model.op1, OperatorMul):
                if isinstance(model.op1.op2, Annihilation):
                    raise AnalogCompilerError("Incorrect NormalOrder")
        pass


class CanVerSortedOrder(RewriteRule):
    """
    Checks whether operators are in sorted order according to TermIndex.
    Please see example for further clarification

    Args:
        model (VisitableBaseModel): The rule only verifies [`Operator`][oqd_core.interface.analog.expr.OperatorExpr] in Analog level

    Returns:
        model (VisitableBaseMode): unchanged

    Assumptions:
        [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr],
        [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute],
        [`ProperOrder`][oqd_core.compiler.analog.rewrite.canonicalize.ProperOrder],
        [`GatherPauli`][oqd_core.compiler.analog.rewrite.canonicalize.GatherPauli],
        [`NormalOrder`][oqd_core.compiler.analog.rewrite.canonicalize.NormalOrder],
        [`PruneIdentity`][oqd_core.compiler.analog.rewrite.canonicalize.PruneIdentity]

    Example:
        - X + I => fail
        - I + X => pass
    """

    def map_OperatorAdd(self, model: OperatorAdd):
        term2 = term_index(model.op2)
        if isinstance(model.op1, OperatorAdd):
            term1 = term_index(model.op1.op2)
        else:
            term1 = term_index(model.op1)
        if term1 > term2:
            raise AnalogCompilerError("Terms are not in sorted order")
        elif term1 == term2:
            raise AnalogCompilerError("Duplicate terms present")
        pass


class CanVerScaleTerm(RewriteRule):
    """
    Checks whether all terms have a scalar multiplication.

    Args:
        model (VisitableBaseModel): The rule only verifies [`Operator`][oqd_core.interface.analog.expr.OperatorExpr] in Analog level

    Returns:
        model (VisitableBaseMode): unchanged

    Assumptions:
        [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr],
        [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute],
        [`ProperOrder`][oqd_core.compiler.analog.rewrite.canonicalize.ProperOrder],
        [`GatherPauli`][oqd_core.compiler.analog.rewrite.canonicalize.GatherPauli]

    Example:
        - X + 2*Y => fail
        - 1*X + 2*Y => pass
    """

    def __init__(self):
        super().__init__()
        self._single_term_scaling_needed = False

    def map_Evolve(self, model):
        self._single_term_scaling_needed = False

    def map_Expectation(self, model):
        self._single_term_scaling_needed = False

    def map_OperatorMul(self, model: OperatorMul):
        if is_scalar_mul(model):
            self._single_term_scaling_needed = True
            return
        if not self._single_term_scaling_needed:
            raise AnalogCompilerError("Single term operator has not been scaled")

    def map_OperatorKron(self, model: OperatorKron):
        if not self._single_term_scaling_needed:
            raise AnalogCompilerError("Single term operator has not been scaled")

    def map_OperatorTerminal(self, model: OperatorTerminal):
        if not self._single_term_scaling_needed:
            raise AnalogCompilerError("Single term operator has not been scaled")

    def map_OperatorAdd(self, model: OperatorAdd):
        self._single_term_scaling_needed = True
        if is_scalar_mul(model.op2) and (
            is_scalar_mul(model.op1) or isinstance(model.op1, OperatorAdd)
        ):
            return
        raise AnalogCompilerError(
            "some operators between addition are not scaled properly"
        )
