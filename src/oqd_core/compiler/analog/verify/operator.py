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

from oqd_core.compiler.analog.error import CanonicalFormError
from oqd_core.compiler.analog.term_index import is_scalar_mul, coeff_and_op


########################################################################################
from oqd_core.interface.analog.expr import (
    Ladder,
    OperatorAdd,
    OperatorMul,
    OperatorTerminal,
    Pauli,
)

########################################################################################

__all__ = [
    "VerifyHilbertSpaceDim",
]

########################################################################################


class VerifyHilbertSpaceDim(RewriteRule):
    """
    Checks whether the hilbert spaces are correct between additions.

    Args:
        model (VisitableBaseModel):
            The rule only verifies [`Operator`][oqd_core.interface.analog.operator.Operator] in Analog level

    Returns:
        model (VisitableBaseModel): unchanged

    Assumptions:
        [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr],
        [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute],
        [`ProperOrder`][oqd_core.compiler.analog.rewrite.canonicalize.ProperOrder],
        [`PauliAlgebra`][oqd_core.compiler.analog.rewrite.canonicalize.PauliAlgebra]

    Example:
        - X@A + Y@Z => fail
        - X@Y + Y@Z => pass
    """

    def __init__(self):
        super().__init__()
        self._dim = (0, 0)
        self._term_dim = None
        self._final_add_term = False

    def _reset(self):
        self._dim = (0, 0)
        self._term_dim = None
        self._final_add_term = False

    def map_Evolve(self, model):
        self._reset()

    def map_Expectation(self, model):
        self._reset()
    
    def _term_dim_from_node(self, node):
        _, op = coeff_and_op(node)
        if isinstance(op, (OperatorTerminal, OperatorMul)):
            return self._get_dim(op)
        return self._dim

    def _get_dim(self, model):
        if isinstance(model, Pauli):
            return (1, 0)
        elif isinstance(model, Union[Ladder, OperatorMul]):
            return (0, 1)

    def map_OperatorKron(self, model):
        new = self._get_dim(model.op2)
        self._dim = (self._dim[0] + new[0], self._dim[1] + new[1])
        if isinstance(model.op1, (OperatorTerminal, OperatorMul)):
            new = self._get_dim(model.op1)
            self._dim = (self._dim[0] + new[0], self._dim[1] + new[1])
            if self._final_add_term:
                if self._term_dim != self._dim:
                    raise CanonicalFormError("Incorrect Hilbert space dimension")

    def map_OperatorAdd(self, model):
        new = self._term_dim_from_node(model.op2)
        
        if self._term_dim is None:
            self._term_dim = new
        else:
            if self._term_dim != new:
                raise CanonicalFormError("Incorrect Hilbert space dimension")
            
        if isinstance(model.op1, OperatorAdd):
            pass
        elif is_scalar_mul(model.op1):
            _, op = coeff_and_op(model.op1)
            if isinstance(op, Union[OperatorTerminal, OperatorMul]):
                if self._term_dim != self._get_dim(op):
                    raise CanonicalFormError("Incorrect Hilbert space dimension")
        elif isinstance(model.op1, Union[OperatorTerminal, OperatorMul]):
            if self._term_dim != self._get_dim(model.op1):
                raise CanonicalFormError("Incorrect Hilbert space dimension")

        if not isinstance(model.op1, OperatorAdd):
            self._final_add_term = True
        self._dim = (0, 0)
