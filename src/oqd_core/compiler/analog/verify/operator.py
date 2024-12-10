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

from typing import Union

from oqd_compiler_infrastructure import RewriteRule

########################################################################################

from oqd_core.interface.analog import (
    Ladder,
    OperatorAdd,
    OperatorMul,
    OperatorScalarMul,
    OperatorTerminal,
    Pauli,
)

########################################################################################

__all__ = [
    "VerifyHilberSpaceDim",
]

########################################################################################


class VerifyHilberSpaceDim(RewriteRule):
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

    def map_AnalogGate(self, model):
        self._reset()

    def map_Expectation(self, model):
        self._reset()

    def _get_dim(self, model):
        if isinstance(model, Pauli):
            return (1, 0)
        elif isinstance(model, Union[Ladder, OperatorMul]):
            return (0, 1)

    def map_OperatorKron(self, model):
        new = self._get_dim(model.op2)
        self._dim = (self._dim[0] + new[0], self._dim[1] + new[1])
        if isinstance(model.op1, Union[OperatorTerminal, OperatorMul]):
            new = self._get_dim(model.op1)
            self._dim = (self._dim[0] + new[0], self._dim[1] + new[1])
            if self._final_add_term:
                assert self._term_dim == self._dim, "Incorrect Hilbert space dimension"

    def map_OperatorAdd(self, model):
        new = self._dim
        if isinstance(model.op2, Union[OperatorMul, OperatorTerminal]):
            new = self._get_dim(model.op2)
        elif isinstance(model.op2, OperatorScalarMul):
            if isinstance(model.op2.op, Union[OperatorTerminal, OperatorMul]):
                new = self._get_dim(model.op2.op)

        if self._term_dim is None:
            self._term_dim = new
        else:
            assert self._term_dim == new, "Incorrect Hilbert space dimension"

        if isinstance(model.op1, Union[OperatorTerminal, OperatorMul]):
            assert self._term_dim == self._get_dim(
                model.op1
            ), "Incorrect Hilbert space dimension"
        elif isinstance(model.op1, OperatorScalarMul):
            if isinstance(model.op1.op, Union[OperatorTerminal, OperatorMul]):
                assert self._term_dim == self._get_dim(
                    model.op1.op
                ), "Incorrect Hilbert space dimension"

        if not isinstance(model.op1, OperatorAdd):
            self._final_add_term = True
        self._dim = (0, 0)
