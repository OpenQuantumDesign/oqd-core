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


from oqd_compiler_infrastructure import ConversionRule, Post

########################################################################################

from oqd_core.interface.analog import (
    OperatorAdd,
    OperatorBinaryOp,
    OperatorKron,
    OperatorMul,
    OperatorScalarMul,
    OperatorSub,
    OperatorTerminal,
)
from oqd_core.interface.math import MathExpr
from oqd_core.compiler.math.rules import PrintMathExpr

########################################################################################

__all__ = [
    "PrintOperator",
]

########################################################################################


class PrintOperator(ConversionRule):
    """
    ConversionRule which prints operators. Level of
    verbosity can be set through the verbose attribute

    Args:
        model (Operator): [`Operator`][oqd_core.interface.analog.operator.Operator] of Analog level

    Returns:
        string (str):

    Assumptions:
        None

    Example:
        X@Y => str(X@Y)
    """

    def __init__(self, *, verbose=False):
        super().__init__()
        self.verbose = verbose

    def map_OperatorTerminal(self, model: OperatorTerminal, operands):
        return model.class_ + "()"

    def map_MathExpr(self, model: MathExpr, operands):
        return Post(PrintMathExpr(verbose=self.verbose))(model)

    def map_OperatorAdd(self, model: OperatorAdd, operands):
        if self.verbose:
            return self._map_OperatorBinaryOp(model, operands)
        string = "{} + {}".format(operands["op1"], operands["op2"])
        return string

    def map_OperatorSub(self, model: OperatorSub, operands):
        if self.verbose:
            return self._map_OperatorBinaryOp(model, operands)
        s2 = (
            f"({operands['op2']})"
            if isinstance(model.op2, (OperatorAdd, OperatorSub))
            else operands["op2"]
        )
        string = "{} - {}".format(operands["op1"], s2)
        return string

    def map_OperatorMul(self, model: OperatorMul, operands):
        if self.verbose:
            return self._map_OperatorBinaryOp(model, operands)
        s1 = (
            f"({operands['op1']})"
            if isinstance(
                model.op1, (OperatorAdd, OperatorSub, OperatorKron, OperatorScalarMul)
            )
            else operands["op1"]
        )
        s2 = (
            f"({operands['op2']})"
            if isinstance(
                model.op2, (OperatorAdd, OperatorSub, OperatorKron, OperatorScalarMul)
            )
            else operands["op2"]
        )
        string = "{} * {}".format(s1, s2)
        return string

    def map_OperatorKron(self, model: OperatorKron, operands):
        if self.verbose:
            return self._map_OperatorBinaryOp(model, operands)
        s1 = (
            f"({operands['op1']})"
            if isinstance(
                model.op1, (OperatorAdd, OperatorSub, OperatorMul, OperatorScalarMul)
            )
            else operands["op1"]
        )
        s2 = (
            f"({operands['op2']})"
            if isinstance(
                model.op2, (OperatorAdd, OperatorSub, OperatorMul, OperatorScalarMul)
            )
            else operands["op2"]
        )

        string = "{} @ {}".format(s1, s2)
        return string

    def map_OperatorScalarMul(self, model: OperatorScalarMul, operands):
        if self.verbose:
            s1 = (
                f"({operands['op']})"
                if not isinstance(model.op, OperatorTerminal)
                else operands["op"]
            )
            s2 = f"({operands['expr']})"
            string = f"{s2} * {s1}"
            return string
        s1 = (
            f"({operands['op']})"
            if isinstance(
                model.op, (OperatorAdd, OperatorSub, OperatorMul, OperatorKron)
            )
            else operands["op"]
        )
        s2 = f"({operands['expr']})"

        string = f"{s2} * {s1}"
        return string

    def _map_OperatorBinaryOp(self, model: OperatorBinaryOp, operands):
        s1 = (
            f"({operands['op1']})"
            if not isinstance(model.op1, OperatorTerminal)
            else operands["op1"]
        )
        s2 = (
            f"({operands['op2']})"
            if not isinstance(model.op2, OperatorTerminal)
            else operands["op2"]
        )
        operator_dict = dict(
            OperatorAdd="+", OperatorSub="-", OperatorMul="*", OperatorKron="@"
        )
        string = f"{s1} {operator_dict[model.__class__.__name__]} {s2}"
        return string
