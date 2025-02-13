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

import math
from typing import Union

import numpy as np
from oqd_compiler_infrastructure import ConversionRule, RewriteRule

########################################################################################
from oqd_core.interface.math import (
    MathAdd,
    MathBinaryOp,
    MathDiv,
    MathFunc,
    MathImag,
    MathMul,
    MathNum,
    MathPow,
    MathSub,
    MathTerminal,
    MathUnaryOp,
    MathVar,
)

########################################################################################

__all__ = [
    "PrintMathExpr",
    "DistributeMathExpr",
    "PartitionMathExpr",
    "ProperOrderMathExpr",
    "PruneMathExpr",
    "SimplifyMathExpr",
    "EvaluateMathExpr",
]

########################################################################################


class PrintMathExpr(ConversionRule):
    """
    This prints [`MathExpr`][oqd_core.interface.math.MathExpr] objects. Verbosity level can be given as an attribute.

    Args:
        model (MathExpr): The rule only acts on [`MathExpr`][oqd_core.interface.math.MathExpr] objects.

    Returns:
        string (str):

    Assumptions:
        None

    Example:
        MathAdd(expr1 = 2, expr2 = 3) => str(2 + 3)
    """

    def __init__(self, *, verbose=False):
        super().__init__()

        self.verbose = verbose

    def map_MathVar(self, model: MathVar, operands):
        string = "{}".format(model.name)
        return string

    def map_MathNum(self, model: MathNum, operands):
        string = "{}".format(model.value)
        return string

    def map_MathImag(self, model: MathImag, operands):
        string = "1j"
        return string

    def map_MathFunc(self, model: MathFunc, operands):
        string = "{}({})".format(model.func, operands["expr"])
        return string

    def map_MathAdd(self, model: MathAdd, operands):
        if self.verbose:
            return self._map_MathBinaryOp(model, operands)
        string = "{} + {}".format(operands["expr1"], operands["expr2"])
        return string

    def map_MathSub(self, model: MathSub, operands):
        if self.verbose:
            return self._map_MathBinaryOp(model, operands)
        s2 = (
            f"({operands['expr2']})"
            if isinstance(model.expr2, (MathAdd, MathSub))
            else operands["expr2"]
        )
        string = "{} - {}".format(operands["expr1"], s2)
        return string

    def map_MathMul(self, model: MathMul, operands):
        if self.verbose:
            return self._map_MathBinaryOp(model, operands)
        s1 = (
            f"({operands['expr1']})"
            if isinstance(operands["expr1"], (MathAdd, MathSub))
            else operands["expr1"]
        )
        s2 = (
            f"({operands['expr2']})"
            if isinstance(model.expr2, (MathAdd, MathSub))
            else operands["expr2"]
        )

        string = "{} * {}".format(s1, s2)
        return string

    def map_MathDiv(self, model: MathDiv, operands):
        if self.verbose:
            return self._map_MathBinaryOp(model, operands)
        s1 = (
            f"({operands['expr1']})"
            if isinstance(model.expr1, (MathAdd, MathSub))
            else operands["expr1"]
        )
        s2 = (
            f"({operands['expr2']})"
            if isinstance(model.expr2, (MathAdd, MathSub))
            else operands["expr2"]
        )

        string = "{} / {}".format(s1, s2)
        return string

    def map_MathPow(self, model: MathPow, operands):
        if self.verbose:
            return self._map_MathBinaryOp(model, operands)
        s1 = (
            f"({operands['expr1']})"
            if isinstance(model.expr1, (MathAdd, MathSub, MathMul, MathDiv))
            else operands["expr1"]
        )
        s2 = (
            f"({operands['expr2']})"
            if isinstance(model.expr2, (MathAdd, MathSub, MathMul, MathDiv))
            else operands["expr2"]
        )

        string = "{} ** {}".format(s1, s2)
        return string

    def _map_MathBinaryOp(self, model: MathBinaryOp, operands):
        s1 = (
            f"({operands['expr1']})"
            if not isinstance(model.expr1, (MathTerminal, MathUnaryOp))
            else operands["expr1"]
        )
        s2 = (
            f"({operands['expr2']})"
            if not isinstance(model.expr2, (MathTerminal, MathUnaryOp))
            else operands["expr2"]
        )
        operator_dict = dict(
            MathAdd="+", MathSub="-", MathMul="*", MathDiv="/", MathPow="**"
        )
        string = f"{s1} {operator_dict[model.__class__.__name__]} {s2}"
        return string


########################################################################################


class DistributeMathExpr(RewriteRule):
    """
    This distributes [`MathExpr`][oqd_core.interface.math.MathExpr] objects.

    Args:
        model (MathExpr): The rule only acts on [`MathExpr`][oqd_core.interface.math.MathExpr] objects.

    Returns:
        model (MathExpr):

    Assumptions:
        None

    Example:
        MathStr(string = '3 * (2 + 1)') => MathStr(string = '3 * 2 + 3 * 1')
    """

    def map_MathMul(self, model: MathMul):
        if isinstance(model.expr1, (MathAdd, MathSub)):
            return model.expr1.__class__(
                expr1=MathMul(expr1=model.expr1.expr1, expr2=model.expr2),
                expr2=MathMul(expr1=model.expr1.expr2, expr2=model.expr2),
            )
        if isinstance(model.expr2, (MathAdd, MathSub)):
            return model.expr2.__class__(
                expr1=MathMul(expr1=model.expr1, expr2=model.expr2.expr1),
                expr2=MathMul(expr1=model.expr1, expr2=model.expr2.expr2),
            )
        pass

    def map_MathSub(self, model: MathSub):
        return MathAdd(
            expr1=model.expr1,
            expr2=MathMul(expr1=MathNum(value=-1), expr2=model.expr2),
        )

    def map_MathDiv(self, model: MathDiv):
        return MathMul(
            expr1=model.expr1,
            expr2=MathPow(expr1=model.expr2, expr2=MathNum(value=-1)),
        )


class PartitionMathExpr(RewriteRule):
    """
    This separates real and complex portions of [`MathExpr`][oqd_core.interface.math.MathExpr] objects.

    Args:
        model (MathExpr): The rule only acts on [`MathExpr`][oqd_core.interface.math.MathExpr] objects.

    Returns:
        model (MathExpr):

    Assumptions:
        [`DistributeMathExpr`][oqd_core.compiler.math.rules.DistributeMathExpr],
        [`ProperOrderMathExpr`][oqd_core.compiler.math.rules.ProperOrderMathExpr]

    Example:
        - MathStr(string = '1 + 1j + 2') => MathStr(string = '1 + 2 + 1j')
        - MathStr(string = '1 * 1j * 2') => MathStr(string = '1j * 1 * 2')
    """

    def map_MathAdd(self, model):
        if not (
            isinstance(model.expr2, MathImag)
            or (
                isinstance(model.expr2, MathMul)
                and isinstance(model.expr2.expr1, MathImag)
            )
            or isinstance(model.expr2, MathAdd)
        ):
            return MathAdd(expr1=model.expr2, expr2=model.expr1)

    def map_MathMul(self, model: MathMul):
        priority = dict(MathImag=4, MathNum=3, MathVar=2, MathFunc=1, MathPow=0)

        if isinstance(model.expr2, (MathImag, MathNum, MathVar, MathFunc, MathPow)):
            if isinstance(model.expr1, MathMul):
                if (
                    priority[model.expr2.__class__.__name__]
                    > priority[model.expr1.expr2.__class__.__name__]
                ):
                    return MathMul(
                        expr1=MathMul(expr1=model.expr1.expr1, expr2=model.expr2),
                        expr2=model.expr1.expr2,
                    )
            else:
                if (
                    priority[model.expr2.__class__.__name__]
                    > priority[model.expr1.__class__.__name__]
                ):
                    return MathMul(
                        expr1=model.expr2,
                        expr2=model.expr1,
                    )
        pass


class ProperOrderMathExpr(RewriteRule):
    """
    This rearranges bracketing of [`MathExpr`][oqd_core.interface.math.MathExpr] objects.

    Args:
        model (MathExpr): The rule only acts on [`MathExpr`][oqd_core.interface.math.MathExpr] objects.

    Returns:
        model (MathExpr):

    Assumptions:
        [`DistributeMathExpr`][oqd_core.compiler.math.rules.DistributeMathExpr]

    Example:
        - MathStr(string = '2 * (3 * 5)') => MathStr(string = '(2 * 3) * 5')
    """

    def map_MathAdd(self, model: MathAdd):
        return self._MathAddMul(model)

    def map_MathMul(self, model: MathMul):
        return self._MathAddMul(model)

    def _MathAddMul(self, model: Union[MathAdd, MathMul]):
        if isinstance(model.expr2, model.__class__):
            return model.__class__(
                expr1=model.__class__(expr1=model.expr1, expr2=model.expr2.expr1),
                expr2=model.expr2.expr2,
            )
        pass


class PruneMathExpr(RewriteRule):
    """
    This is constant fold operation where scalar addition, multiplication and power are simplified
    """

    def map_MathAdd(self, model):
        if model.expr1 == MathNum(value=0):
            return model.expr2
        if model.expr2 == MathNum(value=0):
            return model.expr1

    def map_MathMul(self, model):
        if model.expr1 == MathNum(value=1):
            return model.expr2
        if model.expr2 == MathNum(value=1):
            return model.expr1

        if model.expr1 == MathNum(value=0) or model.expr2 == MathNum(value=0):
            return MathNum(value=0)

    def map_MathPow(self, model):
        if model.expr1 == MathNum(value=1) or model.expr2 == MathNum(value=1):
            return model.expr1


class SimplifyMathExpr(RewriteRule):
    """
    This simplified MathExpr objects
    """

    def map_MathAdd(self, model):
        if isinstance(model.expr1, MathNum) and isinstance(model.expr2, MathNum):
            return MathNum(value=model.expr1.value + model.expr2.value)

    def map_MathMul(self, model):
        if isinstance(model.expr1, MathNum) and isinstance(model.expr2, MathNum):
            return MathNum(value=model.expr1.value * model.expr2.value)

    def map_MathPow(self, model):
        if isinstance(model.expr1, MathNum) and isinstance(model.expr2, MathNum):
            return MathNum(value=model.expr1.value**model.expr2.value)

    def map_MathFunc(self, model):
        if isinstance(model.expr, MathNum):
            if getattr(math, model.func, None):
                value = getattr(math, model.func)(model.expr.value)

            if model.func == "heaviside":
                value = np.heaviside(model.expr.value, 1)

            if model.func == "conj":
                value = np.conj(model.expr.value)
            return MathNum(value=value)


########################################################################################


class EvaluateMathExpr(ConversionRule):
    """
    This evalaluates MathExpr objects
    """

    def map_MathVar(self, model: MathVar, operands):
        raise TypeError

    def map_MathNum(self, model: MathNum, operands):
        return model.value

    def map_MathImag(self, model: MathImag, operands):
        return complex("1j")

    def map_MathFunc(self, model: MathFunc, operands):
        if getattr(math, model.func, None):
            return getattr(math, model.func)(operands["expr"])

        if model.func == "heaviside":
            return np.heaviside(operands["expr"], 1)

        if model.func == "conj":
            return np.conj(operands["expr"])

    def map_MathAdd(self, model: MathAdd, operands):
        return operands["expr1"] + operands["expr2"]

    def map_MathSub(self, model: MathSub, operands):
        return operands["expr1"] - operands["expr2"]

    def map_MathMul(self, model: MathMul, operands):
        return operands["expr1"] * operands["expr2"]

    def map_MathDiv(self, model: MathDiv, operands):
        return operands["expr1"] / operands["expr2"]

    def map_MathPow(self, model: MathPow, operands):
        return operands["expr1"] ** operands["expr2"]
