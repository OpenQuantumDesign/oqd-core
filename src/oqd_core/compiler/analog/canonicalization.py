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

from typing import Union

import numpy as np
from oqd_compiler_infrastructure import (
    Chain,
    ConversionRule,
    FixedPoint,
    In,
    Post,
    Pre,
    RewriteRule,
)
from pydantic import ValidationError

from oqd_core.compiler.analog.error import AnalogCompilerError
from oqd_core.interface.analog.expr import (
    Access,
    Add,
    And,
    Annihilation,
    ArithOp,
    BinaryOp,
    BuiltinCall,
    Constant,
    Creation,
    Div,
    Identity,
    Kron,
    Ladder,
    Mul,
    Neg,
    Not,
    Operator,
    Or,
    Pauli,
    PauliI,
    PauliX,
    PauliY,
    PauliZ,
    Pos,
    Pow,
    RuntimeVar,
    Sub,
    Xor,
)

########################################################################################

# __all__ = [
#     "DistributeMathExpr",
#     "PartitionMathExpr",
#     "ProperOrderMathExpr",
#     "PruneMathExpr",
#     "SimplifyMathExpr",
#     "EvaluateMathExpr",
#     "SubstituteMathVar",
#     "canonicalize_operator_expr",
#     "evaluate_math_expr",
#     "simplify_math_expr",
#     "canonicalize_math_expr",
#     "GatherMathExpr",
#     "GatherPauli",
#     "PruneIdentity",
#     "PauliAlgebra",
#     "NormalOrder",
#     "ProperOrder",
#     "ScaleTerms",
#     "SortedOrder",
#     "PruneZeros",
# ]

########################################################################################


def canonical_ordering(model):
    match model:
        case Constant():
            return 0
        case _:
            return 1


class ReplaceSubtraction(RewriteRule):
    def map_Sub(self, model):
        exprs = [model.exprs[0]] + [Neg(expr=expr) for expr in model.exprs[1:]]
        return Add(exprs=exprs)

    def map_Div(self, model):
        exprs = [model.exprs[0]] + [
            Pow(exprs=[expr, Constant(value=-1)]) for expr in model.exprs[1:]
        ]
        return Mul(exprs=exprs)


class UnaryCombination(RewriteRule):
    def map_Pos(self, model):
        return model.expr

    def map_Neg(self, model):
        match model.expr:
            case Neg():
                return model.expr.expr

    def map_Not(self, model):
        match model.expr:
            case Not():
                return model.expr


class CollectAssociativeOperands(RewriteRule):
    def map_Add(self, model):
        exprs = []
        for e in model.exprs:
            match e:
                case Add():
                    exprs.extend(e.exprs)
                    continue
                case Pos(expr=Add()):
                    exprs.extend(e.expr.exprs)
                case Neg(expr=Add()):
                    exprs.extend([Neg(expr=_e) for _e in e.expr.exprs])
                case _:
                    exprs.append(e)

        return Add(exprs=exprs)

    def map_Mul(self, model):
        exprs = []

        neg = False
        for e in model.exprs:
            match e:
                case Mul():
                    exprs.extend(e.exprs)
                    continue
                case Neg():
                    exprs.append(e.expr)
                    neg = not neg
                case _:
                    exprs.append(e)

        if neg:
            return Neg(expr=Mul(exprs=exprs))
        return Mul(exprs=exprs)


class SortOperands(RewriteRule):
    def map_Add(self, model):
        exprs = sorted(model.exprs, key=canonical_ordering)

        return Add(exprs=exprs)

    def map_Mul(self, model):
        exprs = sorted(model.exprs, key=canonical_ordering)

        return Mul(exprs=exprs)


# class GatherMathExpr(RewriteRule):
#     """
#     Gathers the math expressions of  [`Operator`][oqd_core.interface.analog.expr.OperatorExpr] so that we have math_expr * ( [`Operator`][oqd_core.interface.analog.expr.OperatorExpr] without scalar multiplication)

#     Args:
#         model (VisitableBaseModel):

#     Returns:
#         model (VisitableBaseModel):

#     Assumptions:
#         [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute] (sometimes)

#     Example:
#         (1 * X) @ (2 * Y) => (1 * 2) => (1 * 2) * (X @ Y)
#     """

#     def map_Mul(self, model: Mul):
#         if is_scalar_mul(model):
#             c1, inner = coeff_and_op(model)
#             if is_scalar_mul(inner):
#                 c2, op = coeff_and_op(inner)
#                 return scalar_mul(MathMul(expr1=c1, expr2=c2), op)
#         return self._mulkron(model)

#     def map_Kron(self, model: Kron):
#         return self._mulkron(model)

#     def _mulkron(self, model):
#         c1, t1 = coeff_and_op(model.op1)
#         c2, t2 = coeff_and_op(model.op2)
#         if c1 is None or c2 is None:
#             return None
#         if not (is_scalar_mul(model.op1) or is_scalar_mul(model.op2)):
#             return None
#         coeff = MathMul(expr1=c1, expr2=c2)
#         return scalar_mul(coeff, model.__class__(op1=t1, op2=t2))


# class GatherPauli(RewriteRule):
#     """
#     Gathers ladders and paulis so that we have paulis and then ladders

#     Args:
#         model (VisitableBaseModel):

#     Returns:
#         model (VisitableBaseModel):

#     Assumptions:
#         [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr],
#         [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute],
#         [`ProperOrder`][oqd_core.compiler.analog.rewrite.canonicalize.ProperOrder]
#         [`Operator`][oqd_core.interface.analog.expr.OperatorExpr]

#     Example:
#         X@A@Y => X@Y@A
#     """

#     def map_OperatorKron(self, model: OperatorKron):
#         _, op1 = coeff_and_op(model.op1)
#         if isinstance(model.op2, Pauli):
#             if isinstance(op1, Ladder):
#                 return OperatorKron(
#                     op1=model.op2,
#                     op2=op1,
#                 )
#             if isinstance(op1, OperatorMul) and isinstance(op1.op2, Ladder):
#                 return OperatorKron(
#                     op1=model.op2,
#                     op2=op1,
#                 )
#             if isinstance(op1, OperatorKron) and isinstance(
#                 op1.op2, Union[Ladder, OperatorMul]
#             ):
#                 return OperatorKron(
#                     op1=OperatorKron(op1=op1.op1, op2=model.op2),
#                     op2=model.op1.op2,
#                 )
#         return None


# class PruneIdentity(RewriteRule):
#     """
#     Removes unnecessary ladder Identities from operators

#     Args:
#         model (VisitableBaseModel):

#     Returns:
#         model (VisitableBaseModel):

#     Assumptions:
#         [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr],
#         [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute],
#         [`ProperOrder`][oqd_core.compiler.analog.rewrite.canonicalize.ProperOrder],
#         [`GatherPauli`][oqd_core.compiler.analog.rewrite.canonicalize.GatherPauli],
#         [`NormalOrder`][oqd_core.compiler.analog.rewrite.canonicalize.NormalOrder]

#     Example:
#         A*J => A
#     """

#     def map_OperatorMul(self, model: OperatorMul):
#         if is_scalar_mul(model):
#             return None
#         if isinstance(model.op1, (Identity)):
#             return model.op2
#         if isinstance(model.op2, (Identity)):
#             return model.op1
#         return None


# class PauliAlgebra(RewriteRule):
#     """
#     RewriteRule for Pauli algebra operations

#     Args:
#         model (VisitableBaseModel):

#     Returns:
#         model (VisitableBaseModel):

#     Assumptions:
#         [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr],
#         [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute],
#         [`ProperOrder`][oqd_core.compiler.analog.rewrite.canonicalize.ProperOrder]

#     Example:
#         X*Y => iZ
#     """

#     def map_OperatorMul(self, model: OperatorMul):
#         if is_scalar_mul(model):
#             return None
#         if isinstance(model.op1, Pauli) and isinstance(model.op2, Pauli):
#             if isinstance(model.op1, PauliI):
#                 return model.op2
#             if isinstance(model.op2, PauliI):
#                 return model.op1
#             if model.op1 == model.op2:
#                 return PauliI()
#             if isinstance(model.op1, PauliX) and isinstance(model.op2, PauliY):
#                 return scalar_mul(MathImag(), PauliZ())
#             if isinstance(model.op1, PauliY) and isinstance(model.op2, PauliZ):
#                 return scalar_mul(MathImag(), PauliX())
#             if isinstance(model.op1, PauliZ) and isinstance(model.op2, PauliX):
#                 return scalar_mul(MathImag(), PauliY())
#             return scalar_mul(
#                 MathNum(value=-1),
#                 OperatorMul(op1=model.op2, op2=model.op1),
#             )
#         return None


# class NormalOrder(RewriteRule):
#     """
#     Arranges Ladder oeprators in normal order form

#     Args:
#         model (VisitableBaseModel):

#     Returns:
#         model (VisitableBaseModel):

#     Assumptions:
#         [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr],
#         [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute],
#         [`ProperOrder`][oqd_core.compiler.analog.rewrite.canonicalize.ProperOrder],
#         [`GatherPauli`][oqd_core.compiler.analog.rewrite.canonicalize.GatherPauli]

#     Example:
#         A*C => C*A + J
#     """

#     def map_OperatorMul(self, model: OperatorMul):
#         if is_scalar_mul(model):
#             return None
#         if isinstance(model.op2, Creation):
#             if isinstance(model.op1, Annihilation):
#                 return OperatorAdd(
#                     op1=OperatorMul(op1=model.op2, op2=model.op1), op2=Identity()
#                 )
#             if isinstance(model.op1, Identity):
#                 return OperatorMul(op1=model.op2, op2=model.op1)
#             if isinstance(model.op1, OperatorMul) and isinstance(
#                 model.op1.op2, (Annihilation, Identity)
#             ):
#                 return OperatorMul(
#                     op1=model.op1.op1,
#                     op2=OperatorMul(op1=model.op1.op2, op2=model.op2),
#                 )
#         return None


# class ProperOrder(RewriteRule):
#     """
#     Converts expressions to proper order bracketing. Please see example for clarification.

#     Args:
#         model (VisitableBaseModel):

#     Returns:
#         model (VisitableBaseModel):

#     Assumptions:
#         [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr],
#         [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute]

#     Example:
#         X @ (Y @ Z) =>  (X @ Y) @ Z
#     """

#     def map_OperatorAdd(self, model: OperatorAdd):
#         return self._addmullkron(model=model)

#     def map_OperatorMul(self, model: OperatorMul):
#         if is_scalar_mul(model):
#             return None
#         return self._addmullkron(model=model)

#     def map_OperatorKron(self, model: OperatorKron):
#         return self._addmullkron(model=model)

#     def _addmullkron(self, model: Union[OperatorAdd, OperatorMul, OperatorKron]):
#         if isinstance(model.op2, model.__class__):
#             return model.__class__(
#                 op1=model.__class__(op1=model.op1, op2=model.op2.op1),
#                 op2=model.op2.op2,
#             )
#         return model.__class__(op1=model.op1, op2=model.op2)


# class ScaleTerms(RewriteRule):
#     """
#     Scales operators to ensure consistency

#     Args:
#         model (VisitableBaseModel):

#     Returns:
#         model (VisitableBaseModel):

#     Assumptions:
#         [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr],
#         [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute],
#         [`ProperOrder`][oqd_core.compiler.analog.rewrite.canonicalize.ProperOrder],
#         [`GatherPauli`][oqd_core.compiler.analog.rewrite.canonicalize.GatherPauli],
#         [`NormalOrder`][oqd_core.compiler.analog.rewrite.canonicalize.NormalOrder],
#         [`PruneIdentity`][oqd_core.compiler.analog.rewrite.canonicalize.PruneIdentity]

#     Note:
#         - Requires [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr] right after application of [`ScaleTerms`][oqd_core.compiler.analog.rewrite.canonicalize.ScaleTerms]  for Post walk
#         - [`SortedOrder`][oqd_core.compiler.analog.rewrite.canonicalize.SortedOrder] and  [`ScaleTerms`][oqd_core.compiler.analog.rewrite.canonicalize.ScaleTerms] can be run in either order

#     Example:
#         X + Y + 2*Z => 1*X + 1*Y + 2*Z
#         X@Y => 1*(X@Y)
#     """

#     def __init__(self):
#         super().__init__()
#         self.op_add_root = False

#     def map_Evolve(self, model):
#         self.op_add_root = False

#     def map_Expectation(self, model):
#         self.op_add_root = False

#     def map_OperatorAdd(self, model: OperatorAdd):
#         self.op_add_root = True
#         op1 = (
#             model.op1
#             if (is_scalar_mul(model.op1) or isinstance(model.op1, OperatorAdd))
#             else scalar_mul(MathNum(value=1), model.op1)
#         )
#         op2 = (
#             model.op2
#             if (is_scalar_mul(model.op2) or isinstance(model.op2, OperatorAdd))
#             else scalar_mul(MathNum(value=1), model.op2)
#         )
#         return OperatorAdd(op1=op1, op2=op2)

#     def map_OperatorTerminal(self, model):
#         if not self.op_add_root:
#             self.op_add_root = True
#             return scalar_mul(MathNum(value=1), model)

#     def map_OperatorKron(self, model):
#         if not self.op_add_root:
#             self.op_add_root = True
#             return scalar_mul(MathNum(value=1), model)

#     def map_OperatorMul(self, model):
#         if is_scalar_mul(model):
#             return None
#         if not self.op_add_root:
#             self.op_add_root = True
#             return scalar_mul(MathNum(value=1), model)


# class SortedOrder(RewriteRule):
#     """
#     Sorts operators based on TermIndex and collects duplicate terms.
#     Please see example for clarification

#     Args:
#         model (VisitableBaseModel):

#     Returns:
#         model (VisitableBaseModel)

#     Assumptions:
#         [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr],
#         [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute],
#         [`ProperOrder`][oqd_core.compiler.analog.rewrite.canonicalize.ProperOrder],
#         [`GatherPauli`][oqd_core.compiler.analog.rewrite.canonicalize.GatherPauli],
#         [`NormalOrder`][oqd_core.compiler.analog.rewrite.canonicalize.NormalOrder],
#         [`PruneIdentity`][oqd_core.compiler.analog.rewrite.canonicalize.PruneIdentity]

#     Note:
#         - [`SortedOrder`][oqd_core.compiler.analog.rewrite.canonicalize.SortedOrder] and  [`ScaleTerms`][oqd_core.compiler.analog.rewrite.canonicalize.ScaleTerms] can be run in either order

#     Example:
#         (X@Y) + (X@I) => (X@I) + (X@Y)
#         X + I + Z + Y => I + X + Y + Z
#     """

#     def map_OperatorAdd(self, model: OperatorAdd):
#         if isinstance(model.op1, OperatorAdd):
#             term1 = term_index(model.op1.op2)
#             term2 = term_index(model.op2)

#             if term1 == term2:
#                 expr1, _ = coeff_and_op(model.op1.op2)
#                 expr2, op_part = coeff_and_op(model.op2)
#                 return OperatorAdd(
#                     op1=model.op1.op1,
#                     op2=scalar_mul(
#                         MathAdd(expr1=expr1, expr2=expr2),
#                         op_part,
#                     ),
#                 )

#             elif term1 > term2:
#                 return OperatorAdd(
#                     op1=OperatorAdd(op1=model.op1.op1, op2=model.op2),
#                     op2=model.op1.op2,
#                 )

#             elif term1 < term2:
#                 return OperatorAdd(op1=model.op1, op2=model.op2)

#         else:
#             if isinstance(model.op1, Access) and isinstance(model.op2, Access):
#                 return model
#             if isinstance(model.op1, Access):
#                 return OperatorAdd(op1=model.op1, op2=model.op2)
#             if isinstance(model.op2, Access):
#                 return OperatorAdd(op1=model.op2, op2=model.op1)
#             term1 = term_index(model.op1)
#             term2 = term_index(model.op2)

#             if term1 == term2:
#                 expr1, _ = coeff_and_op(model.op1)
#                 expr2, op_part = coeff_and_op(model.op2)
#                 return scalar_mul(MathAdd(expr1=expr1, expr2=expr2), op_part)

#             elif term1 > term2:
#                 return OperatorAdd(
#                     op1=model.op2,
#                     op2=model.op1,
#                 )

#             elif term1 < term2:
#                 return OperatorAdd(op1=model.op1, op2=model.op2)


# class PruneZeros(RewriteRule):
#     """
#     Removes operators multiplied by zero

#     Args:
#         model (VisitableBaseModel):

#     Returns:
#         model (VisitableBaseModel):

#     Assumptions:
#         [`GatherMathExpr`][oqd_core.compiler.analog.rewrite.canonicalize.GatherMathExpr],
#         [`OperatorDistribute`][oqd_core.compiler.analog.rewrite.canonicalize.OperatorDistribute],
#         [`ProperOrder`][oqd_core.compiler.analog.rewrite.canonicalize.ProperOrder],
#         [`GatherPauli`][oqd_core.compiler.analog.rewrite.canonicalize.GatherPauli],
#         [`NormalOrder`][oqd_core.compiler.analog.rewrite.canonicalize.NormalOrder]
#         [`PruneIdentity`][oqd_core.compiler.analog.rewrite.canonicalize.PruneIdentity]

#     """

#     def map_OperatorAdd(self, model):
#         c1, _ = coeff_and_op(model.op1)
#         c2, _ = coeff_and_op(model.op2)
#         if c1 == MathNum(value=0):
#             return model.op2
#         if c2 == MathNum(value=0):
#             return model.op1


# ########################################################################################

# dist_chain = Chain(
#     FixedPoint(Post(GatherMathExpr())),
# )

# pauli_chain = Chain(
#     FixedPoint(Post(PauliAlgebra())),
#     FixedPoint(Post(GatherMathExpr())),
#     FixedPoint(Post(PauliAlgebra())),
# )

# normal_order_chain = Chain(
#     FixedPoint(Post(NormalOrder())),
#     FixedPoint(Post(GatherMathExpr())),
#     FixedPoint(Post(ProperOrder())),
#     FixedPoint(Post(NormalOrder())),
# )

# scale_terms_chain = Chain(
#     FixedPoint(Pre(ScaleTerms())),
#     FixedPoint(Post(GatherMathExpr())),
# )


# def canonicalize_operator_expr(model):
#     return Chain(
#         FixedPoint(dist_chain),
#         FixedPoint(Post(ProperOrder())),
#         FixedPoint(pauli_chain),
#         FixedPoint(Post(GatherPauli())),
#         FixedPoint(normal_order_chain),
#         FixedPoint(Post(PruneIdentity())),
#         FixedPoint(scale_terms_chain),
#         FixedPoint(Post(SortedOrder())),
#         canonicalize_math_expr,
#         FixedPoint(Post(PruneZeros())),
#         verify_canonicalization,
#     )(model=model)


# ########################################################################################


# class DistributeMathExpr(RewriteRule):
#     """
#     This distributes [`MathExpr`][oqd_core.interface.analog.MathExpr] objects.

#     Args:
#         model (MathExpr): The rule only acts on [`MathExpr`][oqd_core.interface.analog.MathExpr] objects.

#     Returns:
#         model (MathExpr):

#     Assumptions:
#         None

#     Example:
#         MathStr(string = '3 * (2 + 1)') => MathStr(string = '3 * 2 + 3 * 1')
#     """

#     def map_MathMul(self, model: MathMul):
#         if isinstance(model.expr1, (MathAdd, MathSub)):
#             return model.expr1.__class__(
#                 expr1=MathMul(expr1=model.expr1.expr1, expr2=model.expr2),
#                 expr2=MathMul(expr1=model.expr1.expr2, expr2=model.expr2),
#             )
#         if isinstance(model.expr2, (MathAdd, MathSub)):
#             return model.expr2.__class__(
#                 expr1=MathMul(expr1=model.expr1, expr2=model.expr2.expr1),
#                 expr2=MathMul(expr1=model.expr1, expr2=model.expr2.expr2),
#             )
#         pass

#     def map_MathSub(self, model: MathSub):
#         return MathAdd(
#             expr1=model.expr1,
#             expr2=MathMul(expr1=MathNum(value=-1), expr2=model.expr2),
#         )

#     def map_MathDiv(self, model: MathDiv):
#         return MathMul(
#             expr1=model.expr1,
#             expr2=MathPow(expr1=model.expr2, expr2=MathNum(value=-1)),
#         )


# class PartitionMathExpr(RewriteRule):
#     """
#     This separates real and complex portions of [`MathExpr`][oqd_core.interface.analog.MathExpr] objects.

#     Args:
#         model (MathExpr): The rule only acts on [`MathExpr`][oqd_core.interface.analog.MathExpr] objects.

#     Returns:
#         model (MathExpr):

#     Assumptions:
#         [`DistributeMathExpr`][oqd_core.compiler.analog.math.rules.DistributeMathExpr],
#         [`ProperOrderMathExpr`][oqd_core.compiler.analog.math.rules.ProperOrderMathExpr]

#     Example:
#         - MathStr(string = '1 + 1j + 2') => MathStr(string = '1 + 2 + 1j')
#         - MathStr(string = '1 * 1j * 2') => MathStr(string = '1j * 1 * 2')
#     """

#     def map_MathAdd(self, model):
#         priority = dict(
#             MathImag=6, MathNum=5, MathVar=4, Access=3, MathFunc=2, MathPow=1, MathMul=0
#         )

#         if isinstance(
#             model.expr2,
#             (MathImag, MathNum, MathVar, Access, MathFunc, MathPow, MathMul),
#         ):
#             if isinstance(model.expr1, MathAdd):
#                 if (
#                     priority[model.expr2.__class__.__name__]
#                     > priority[model.expr1.expr2.__class__.__name__]
#                 ):
#                     return MathAdd(
#                         expr1=MathAdd(expr1=model.expr1.expr1, expr2=model.expr2),
#                         expr2=model.expr1.expr2,
#                     )
#             else:
#                 if (
#                     priority[model.expr2.__class__.__name__]
#                     > priority[model.expr1.__class__.__name__]
#                 ):
#                     return MathAdd(
#                         expr1=model.expr2,
#                         expr2=model.expr1,
#                     )

#     def map_MathMul(self, model: MathMul):
#         priority = dict(
#             MathImag=5, MathNum=4, MathVar=3, Access=2, MathFunc=1, MathPow=0
#         )

#         if isinstance(
#             model.expr2, (MathImag, MathNum, MathVar, Access, MathFunc, MathPow)
#         ):
#             if isinstance(model.expr1, MathMul):
#                 if (
#                     priority[model.expr2.__class__.__name__]
#                     > priority[model.expr1.expr2.__class__.__name__]
#                 ):
#                     return MathMul(
#                         expr1=MathMul(expr1=model.expr1.expr1, expr2=model.expr2),
#                         expr2=model.expr1.expr2,
#                     )
#             else:
#                 if (
#                     priority[model.expr2.__class__.__name__]
#                     > priority[model.expr1.__class__.__name__]
#                 ):
#                     return MathMul(
#                         expr1=model.expr2,
#                         expr2=model.expr1,
#                     )


# class ProperOrderMathExpr(RewriteRule):
#     """
#     This rearranges bracketing of [`MathExpr`][oqd_core.interface.analog.MathExpr] objects.

#     Args:
#         model (MathExpr): The rule only acts on [`MathExpr`][oqd_core.interface.analog.MathExpr] objects.

#     Returns:
#         model (MathExpr):

#     Assumptions:
#         [`DistributeMathExpr`][oqd_core.compiler.analog.math.rules.DistributeMathExpr]

#     Example:
#         - MathStr(string = '2 * (3 * 5)') => MathStr(string = '(2 * 3) * 5')
#     """

#     def map_MathAdd(self, model: MathAdd):
#         return self._MathAddMul(model)

#     def map_MathMul(self, model: MathMul):
#         return self._MathAddMul(model)

#     def _MathAddMul(self, model: Union[MathAdd, MathMul]):
#         if isinstance(model.expr2, model.__class__):
#             return model.__class__(
#                 expr1=model.__class__(expr1=model.expr1, expr2=model.expr2.expr1),
#                 expr2=model.expr2.expr2,
#             )
#         pass


# class PruneMathExpr(RewriteRule):
#     """
#     This is constant fold operation where scalar addition, multiplication and power are simplified

#     Args:
#         model (MathExpr): The rule only acts on [`MathExpr`][oqd_core.interface.analog.MathExpr] objects.

#     Returns:
#         model (MathExpr):

#     Assumptions:
#         None

#     """

#     def map_MathAdd(self, model):
#         if model.expr1 == MathNum(value=0):
#             return model.expr2
#         if model.expr2 == MathNum(value=0):
#             return model.expr1

#     def map_MathMul(self, model):
#         if model.expr1 == MathNum(value=1):
#             return model.expr2
#         if model.expr2 == MathNum(value=1):
#             return model.expr1

#         if model.expr1 == MathNum(value=0) or model.expr2 == MathNum(value=0):
#             return MathNum(value=0)

#     def map_MathPow(self, model):
#         if model.expr1 == MathNum(value=1) or model.expr2 == MathNum(value=1):
#             return model.expr1

#         if model.expr2 == MathNum(value=0):
#             return MathNum(value=1)

#         if model.expr1 == MathNum(value=0):
#             return MathNum(value=0)


# ########################################################################################


# class SubstituteMathVar(RewriteRule):
#     """
#     This rule substitutes a MathVar with another MathExpr

#     Args:
#         model (MathExpr): The rule only acts on [`MathExpr`][oqd_core.interface.analog.MathExpr] objects.

#     Returns:
#         model (MathExpr):

#     Assumptions:
#         None

#     """

#     def __init__(self, variable, substitution):
#         super().__init__()

#         if not isinstance(variable, MathVar):
#             raise TypeError("Variable must be a MathVar")

#         if not isinstance(variable, MathExpr):
#             raise TypeError("Substituted value must be a MathExpr")

#         self.variable = variable
#         self.substitution = substitution

#     def map_MathVar(self, model):
#         if model == self.variable:
#             return self.substitution


# ########################################################################################


# class EvaluateMathExpr(ConversionRule):
#     """
#     This evaluates MathExpr objects and raises a type error if a MathVar exist in the AST.

#     Args:
#         model (MathExpr): The rule only acts on [`MathExpr`][oqd_core.interface.analog.MathExpr] objects.

#     Returns:
#         model (MathExpr):

#     Assumptions:
#         None
#     """

#     def map_MathVar(self, model: MathVar, operands):
#         raise TypeError(
#             "Evaluation requires the substitution of all MathVar to constants"
#         )

#     def map_Access(self, model: Access, operands):
#         raise TypeError("Evaluation requires Access to be resolved")

#     def map_MathNum(self, model: MathNum, operands):
#         return model.value

#     def map_MathImag(self, model: MathImag, operands):
#         return complex("1j")

#     def map_MathFunc(self, model: MathFunc, operands):
#         if model.func in [
#             "abs",
#             "sin",
#             "cos",
#             "tan",
#             "exp",
#             "log",
#             "sinh",
#             "cosh",
#             "tanh",
#             "atan",
#             "acos",
#             "asin",
#             "atanh",
#             "asinh",
#             "acosh",
#             "conj",
#             "real",
#             "imag",
#             "atan2",
#         ]:
#             if isinstance(operands["expr"], list):
#                 return getattr(np, model.func)(*operands["expr"])

#             return getattr(np, model.func)(operands["expr"])

#         if model.func == "heaviside":
#             return np.heaviside(operands["expr"], 1)

#         raise AnalogCompilerError("Unsupported function")

#     def map_MathAdd(self, model: MathAdd, operands):
#         return operands["expr1"] + operands["expr2"]

#     def map_MathSub(self, model: MathSub, operands):
#         return operands["expr1"] - operands["expr2"]

#     def map_MathMul(self, model: MathMul, operands):
#         return operands["expr1"] * operands["expr2"]

#     def map_MathDiv(self, model: MathDiv, operands):
#         return operands["expr1"] / operands["expr2"]

#     def map_MathPow(self, model: MathPow, operands):
#         return operands["expr1"] ** operands["expr2"]


# ########################################################################################


# class SimplifyMathExpr(RewriteRule):
#     """
#     This simplifies MathExpr objects by evaluating all constants in the AST.

#     Args:
#         model (MathExpr): The rule only acts on [`MathExpr`][oqd_core.interface.analog.MathExpr] objects.

#     Returns:
#         model (MathExpr):

#     Assumptions:
#         None

#     """

#     def map_MathNum(self, model):
#         # This empty function overrides the map_MathExpr definition for child class MathNum
#         pass

#     def map_MathImag(self, model):
#         # This empty function overrides the map_MathExpr definition for child class MathImag
#         pass

#     def map_MathExpr(self, model):
#         try:
#             if not _is_constant_math(model):
#                 raise ValidationError.from_exception_data("ConstantMathExpr", [])

#             value = Post(EvaluateMathExpr())(model)

#             if isinstance(value, (int, float)):
#                 return MathNum(value=value)
#             elif value == 1j:
#                 return MathImag()
#             elif value.real == 0:
#                 return MathImag() * MathNum(value=value.imag)
#             else:
#                 return MathNum(value=value.real) + MathImag() * MathNum(
#                     value=value.imag
#                 )

#         except (ValidationError, TypeError, ValueError):
#             return model


# ########################################################################################

# evaluate_math_expr = Post(EvaluateMathExpr())
# """
# Pass for evaluating math expression
# """

# simplify_math_expr = Post(SimplifyMathExpr())
# """
# Pass for simplifying math expression
# """


# canonicalize_math_expr = Chain(
#     FixedPoint(
#         Post(
#             Chain(
#                 PruneMathExpr(),
#                 SimplifyMathExpr(),
#                 DistributeMathExpr(),
#                 ProperOrderMathExpr(),
#             )
#         )
#     ),
#     FixedPoint(Post(PartitionMathExpr())),
#     simplify_math_expr,
# )


# def _is_constant_math(model) -> bool:
#     if isinstance(model, (MathNum, MathImag, Access)):
#         return True
#     if isinstance(model, MathVar):
#         return False
#     if isinstance(model, MathFunc):
#         arg = model.expr
#         if isinstance(arg, list):
#             return all(_is_constant_math(a) for a in arg)
#         return _is_constant_math(arg)
#     if isinstance(model, (MathAdd, MathSub, MathMul, MathDiv, MathPow)):
#         return _is_constant_math(model.expr1) and _is_constant_math(model.expr2)
#     return False
