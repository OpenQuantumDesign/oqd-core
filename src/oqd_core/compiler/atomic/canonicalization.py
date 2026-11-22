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


from oqd_compiler_infrastructure import RewriteRule

from oqd_core.atomic import (
    Add,
    Constant,
    Mul,
    Neg,
    Not,
    Pos,
    Pow,
)

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


