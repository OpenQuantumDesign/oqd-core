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

from functools import partial, reduce

from oqd_compiler_infrastructure import RewriteRule

from oqd_core.atomic import (
    Add,
    Constant,
    Mul,
    Neg,
    Not,
    ParallelProtocol,
    Pos,
    Pow,
    Pulse,
    SerialProtocol,
    Sub,
)
from oqd_core.compiler.atomic.error import AtomicCompilerError
from oqd_core.interface.atomic import Declaration, IfElse, While

########################################################################################


class ResolveNestedProtocol(RewriteRule):
    """
    Unfolds nested protocols into a standard form with only 2 hierarchy levels, a sequential protocol of parallel protocols.

    Args:
        model: A protocol node (`Pulse`, `ParallelProtocol`, or `SerialProtocol`).

    Returns:
        model: The canonicalized protocol node.

    Assumptions:
        None
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def _get_continuous_duration(cls, model):
        if isinstance(model, ParallelProtocol):
            if not model.pulses:
                raise AtomicCompilerError("Parallel block is empty.")
            if len(model.pulses) == 1:
                return cls._get_continuous_duration(model.pulses[0])

            return min(map(cls._get_continuous_duration, model.pulses))

        if isinstance(model, SerialProtocol):
            if not model.pulses:
                raise AtomicCompilerError("Serial block is empty.")
            return cls._get_continuous_duration(model.pulses[0])

        return model.duration

    @classmethod
    def _cut_protocol(cls, model, continuous_duration):
        if isinstance(model, ParallelProtocol):
            pairs = list(
                map(
                    partial(cls._cut_protocol, continuous_duration=continuous_duration),
                    model.pulses,
                )
            )

            cut = reduce(lambda x, y: x + y, map(lambda x: x[0], pairs))

            remainder = [r for r in map(lambda x: x[1], pairs) if r is not None]

            if remainder:
                return cut, ParallelProtocol(pulses=remainder)

            return cut, None

        if isinstance(model, SerialProtocol):
            cut, remainder = cls._cut_protocol(
                model.pulses[0], continuous_duration=continuous_duration
            )

            if remainder:
                return cut, SerialProtocol(pulses=[remainder, *model.pulses[1:]])
            if model.pulses[1:]:
                return cut, SerialProtocol(pulses=model.pulses[1:])

            return cut, None

        total = model.duration
        cut = model.model_copy(deep=True)

        if total == continuous_duration:
            return [cut], None
        cut.duration = Constant(value=continuous_duration)
        remainder = model.model_copy(deep=True)
        remainder.duration = Sub(
            expr1=model.duration,
            expr2=Constant(value=continuous_duration),
        )

        return [cut], remainder

    def map_ParallelProtocol(self, model):
        statements = model.pulses

        protocols = []
        while statements:
            continuous_duration = min(map(self._get_continuous_duration, statements))

            pairs = list(
                map(
                    partial(
                        self._cut_protocol, continuous_duration=continuous_duration
                    ),
                    statements,
                )
            )

            protocols.append(
                ParallelProtocol(
                    pulses=reduce(lambda x, y: x + y, map(lambda x: x[0], pairs))
                )
            )

            statements = [r for r in map(lambda x: x[1], pairs) if r is not None]

        return SerialProtocol(pulses=protocols)

    def map_SerialProtocol(self, model):
        if len(model.pulses) == 1:
            return model.pulses[0]

        new_statements = []
        for subprotocol in model.pulses:
            if isinstance(subprotocol, SerialProtocol):
                new_statements.extend(
                    list(
                        map(
                            lambda x: (
                                x
                                if isinstance(x, ParallelProtocol)
                                else ParallelProtocol(pulses=[x])
                            ),
                            subprotocol.pulses,
                        )
                    )
                )
            elif isinstance(subprotocol, ParallelProtocol):
                new_statements.append(subprotocol)
            else:
                new_statements.append(ParallelProtocol(pulses=[subprotocol]))
        return model.__class__(pulses=new_statements)

    def map_Pulse(self, model):
        return SerialProtocol(pulses=[model])

    def map_Declaration(self, model: Declaration):
        pass

    def map_IfElse(self, model: IfElse):
        pass

    def map_While(self, model: While):
        pass


class ResolveRelativeTime(RewriteRule):
    """
    Handles conversion of relative time to absolute time.

    Args:
        model: A protocol node (`Pulse`, `ParallelProtocol`, or `SerialProtocol`).

    Returns:
        model: The canonicalized protocol node.

    Assumptions:
        None
    """

    def __init__(self):
        super().__init__()

    def _get_segment_duration(self, model):
        if isinstance(model, SerialProtocol):
            if not model.pulses:
                raise AtomicCompilerError("Serial block is empty.")
            return sum(self._get_segment_duration(p) for p in model.pulses)

        if isinstance(model, ParallelProtocol):
            if not model.pulses:
                raise AtomicCompilerError("Parallel block is empty.")
            return max(self._get_segment_duration(p) for p in model.pulses)

        if isinstance(model, Pulse):
            return model.duration

        return 0


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


