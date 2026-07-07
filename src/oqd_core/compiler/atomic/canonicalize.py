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

from oqd_compiler_infrastructure import Chain, Post, RewriteRule
from oqd_core.interface.atomic.expr import MathNum, MathSub, MathVar, Pulse
from oqd_core.interface.atomic import Declaration, IfElse, While
from oqd_core.compiler.atomic.math.passes import simplify_math_expr
from oqd_core.compiler.atomic.math.rules import SubstituteMathVar
from oqd_core.compiler.atomic.error import AtomicCompilerError
from oqd_core.interface.atomic.statement import SerialProtocol, ParallelProtocol


########################################################################################

PROTOCOL_STMT_TYPES = (Pulse, ParallelProtocol, SerialProtocol)

def _as_numeric_duration(duration):
    simplified = simplify_math_expr(duration)
    if isinstance(simplified, MathNum):
        return simplified.value
    raise AtomicCompilerError(f"Duration must be constant: {duration}")

class ResolveNestedProtocol(RewriteRule):
    """
    Unfolds nested protocols into a standard form with only 2 hierarchy levels, a sequential protocol of parallel protocols.

    Args:
        model (AtomicCircuit): The rule only acts on [`AtomicCircuit`][oqd_core.interface.atomic.AtomicCircuit] objects.

    Returns:
        model (AtomicCircuit):

    Assumptions:
        None
    """

    def __init__(self):
        super().__init__()

        self.durations = []

    @classmethod
    def _get_continuous_duration(self, model):
        if isinstance(model, ParallelProtocol):
            if not model.pulses:
                raise AtomicCompilerError(f"Parallel block is empty.")
            if len(model.pulses) == 1:
                return self._get_continuous_duration(model.pulses[0])

            return min(map(self._get_continuous_duration, model.pulses))

        if isinstance(model, SerialProtocol):
            if not model.pulses:
                raise AtomicCompilerError(f"Serial block is empty.")
            return self._get_continuous_duration(model.pulses[0])

        return _as_numeric_duration(model.duration)

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
                return cut, SerialProtocol(
                    pulses=[remainder, *model.pulses[1:]]
                )
            if model.pulses[1:]:
                return cut, SerialProtocol(pulses=model.pulses[1:])

            return cut, None

        total = _as_numeric_duration(model.duration)
        cut = model.model_copy(deep=True)
        
        if total == continuous_duration:
            return [cut], None
        cut.duration = MathNum(value=continuous_duration)
        remainder = model.model_copy(deep=True)
        remainder.duration = MathSub(
            expr1=model.duration,
            expr2=MathNum(value=continuous_duration),
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
                            lambda x: x
                            if isinstance(x, ParallelProtocol)
                            else ParallelProtocol(pulses=[x]),
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
        model (AtomicCircuit): The rule only acts on [`AtomicCircuit`][oqd_core.interface.atomic.AtomicCircuit] objects.

    Returns:
        model (AtomicCircuit):

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
            return _as_numeric_duration(model.duration)
        
        return 0
    
    def _substitute_at(self, offset):
        return SubstituteMathVar(
            variable=MathVar(name="#s"),
            substitution=MathSub(
                expr1=MathVar(name="#t"),
                expr2=MathNum(value=offset),
            ),
        )

    def map_SerialProtocol(self, model):
        current_time = 0

        new_pulses = []
        for p in model.pulses:
            new_pulses.append(Post(self._substitute_at(current_time))(p))
            current_time += self._get_segment_duration(p)

        return SerialProtocol(pulses=new_pulses)
    
    def map_ParallelProtocol(self, model):
        return ParallelProtocol(
            pulses=[Post(self._substitute_at(0))(p) for p in model.pulses]
        )
    
    def map_Pulse(self, model):
        return Post(self._substitute_at(0))(model)
    
    def map_Declaration(self, model: Declaration):
        pass
    
    def map_IfElse(self, model: IfElse):
        pass
    
    def map_While(self, model: While):
        pass


########################################################################################



def canonicalize_atomic_circuit_factory():
    """
    Factory for creating a pass for canonicalizing an atomic circuit.
    """
    return Chain(
        Post(ResolveNestedProtocol()),
        Post(ResolveRelativeTime()),
    )
