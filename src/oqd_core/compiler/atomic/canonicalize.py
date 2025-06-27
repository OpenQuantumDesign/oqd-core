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

from oqd_compiler_infrastructure import Chain, Post, Pre, RewriteRule

from oqd_core.compiler.math.rules import SubstituteMathVar
from oqd_core.interface.atomic import Level, Transition
from oqd_core.interface.atomic.protocol import ParallelProtocol, SequentialProtocol
from oqd_core.interface.math import MathVar

########################################################################################


class UnrollLevelLabel(RewriteRule):
    """
    Unrolls the [`Level`][oqd_core.interface.atomic.system.Level] labels present in [`Transitions`][oqd_core.interface.atomic.system.Transition].

    Args:
        model (AtomicCircuit): The rule only acts on [`AtomicCircuit`][oqd_core.interface.atomic.AtomicCircuit] objects.

    Returns:
        model (AtomicCircuit):

    Assumptions:
        None

    """

    def map_Ion(self, model):
        self.ion_levels = {level.label: level for level in model.levels}

    def map_Transition(self, model):
        if isinstance(model.level1, Level) and isinstance(model.level2, Level):
            return

        level1 = (
            self.ion_levels[model.level1]
            if isinstance(model.level1, str)
            else model.level1
        )
        level2 = (
            self.ion_levels[model.level2]
            if isinstance(model.level2, str)
            else model.level2
        )
        return model.__class__(
            label=model.label,
            level1=level1,
            level2=level2,
            einsteinA=model.einsteinA,
            multipole=model.multipole,
        )


class UnrollTransitionLabel(RewriteRule):
    """
    Unrolls the [`Transition`][oqd_core.interface.atomic.system.Transition] labels present in [`Beams`][oqd_core.interface.atomic.protocol.Beam].

    Args:
        model (AtomicCircuit): The rule only acts on [`AtomicCircuit`][oqd_core.interface.atomic.AtomicCircuit] objects.

    Returns:
        model (AtomicCircuit):

    Assumptions:
        None
    """

    def map_System(self, model):
        self.ions_transitions = [
            {transition.label: transition for transition in ion.transitions}
            for ion in model.ions
        ]

    def map_Beam(self, model):
        if isinstance(model.transition, Transition):
            return

        if isinstance(model.transition, str):
            transition_label = model.transition
            reference_ion = model.target
        else:
            transition_label = model.transition[0]
            reference_ion = model.transition[1]

        transition = self.ions_transitions[reference_ion][transition_label]
        return model.__class__(
            transition=transition,
            rabi=model.rabi,
            detuning=model.detuning,
            phase=model.phase,
            polarization=model.polarization,
            wavevector=model.wavevector,
            target=model.target,
        )


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
            if len(model.sequence) == 1:
                return model.sequence[0].duration

            return min(map(lambda x: x.duration, model.sequence))

        if isinstance(model, SequentialProtocol):
            return self._get_continuous_duration(model.sequence[0])

        return model.duration

    @classmethod
    def _cut_protocol(cls, model, continuous_duration):
        if isinstance(model, ParallelProtocol):
            pairs = list(
                map(
                    partial(cls._cut_protocol, continuous_duration=continuous_duration),
                    model.sequence,
                )
            )

            cut = reduce(lambda x, y: x + y, map(lambda x: x[0], pairs))

            remainder = [r for r in map(lambda x: x[1], pairs) if r is not None]

            if remainder:
                return cut, ParallelProtocol(sequence=remainder)

            return cut, None

        if isinstance(model, SequentialProtocol):
            cut, remainder = cls._cut_protocol(
                model.sequence[0], continuous_duration=continuous_duration
            )

            if remainder:
                return cut, SequentialProtocol(
                    sequence=[remainder, *model.sequence[1:]]
                )
            if model.sequence[1:]:
                return cut, SequentialProtocol(sequence=model.sequence[1:])

            return cut, None

        cut = model.model_copy(deep=True)
        if cut.duration == continuous_duration:
            return [cut], None
        cut.duration = continuous_duration

        remainder = model.model_copy(deep=True)
        remainder.duration = remainder.duration - continuous_duration

        return [cut], remainder

    def map_ParallelProtocol(self, model):
        sequence = model.sequence

        protocols = []
        while sequence:
            continuous_duration = min(map(self._get_continuous_duration, sequence))

            pairs = list(
                map(
                    partial(
                        self._cut_protocol, continuous_duration=continuous_duration
                    ),
                    sequence,
                )
            )

            protocols.append(
                ParallelProtocol(
                    sequence=reduce(lambda x, y: x + y, map(lambda x: x[0], pairs))
                )
            )

            sequence = [r for r in map(lambda x: x[1], pairs) if r is not None]

        return SequentialProtocol(sequence=protocols)

    def map_SequentialProtocol(self, model):
        if len(model.sequence) == 1:
            return model.sequence[0]

        new_sequence = []
        for subprotocol in model.sequence:
            if isinstance(subprotocol, SequentialProtocol):
                new_sequence.extend(
                    list(
                        map(
                            lambda x: x
                            if isinstance(x, ParallelProtocol)
                            else ParallelProtocol(sequence=[x]),
                            subprotocol.sequence,
                        )
                    )
                )
            elif isinstance(subprotocol, ParallelProtocol):
                new_sequence.append(subprotocol)
            else:
                new_sequence.append(ParallelProtocol(sequence=[subprotocol]))
        return model.__class__(sequence=new_sequence)

    def map_Pulse(self, model):
        return SequentialProtocol(sequence=[model])


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

    def map_AtomicCircuit(self, model):
        protocol = Post(
            SubstituteMathVar(
                variable=MathVar(name="s"), substitution=MathVar(name="t")
            )
        )(model.protocol)

        return model.__class__(system=model.system, protocol=protocol)

    @classmethod
    def _get_duration(cls, model):
        if isinstance(model, SequentialProtocol):
            return reduce(
                lambda x, y: x + y,
                [cls._get_duration(p) for p in model.sequence],
            )
        if isinstance(model, ParallelProtocol):
            return max(
                *[cls._get_duration(p) for p in model.sequence],
            )
        return model.duration

    def map_SequentialProtocol(self, model):
        current_time = 0

        new_sequence = []
        for p in model.sequence:
            duration = self._get_duration(p)

            new_p = Post(
                SubstituteMathVar(
                    variable=MathVar(name="s"),
                    substitution=MathVar(name="s") - current_time,
                )
            )(p)
            new_sequence.append(new_p)

            current_time += duration

        return model.__class__(sequence=new_sequence)


########################################################################################

unroll_label_pass = Chain(
    Pre(UnrollLevelLabel()),
    Pre(UnrollTransitionLabel()),
)
"""
Pass that unrolls the references to levels and transitions
"""


def canonicalize_atomic_circuit_factory():
    """
    Factory for creating a pass for canonicalizing an atomic circuit.
    """
    return Chain(
        unroll_label_pass,
        Post(ResolveRelativeTime()),
        Post(ResolveNestedProtocol()),
    )
