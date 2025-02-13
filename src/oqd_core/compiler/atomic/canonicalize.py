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

from oqd_compiler_infrastructure import Chain, Pre, RewriteRule

########################################################################################
from oqd_core.interface.atomic import Level, Transition

########################################################################################


class UnrollLevelLabel(RewriteRule):
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


unroll_label_pass = Chain(
    Pre(UnrollLevelLabel()),
    Pre(UnrollTransitionLabel()),
)
