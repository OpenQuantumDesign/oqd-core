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

from abc import ABC, abstractmethod

import numpy as np
from oqd_compiler_infrastructure import Post, Pre, PrettyPrint, RewriteRule
from scipy.constants import physical_constants

from oqd_core.interface.atomic.system import Ion, Level, Transition

########################################################################################


class ZeemanShift(RewriteRule):
    def __init__(self, magnetic_field):
        super().__init__()

        self.magnetic_field = magnetic_field

    @staticmethod
    def _angular_momentum(x):
        return x * (x + 1)

    def _Lande_g(self, level):
        gL = 1
        gS = 2

        S = level.spin
        L = level.orbital
        J = level.spin_orbital

        gJ = (
            gL
            * (
                ZeemanShift._angular_momentum(J)
                - ZeemanShift._angular_momentum(S)
                + ZeemanShift._angular_momentum(L)
            )
            + gS
            * (
                ZeemanShift._angular_momentum(J)
                + ZeemanShift._angular_momentum(S)
                - ZeemanShift._angular_momentum(L)
            )
        ) / (2 * ZeemanShift._angular_momentum(J))

        return gS, gL, gJ

    def map_Level(self, model):
        zeeman = (
            physical_constants["Bohr magneton"][0]
            * self.magnetic_field
            * model.spin_orbital_nuclear_magnetization
            * self._Lande_g(model)[-1]
            / physical_constants["reduced Planck constant"][0]
        )

        level = model.model_copy()
        level.energy = level.energy + zeeman
        return level


########################################################################################


class IonBuilder(ABC):
    def build(
        self,
        levels=None,
        magnetic_field=1e-4,
        *,
        excluded_transitions=[],
        position=[0, 0, 0],
    ):
        if levels is None:
            _levels = self._levels
        else:
            _levels = list(filter(lambda x: x.label in levels, self._levels))

        _level_labels = list(map(lambda x: x.label, _levels))
        _transitions = list(
            filter(
                lambda x: x.label not in excluded_transitions
                and x.level1 in _level_labels
                and x.level2 in _level_labels,
                self._transitions,
            )
        )

        ion = Ion(
            mass=self._mass,
            charge=self._charge,
            levels=_levels,
            transitions=_transitions,
            position=position,
        )

        ion = Pre(ZeemanShift(magnetic_field=magnetic_field))(ion)

        return ion

    @property
    @abstractmethod
    def _levels(self):
        pass

    @property
    @abstractmethod
    def _transitions(self):
        pass

    @property
    @abstractmethod
    def _mass(self):
        pass

    @property
    @abstractmethod
    def _charge(self):
        pass

    @property
    def _level_labels(self):
        return list(map(lambda x: x.label, self._levels))

    @property
    def _transition_labels(self):
        return list(map(lambda x: x.label, self._transitions))

    def summary(self, *, verbose=False):
        printer = Post(PrettyPrint())

        s = "{:=^80}\n".format(" Yb171+ Ion ")

        s += "{:-^80}\n".format(" Levels ")

        if verbose:
            s += printer(self._levels)
        else:
            s += printer(self._level_labels)

        s += "\n{:-^80}\n".format(" Transitions ")

        if verbose:
            s += printer(self._transitions)
        else:
            s += printer(self._transition_labels)

        print(s)


########################################################################################


class Yb171IIBuilder(IonBuilder):
    @property
    def _mass(self):
        return 170.936331515

    @property
    def _charge(self):
        return 1

    @property
    def _levels(self):
        qubit = 2 * np.pi * 12.6428 * 1e9
        laser = 2 * np.pi * (811.2888 * 1e12 + 210 * 1e6)
        pump = 2 * np.pi * 2.106 * 1e9
        return [
            Level(
                principal=6,
                spin=1 / 2,
                orbital=0,
                nuclear=1 / 2,
                spin_orbital=1 / 2,
                spin_orbital_nuclear=0,
                spin_orbital_nuclear_magnetization=0,
                energy=0,
                label="q0",
            ),
            Level(
                principal=6,
                spin=1 / 2,
                orbital=0,
                nuclear=1 / 2,
                spin_orbital=1 / 2,
                spin_orbital_nuclear=1,
                spin_orbital_nuclear_magnetization=0,
                energy=qubit,
                label="q1",
            ),
            Level(
                principal=6,
                spin=1 / 2,
                orbital=0,
                nuclear=1 / 2,
                spin_orbital=1 / 2,
                spin_orbital_nuclear=1,
                spin_orbital_nuclear_magnetization=1,
                energy=qubit,
                label="zp",
            ),
            Level(
                principal=6,
                spin=1 / 2,
                orbital=0,
                nuclear=1 / 2,
                spin_orbital=1 / 2,
                spin_orbital_nuclear=1,
                spin_orbital_nuclear_magnetization=-1,
                energy=qubit,
                label="zm",
            ),
            Level(
                principal=6,
                spin=1 / 2,
                orbital=1,
                nuclear=1 / 2,
                spin_orbital=1 / 2,
                spin_orbital_nuclear=0,
                spin_orbital_nuclear_magnetization=0,
                energy=qubit + laser,
                label="e0",
            ),
            Level(
                principal=6,
                spin=1 / 2,
                orbital=1,
                nuclear=1 / 2,
                spin_orbital=1 / 2,
                spin_orbital_nuclear=1,
                spin_orbital_nuclear_magnetization=-1,
                energy=qubit + laser + pump,
                label="e1m",
            ),
            Level(
                principal=6,
                spin=1 / 2,
                orbital=1,
                nuclear=1 / 2,
                spin_orbital=1 / 2,
                spin_orbital_nuclear=1,
                spin_orbital_nuclear_magnetization=0,
                energy=qubit + laser + pump,
                label="e10",
            ),
            Level(
                principal=6,
                spin=1 / 2,
                orbital=1,
                nuclear=1 / 2,
                spin_orbital=1 / 2,
                spin_orbital_nuclear=1,
                spin_orbital_nuclear_magnetization=1,
                energy=qubit + laser + pump,
                label="e1p",
            ),
        ]

    @property
    def _transitions(self):
        return [
            Transition(
                level1="q0",
                level2="q1",
                einsteinA=2 * np.pi / (60 * 60),
                multipole="M1",
                label="q0->q1",
            ),
            Transition(
                level1="q1",
                level2="e0",
                einsteinA=1 / (3 * 8.12 * 1e-9),
                multipole="E1",
                label="q1->e0",
            ),
            Transition(
                level1="zp",
                level2="e0",
                einsteinA=1 / (3 * 8.12 * 1e-9),
                multipole="E1",
                label="zp->e0",
            ),
            Transition(
                level1="zm",
                level2="e0",
                einsteinA=1 / (3 * 8.12 * 1e-9),
                multipole="E1",
                label="zm->e0",
            ),
            Transition(
                level1="q0",
                level2="e10",
                einsteinA=1 / (3 * 8.12 * 1e-9),
                multipole="E1",
                label="q0->e10",
            ),
            Transition(
                level1="zp",
                level2="e10",
                einsteinA=1 / (3 * 8.12 * 1e-9),
                multipole="E1",
                label="zp->e10",
            ),
            Transition(
                level1="zm",
                level2="e10",
                einsteinA=1 / (3 * 8.12 * 1e-9),
                multipole="E1",
                label="zm->e10",
            ),
            Transition(
                level1="q0",
                level2="e1m",
                einsteinA=1 / (3 * 8.12 * 1e-9),
                multipole="E1",
                label="q0->e1m",
            ),
            Transition(
                level1="q1",
                level2="e1m",
                einsteinA=1 / (3 * 8.12 * 1e-9),
                multipole="E1",
                label="q1->e1m",
            ),
            Transition(
                level1="zm",
                level2="e1m",
                einsteinA=1 / (3 * 8.12 * 1e-9),
                multipole="E1",
                label="zm->e1m",
            ),
            Transition(
                level1="q0",
                level2="e1p",
                einsteinA=1 / (3 * 8.12 * 1e-9),
                multipole="E1",
                label="q0->e1p",
            ),
            Transition(
                level1="q1",
                level2="e1p",
                einsteinA=1 / (3 * 8.12 * 1e-9),
                multipole="E1",
                label="q1->e1p",
            ),
            Transition(
                level1="zp",
                level2="e1p",
                einsteinA=1 / (3 * 8.12 * 1e-9),
                multipole="E1",
                label="zp->e1p",
            ),
        ]


########################################################################################


class Ba133IIBuilder(IonBuilder):
    @property
    def _mass(self):
        return 132.9060074

    @property
    def _charge(self):
        return 1

    @property
    def _levels(self):
        qubit = -2 * np.pi * 9.9254535544 * 1e9
        laser = 2 * np.pi * (607.605 * 1e12 + 1.3800825 * 1e9)
        pump = -2 * np.pi * 1.84011 * 1e9
        return [
            Level(
                principal=6,
                spin=1 / 2,
                orbital=0,
                nuclear=1 / 2,
                spin_orbital=1 / 2,
                spin_orbital_nuclear=0,
                spin_orbital_nuclear_magnetization=0,
                energy=0,
                label="q0",
            ),
            Level(
                principal=6,
                spin=1 / 2,
                orbital=0,
                nuclear=1 / 2,
                spin_orbital=1 / 2,
                spin_orbital_nuclear=1,
                spin_orbital_nuclear_magnetization=0,
                energy=qubit,
                label="q1",
            ),
            Level(
                principal=6,
                spin=1 / 2,
                orbital=0,
                nuclear=1 / 2,
                spin_orbital=1 / 2,
                spin_orbital_nuclear=1,
                spin_orbital_nuclear_magnetization=1,
                energy=qubit,
                label="zp",
            ),
            Level(
                principal=6,
                spin=1 / 2,
                orbital=0,
                nuclear=1 / 2,
                spin_orbital=1 / 2,
                spin_orbital_nuclear=1,
                spin_orbital_nuclear_magnetization=-1,
                energy=qubit,
                label="zm",
            ),
            Level(
                principal=6,
                spin=1 / 2,
                orbital=1,
                nuclear=1 / 2,
                spin_orbital=1 / 2,
                spin_orbital_nuclear=0,
                spin_orbital_nuclear_magnetization=0,
                energy=qubit + laser,
                label="e0",
            ),
            Level(
                principal=6,
                spin=1 / 2,
                orbital=1,
                nuclear=1 / 2,
                spin_orbital=1 / 2,
                spin_orbital_nuclear=1,
                spin_orbital_nuclear_magnetization=-1,
                energy=qubit + laser + pump,
                label="e1m",
            ),
            Level(
                principal=6,
                spin=1 / 2,
                orbital=1,
                nuclear=1 / 2,
                spin_orbital=1 / 2,
                spin_orbital_nuclear=1,
                spin_orbital_nuclear_magnetization=0,
                energy=qubit + laser + pump,
                label="e10",
            ),
            Level(
                principal=6,
                spin=1 / 2,
                orbital=1,
                nuclear=1 / 2,
                spin_orbital=1 / 2,
                spin_orbital_nuclear=1,
                spin_orbital_nuclear_magnetization=1,
                energy=qubit + laser + pump,
                label="e1p",
            ),
        ]

    @property
    def _transitions(self):
        return [
            Transition(
                level1="q0",
                level2="q1",
                einsteinA=2 * np.pi / (60 * 60),
                multipole="M1",
                label="q0->q1",
            ),
            Transition(
                level1="q1",
                level2="e0",
                einsteinA=1 / (3 * 7.9 * 1e-9),
                multipole="E1",
                label="q1->e0",
            ),
            Transition(
                level1="zp",
                level2="e0",
                einsteinA=1 / (3 * 7.9 * 1e-9),
                multipole="E1",
                label="zp->e0",
            ),
            Transition(
                level1="zm",
                level2="e0",
                einsteinA=1 / (3 * 7.9 * 1e-9),
                multipole="E1",
                label="zm->e0",
            ),
            Transition(
                level1="q0",
                level2="e10",
                einsteinA=1 / (3 * 7.9 * 1e-9),
                multipole="E1",
                label="q0->e10",
            ),
            Transition(
                level1="zp",
                level2="e10",
                einsteinA=1 / (3 * 7.9 * 1e-9),
                multipole="E1",
                label="zp->e10",
            ),
            Transition(
                level1="zm",
                level2="e10",
                einsteinA=1 / (3 * 7.9 * 1e-9),
                multipole="E1",
                label="zm->e10",
            ),
            Transition(
                level1="q0",
                level2="e1m",
                einsteinA=1 / (3 * 7.9 * 1e-9),
                multipole="E1",
                label="q0->e1m",
            ),
            Transition(
                level1="q1",
                level2="e1m",
                einsteinA=1 / (3 * 7.9 * 1e-9),
                multipole="E1",
                label="q1->e1m",
            ),
            Transition(
                level1="zm",
                level2="e1m",
                einsteinA=1 / (3 * 7.9 * 1e-9),
                multipole="E1",
                label="zm->e1m",
            ),
            Transition(
                level1="q0",
                level2="e1p",
                einsteinA=1 / (3 * 7.9 * 1e-9),
                multipole="E1",
                label="q0->e1p",
            ),
            Transition(
                level1="q1",
                level2="e1p",
                einsteinA=1 / (3 * 7.9 * 1e-9),
                multipole="E1",
                label="q1->e1p",
            ),
            Transition(
                level1="zp",
                level2="e1p",
                einsteinA=1 / (3 * 7.9 * 1e-9),
                multipole="E1",
                label="zp->e1p",
            ),
        ]
