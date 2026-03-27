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
from oqd_compiler_infrastructure import Post, Pre, PrettyPrint, RewriteRule, TypeReflectBaseModel
from scipy.constants import physical_constants
from typing import Union, List, Annotated, Literal
from pydantic import (
    AfterValidator,
    NonNegativeFloat,
    NonNegativeInt,
)
########################################################################################

def is_halfint(v: float) -> bool:
    """
    Function that verifies a number is an integer or half-integer.

    Args:
        v: Number to verify.
    """
    if not (v * 2).is_integer():
        raise ValueError()
    return v

AngularMomentumNumber = Annotated[float, AfterValidator(is_halfint)]
"""
A valid positive or negative integer or half-integer for angular momentum.
"""
NonNegativeAngularMomentumNumber = Annotated[
    NonNegativeFloat, AfterValidator(is_halfint)
]
"""
A valid non-negative integer or half-integer for angular momentum.
"""


class Level(TypeReflectBaseModel):
    """ "
    Class representing an electronic energy level of an ion.

    Attributes:
        label: Label for the Level
        principal: Principal quantum number.
        spin: Spin of an electron.
        orbital: Orbital angular momentum of an electron.
        nuclear: Nuclear angular momentum.
        spin_orbital: Angular momentum of the spin-orbital coupling.
        spin_orbital_nuclear: Angular momentum of the spin-orbital-nuclear coupling.
        spin_orbital_nuclear_magnetization: Magnetization of the spin-orbital-nuclear coupled angular momentum.
        energy: Energy of the electronic state.

    """

    label: str
    principal: NonNegativeInt
    spin: NonNegativeAngularMomentumNumber
    orbital: NonNegativeAngularMomentumNumber
    nuclear: NonNegativeAngularMomentumNumber
    spin_orbital: NonNegativeAngularMomentumNumber
    spin_orbital_nuclear: NonNegativeAngularMomentumNumber
    spin_orbital_nuclear_magnetization: AngularMomentumNumber
    energy: float
    
    # @model_validator(mode="after")
    # def orbital_validate(self):
    #     if self.orbital >= self.principal:
    #         raise ValueError("Invalid orbital quantum # (L)")
    #     return self

    # @model_validator(mode="after")
    # def spin_orbital_validate(self):
    #     if (
    #         self.spin_orbital < abs(self.spin - self.orbital)
    #         or self.spin_orbital > self.spin + self.orbital
    #     ):
    #         raise ValueError("Invalid spin orbital quantum # (J)")
    #     return self

    # @model_validator(mode="after")
    # def spin_orbital_nuclear_validate(self):
    #     if (
    #         self.spin_orbital_nuclear < abs(self.spin_orbital - self.nuclear)
    #         or self.spin_orbital_nuclear > self.spin_orbital + self.nuclear
    #     ):
    #         raise ValueError("Invalid spin orbital nuclear quantum # (F)")
    #     return self

    # @model_validator(mode="after")
    # def spin_orbital_nuclear_magnetization_validate(self):
    #     if abs(self.spin_orbital_nuclear_magnetization) > self.spin_orbital_nuclear:
    #         raise ValueError("Invalid spin orbital nuclear magnetization (m_F)")
    #     elif not (
    #         self.spin_orbital_nuclear_magnetization - self.spin_orbital_nuclear
    #     ).is_integer():
    #         raise ValueError("Invalid spin orbital nuclear magnetization (m_F)")
    #     return self
    

class Transition(TypeReflectBaseModel):
    """
    Class representing a transition between electronic states of an ion.

    Attributes:
        label: Label for the Transition
        level1: Label for energy level 1.
        level2: Label for energy level 2.
        einsteinA: Einstein A coefficient that characterizes the strength of coupling between energy level 1 and 2.

    """
    label: str
    level1: Union[str, Level]
    level2: Union[str, Level]
    einsteinA: float
    multipole: Literal["E1", "E2", "M1"]
    

class Ion(TypeReflectBaseModel):
    """
    Class representing an ion.

    Attributes:
        mass: Mass of the ion.
        charge: Charge of the ion.
        levels: Electronic energy levels of the ion.
        transitions: Allowed transitions in the ion.
        position: Spatial position of the ion.
    """

    mass: float
    charge: float
    levels: List[Level]
    transitions: List[Transition]
    position: List[float]

    @property
    def _level_dict(self):
        return {level.label: level for level in self.levels}

    @property
    def _transition_dict(self):
        return {transition.label: transition for transition in self.transitions}

    def __getitem__(self, label):
        if label in self._level_dict.keys():
            return self._level_dict[label]

        if label in self._transition_dict.keys():
            return self._transition_dict[label]

        raise KeyError("Invalid key, label not in levels or transitions.")


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
