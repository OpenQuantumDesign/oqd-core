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

from typing import List, Optional
from typing_extensions import Annotated
from pydantic import (
    NonNegativeInt,
    AfterValidator,
    NonNegativeFloat,
)

from oqd_compiler_infrastructure import TypeReflectBaseModel

########################################################################################

__all__ = [
    "Level",
    "Transition",
    "Ion",
    "Phonon",
    "System",
]

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


########################################################################################


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

# ########################################################################################


class Level(TypeReflectBaseModel):
    """ "
    Class representing an electronic energy level of an ion.

    Attributes:
        principal: Principal quantum number.
        spin: Spin of an electron.
        orbital: Orbital angular momentum of an electron.
        nuclear: Nuclear angular momentum.
        spin_orbital: Angular momentum of the spin-orbital coupling.
        spin_orbital_nuclear: Angular momentum of the spin-orbital-nuclear coupling.
        spin_orbital_nuclear_magnetization: Magnetization of the spin-orbital-nuclear coupled angular momentum.
        energy: Energy of the electronic state.

    """

    principal: Optional[NonNegativeInt] = None
    spin: Optional[NonNegativeAngularMomentumNumber] = None
    orbital: Optional[NonNegativeAngularMomentumNumber] = None
    nuclear: Optional[NonNegativeAngularMomentumNumber] = None
    spin_orbital: Optional[NonNegativeAngularMomentumNumber] = None
    spin_orbital_nuclear: Optional[NonNegativeAngularMomentumNumber] = None
    spin_orbital_nuclear_magnetization: Optional[AngularMomentumNumber] = None
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
        level1: Energy level 1.
        level2: Energy level 2.
        einsteinA: Einstein A coefficient that characterizes the strength of coupling between energy level 1 and 2.

    """

    level1: Level
    level2: Level
    einsteinA: float


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


########################################################################################


class Phonon(TypeReflectBaseModel):
    """
    Class representing a collective phonon mode of the trapped-ion system.

    Attributes:
        energy: Quanta of energy for the phonon mode.
        eigenvector: Profile of the phonon mode in terms of the vibration of the ions.

    """

    energy: float
    eigenvector: List[float]


########################################################################################


class System(TypeReflectBaseModel):
    """
    Class representing a trapped-ion system.

    Attributes:
        ions: List of ions in the trapped-ion system.
        modes: List of collective phonon modes for the trapped-ion system.

    """

    ions: List[Ion]
    modes: List[Phonon]
