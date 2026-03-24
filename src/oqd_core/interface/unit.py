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

from typing import Annotated, Union

import numpy as np
from oqd_compiler_infrastructure import TypeReflectBaseModel
from pydantic import BeforeValidator

########################################################################################
from oqd_core.interface.math import CastMathExpr

########################################################################################


class UnitDimensionBase(TypeReflectBaseModel):
    length: Union[int, float] = 0
    time: Union[int, float] = 0
    mass: Union[int, float] = 0
    current: Union[int, float] = 0
    temperature: Union[int, float] = 0
    luminosity: Union[int, float] = 0
    substance: Union[int, float] = 0
    pass

    def __mul__(self, other):
        assert isinstance(other, UnitDimensionBase)

        dimension_dict = {}
        for k in UnitDimensionBase.model_fields.keys():
            if k == "class_":
                continue
            else:
                dimension_dict[k] = self.__dict__[k] + other.__dict__[k]

        return UnitDimensionBase(**dimension_dict)

    def __truediv__(self, other):
        assert isinstance(other, UnitDimensionBase)

        dimension_dict = {}
        for k in UnitDimensionBase.model_fields.keys():
            if k == "class_":
                continue
            else:
                dimension_dict[k] = self.__dict__[k] - other.__dict__[k]

        return UnitDimensionBase(**dimension_dict)

    def __pow__(self, other):
        assert isinstance(other, (int, float))

        dimension_dict = {}
        for k in UnitDimensionBase.model_fields.keys():
            if k == "class_":
                continue
            else:
                dimension_dict[k] = self.__dict__[k] * other

        return UnitDimensionBase(**dimension_dict)


########################################################################################

Dimensionless = UnitDimensionBase()
LengthDimension = UnitDimensionBase(length=1)
TimeDimension = UnitDimensionBase(time=1)
MassDimension = UnitDimensionBase(mass=1)
CurrentDimension = UnitDimensionBase(current=1)
TemperatureDimension = UnitDimensionBase(temperature=1)
LuminosityDimension = UnitDimensionBase(luminosity=1)
SubstanceDimension = UnitDimensionBase(substance=1)

AreaDimension = LengthDimension**2
VolumeDimension = LengthDimension**3
VelocityDimension = LengthDimension / TimeDimension
AccelerationDimension = LengthDimension / TimeDimension**2
ChargeDimension = CurrentDimension * TimeDimension
EnergyDimension = MassDimension * LengthDimension**2 / TimeDimension**2
FrequencyDimension = TimeDimension ** (-1)
VoltageDimension = EnergyDimension / ChargeDimension
PowerDimension = EnergyDimension / TimeDimension
ForceDimension = MassDimension * LengthDimension / TimeDimension**2

########################################################################################


class UnitBase(TypeReflectBaseModel):
    scale: float
    dimension: UnitDimensionBase = Dimensionless
    pass

    @classmethod
    def cast(cls, value):
        if isinstance(value, UnitBase):
            return value
        if isinstance(value, (int, float)):
            return UnitBase(scale=value)
        raise TypeError

    def __mul__(self, other):
        other = UnitBase.cast(other)
        return UnitMul(
            scale=self.scale * other.scale,
            dimension=self.dimension * other.dimension,
            unit1=self,
            unit2=other,
        )

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = UnitBase.cast(other)
        return UnitDiv(
            scale=self.scale / other.scale,
            dimension=self.dimension / other.dimension,
            unit1=self,
            unit2=other,
        )

    def __rtruediv__(self, other):
        other = UnitBase.cast(other)
        return other / self

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        return UnitPow(
            scale=self.scale**other,
            dimension=self.dimension**other,
            unit=self,
            exponent=other,
        )


########################################################################################

CastUnitBase = Annotated[UnitBase, BeforeValidator(UnitBase.cast)]

########################################################################################


class UnitMul(UnitBase):
    unit1: CastUnitBase
    unit2: CastUnitBase


class UnitDiv(UnitBase):
    unit1: CastUnitBase
    unit2: CastUnitBase


class UnitPow(UnitBase):
    unit: CastUnitBase
    exponent: Union[int, float]


########################################################################################

unitless = UnitBase(scale=1.0, dimension=Dimensionless)

deci = UnitBase(scale=1e-1, dimension=Dimensionless)
centi = UnitBase(scale=1e-2, dimension=Dimensionless)
milli = UnitBase(scale=1e-3, dimension=Dimensionless)
micro = UnitBase(scale=1e-6, dimension=Dimensionless)
nano = UnitBase(scale=1e-9, dimension=Dimensionless)
pico = UnitBase(scale=1e-12, dimension=Dimensionless)
femto = UnitBase(scale=1e-15, dimension=Dimensionless)
atto = UnitBase(scale=1e-18, dimension=Dimensionless)
zepto = UnitBase(scale=1e-21, dimension=Dimensionless)
yocto = UnitBase(scale=1e-24, dimension=Dimensionless)
ronto = UnitBase(scale=1e-27, dimension=Dimensionless)
quecto = UnitBase(scale=1e-30, dimension=Dimensionless)

deca = UnitBase(scale=1e1, dimension=Dimensionless)
hecto = UnitBase(scale=1e2, dimension=Dimensionless)
kilo = UnitBase(scale=1e3, dimension=Dimensionless)
mega = UnitBase(scale=1e6, dimension=Dimensionless)
giga = UnitBase(scale=1e9, dimension=Dimensionless)
tera = UnitBase(scale=1e12, dimension=Dimensionless)
peta = UnitBase(scale=1e15, dimension=Dimensionless)
exa = UnitBase(scale=1e18, dimension=Dimensionless)
zetta = UnitBase(scale=1e21, dimension=Dimensionless)
yotta = UnitBase(scale=1e24, dimension=Dimensionless)
ronna = UnitBase(scale=1e27, dimension=Dimensionless)
quetta = UnitBase(scale=1e30, dimension=Dimensionless)

pi = UnitBase(scale=np.pi, dimension=Dimensionless)

########################################################################################s

metre = UnitBase(scale=1.0, dimension=LengthDimension)
astronomicalunit = UnitBase(scale=1.495978707e11, dimension=LengthDimension)

second = UnitBase(scale=1.0, dimension=TimeDimension)
minute = UnitBase(scale=60, dimension=TimeDimension)
hour = UnitBase(scale=3600, dimension=TimeDimension)
day = UnitBase(scale=86400, dimension=TimeDimension)
year = UnitBase(scale=31557600, dimension=TimeDimension)

speedoflight = UnitBase(scale=299792458, dimension=VelocityDimension)

lightyear = speedoflight * year

hertz = UnitBase(scale=1.0, dimension=FrequencyDimension)

########################################################################################

volt = UnitBase(scale=1.0, dimension=VoltageDimension)

gram = UnitBase(scale=1e-3, dimension=MassDimension)
atomicmassunit = UnitBase(scale=1.66053906660e-27, dimension=MassDimension)

joule = UnitBase(scale=1.0, dimension=EnergyDimension)
electronvolt = UnitBase(scale=1.602176634e-19, dimension=EnergyDimension)

watt = UnitBase(scale=1.0, dimension=PowerDimension)

newton = UnitBase(scale=1.0, dimension=ForceDimension)

########################################################################################

ampere = UnitBase(scale=1.0, dimension=CurrentDimension)

coulomb = UnitBase(scale=1.0, dimension=ChargeDimension)
elementarycharge = UnitBase(scale=1.602176634e-19, dimension=ChargeDimension)


########################################################################################

planckconstant = UnitBase(
    scale=6.62607015e-34, dimension=EnergyDimension * TimeDimension
)
reducedplanckconstant = planckconstant / (pi * 2)

boltzmannconstant = UnitBase(
    scale=1.380649e-23, dimension=EnergyDimension / TemperatureDimension
)

avogadroconstant = UnitBase(scale=6.02214076e23, dimension=SubstanceDimension ** (-1))

gravitationalconstant = UnitBase(
    scale=6.67430e-11, dimension=ForceDimension * LengthDimension**2 / MassDimension**2
)


plancklength = (reducedplanckconstant * gravitationalconstant / speedoflight**3) ** 0.5
plancktime = (reducedplanckconstant * gravitationalconstant / speedoflight**5) ** 0.5
planckmass = (reducedplanckconstant * speedoflight / gravitationalconstant) ** 0.5
plancktemperature = (
    reducedplanckconstant
    * speedoflight**5
    / (gravitationalconstant * boltzmannconstant**2)
) ** 0.5

########################################################################################

kelvin = UnitBase(scale=1.0, dimension=TemperatureDimension)

mole = UnitBase(scale=1.0, dimension=SubstanceDimension)

candela = UnitBase(scale=1.0, dimension=LuminosityDimension)

########################################################################################


class UnitfulMathExpr(TypeReflectBaseModel):
    expr: CastMathExpr
    unit: UnitBase = unitless

    def __add__(self, other):
        return UnitfulMathExpr(self.expr + other.expr, self.unit)

    def __sub__(self, other):
        return UnitfulMathExpr(self.expr - other.expr, self.unit)

    def __mul__(self, other):
        return UnitfulMathExpr(self.expr * other.expr, self.unit * other.unit)

    def __truediv__(self, other):
        return UnitfulMathExpr(self.expr / other.expr, self.unit / other.unit)
