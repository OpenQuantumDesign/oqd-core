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

import numpy as np
import pytest

from oqd_core.interface.atomic import (
    Beam,
    MathVar,
    AtomicList,
    MathFunc,
    MathMul,
    MathImag,
    MathNum,
)

########################################################################################

T = MathVar(name="#t")
A = MathVar(name="#A")
DEFAULT_WV = AtomicList(values=[1, 0, 0])
DEFAULT_POL = AtomicList(values=[0, 0, 1])

class TestBeam:
    def test_zero_beam(self):
        Beam(
            frequency=0,
            rabi=0,
            phase=0,
            polarization=DEFAULT_POL,
            wavevector=DEFAULT_WV,
        )

    def test_constant_beam(self):
        Beam(
            frequency=0,
            rabi=1,
            phase=np.pi,
            polarization=DEFAULT_POL,
            wavevector=DEFAULT_WV,
        )

    @pytest.mark.parametrize(
        "rabi",
        [
            T,
            MathFunc(func="sin", expr=T),
            MathMul(expr1=A, expr2=MathFunc(func="sin", expr=T)),
        ],
    )
    def test_time_dependent_rabi(self, rabi):
        Beam(
            frequency=0,
            rabi=rabi,
            phase=0,
            polarization=DEFAULT_POL,
            wavevector=DEFAULT_WV,
        )

    @pytest.mark.parametrize(
        "phase",
        [
            T,
            MathFunc(func="sin", expr=T),
            MathMul(expr1=A, expr2=MathFunc(func="sin", expr=T)),
        ],
    )
    def test_time_dependent_phase(self, phase):
        Beam(
            frequency=0,
            rabi=1,
            phase=phase,
            polarization=DEFAULT_POL,
            wavevector=DEFAULT_WV,
        )

    @pytest.mark.parametrize(
        "polarization",
        [
            AtomicList(values=[1, 0, 0]),
            AtomicList(values=[0, 1, 0]),
            AtomicList(values=[0, 0, 1]),
            AtomicList(values=[1, 1, 0]),
            AtomicList(values=[1, 1j, 0]),
            AtomicList(values=[1, -1j, 0]),
            AtomicList(values=[1, -1j, 0]),
            AtomicList(values=[0, 0, MathFunc(func="exp", expr=MathMul(expr1=MathImag(), expr2=MathNum(value=3.1415926535)),),]),
        ],
    )
    def test_polarization(self, polarization):
        Beam(
            frequency=0,
            rabi=1,
            phase=0,
            polarization=polarization,
            wavevector=DEFAULT_WV,
        )

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        "polarization",
        [
            AtomicList(values=[0, 0, T]),
            AtomicList(values=[0, 0, MathFunc(func="sin", expr=T)]),
            AtomicList(values=[0, 0, MathMul(expr1=A, expr2=MathFunc(func="sin", expr=T))]),
        ],
    )
    def test_non_constant_polarization(self, polarization):
        Beam(
            frequency=0,
            rabi=1,
            phase=0,
            polarization=polarization,
            wavevector=DEFAULT_WV,
        )

    @pytest.mark.parametrize(
        "wavevector",
        [
            AtomicList(values=[1, 0, 0]),
            AtomicList(values=[0, 1, 0]),
            AtomicList(values=[0, 0, 1]),
            AtomicList(values=[1, 1, 0]),
            AtomicList(values=[0, 0, MathFunc(func="exp", expr=MathMul(expr1=MathImag(), expr2=MathNum(value=3.1415926535)),),]),
        ],
    )
    def test_wavevector(self, wavevector):
        Beam(
            frequency=0,
            rabi=1,
            phase=0,
            polarization=DEFAULT_POL,
            wavevector=wavevector,
        )

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        "wavevector",
        [
            AtomicList(values=[0, 0, T]),
            AtomicList(values=[0, 0, MathFunc(func="sin", expr=T)]),
            AtomicList(values=[0, 0, MathMul(expr1=A, expr2=MathFunc(func="sin", expr=T))]),
        ],
    )
    def test_non_constant_wavevector(self, wavevector):
        Beam(
            frequency=0,
            rabi=1,
            phase=0,
            polarization=DEFAULT_POL,
            wavevector=wavevector,
        )
