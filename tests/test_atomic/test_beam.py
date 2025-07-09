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
from oqd_compiler_infrastructure import Post

from oqd_core.compiler.atomic.verify import VerifyBeam
from oqd_core.interface.atomic import Beam, Level, Transition
from oqd_core.interface.atomic.species import Yb171IIBuilder
from oqd_core.interface.math import MathStr

########################################################################################


@pytest.fixture
def ion():
    return Yb171IIBuilder().build()


class TestBeamDef:
    def test_zero_beam(self, ion):
        Beam(
            transition=ion.transitions[0],
            detuning=0,
            rabi=0,
            phase=0,
            polarization=[0, 0, 1],
            wavevector=[1, 0, 0],
            target=0,
        )

    def test_constant_beam(self, ion):
        Beam(
            transition=ion.transitions[0],
            detuning=0,
            rabi=1,
            phase=np.pi,
            polarization=[0, 0, 1],
            wavevector=[1, 0, 0],
            target=0,
        )

    @pytest.mark.parametrize(
        "rabi",
        [
            MathStr(string="t"),
            MathStr(string="sin(t)"),
            MathStr(string="A*sin(t)"),
        ],
    )
    def test_time_dependent_rabi(self, ion, rabi):
        Beam(
            transition=ion.transitions[0],
            detuning=0,
            rabi=rabi,
            phase=0,
            polarization=[0, 0, 1],
            wavevector=[1, 0, 0],
            target=0,
        )

    @pytest.mark.parametrize(
        "detuning",
        [
            MathStr(string="t"),
            MathStr(string="sin(t)"),
            MathStr(string="A*sin(t)"),
        ],
    )
    def test_time_dependent_detuning(self, ion, detuning):
        Beam(
            transition=ion.transitions[0],
            detuning=detuning,
            rabi=1,
            phase=0,
            polarization=[0, 0, 1],
            wavevector=[1, 0, 0],
            target=0,
        )

    @pytest.mark.parametrize(
        "phase",
        [
            MathStr(string="t"),
            MathStr(string="sin(t)"),
            MathStr(string="A*sin(t)"),
        ],
    )
    def test_time_dependent_phase(self, ion, phase):
        Beam(
            transition=ion.transitions[0],
            detuning=0,
            rabi=1,
            phase=phase,
            polarization=[0, 0, 1],
            wavevector=[1, 0, 0],
            target=0,
        )

    @pytest.mark.parametrize(
        "polarization",
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 1j, 0],
            [1, -1j, 0],
            [1, -1j, 0],
            [0, 0, MathStr(string="exp(1j*3.1415926535)")],
        ],
    )
    def test_polarization(self, ion, polarization):
        Beam(
            transition=ion.transitions[0],
            detuning=0,
            rabi=1,
            phase=0,
            polarization=polarization,
            wavevector=[1, 0, 0],
            target=0,
        )

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        "polarization",
        [
            [0, 0, MathStr(string="t")],
            [0, 0, MathStr(string="sin(t)")],
            [0, 0, MathStr(string="A*sin(t)")],
        ],
    )
    def test_non_constant_polarization(self, ion, polarization):
        Beam(
            transition=ion.transitions[0],
            detuning=0,
            rabi=1,
            phase=0,
            polarization=polarization,
            wavevector=[1, 0, 0],
            target=0,
        )

    @pytest.mark.parametrize(
        "wavevector",
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, MathStr(string="exp(1j*3.1415926535)")],
        ],
    )
    def test_wavevector(self, ion, wavevector):
        Beam(
            transition=ion.transitions[0],
            detuning=0,
            rabi=1,
            phase=0,
            polarization=[0, 0, 1],
            wavevector=wavevector,
            target=0,
        )

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        "wavevector",
        [
            [0, 0, MathStr(string="t")],
            [0, 0, MathStr(string="sin(t)")],
            [0, 0, MathStr(string="A*sin(t)")],
        ],
    )
    def test_non_constant_wavevector(self, ion, wavevector):
        Beam(
            transition=ion.transitions[0],
            detuning=0,
            rabi=1,
            phase=0,
            polarization=[0, 0, 1],
            wavevector=wavevector,
            target=0,
        )


class TestBeamVerify:
    @pytest.fixture
    def verify_pass(self):
        return Post(VerifyBeam())

    @pytest.fixture
    def transition_polarization(self, request):
        level1 = Level(
            label="0",
            principal=1,
            spin=0.5,
            orbital=0,
            nuclear=0.5,
            spin_orbital=0.5,
            spin_orbital_nuclear=0,
            spin_orbital_nuclear_magnetization=0,
            energy=0,
        )

        if request.param[0] == "M1":
            level2 = Level(
                label="1",
                principal=1,
                spin=0.5,
                orbital=0,
                nuclear=0.5,
                spin_orbital=0.5,
                spin_orbital_nuclear=1,
                spin_orbital_nuclear_magnetization=0,
                energy=1,
            )

        elif request.param[0] == "E1":
            level2 = Level(
                label="1",
                principal=1,
                spin=0.5,
                orbital=1,
                nuclear=0.5,
                spin_orbital=0.5,
                spin_orbital_nuclear=1,
                spin_orbital_nuclear_magnetization=0,
                energy=1,
            )

        elif request.param[0] == "E2":
            level2 = Level(
                label="1",
                principal=1,
                spin=0.5,
                orbital=2,
                nuclear=0.5,
                spin_orbital=1.5,
                spin_orbital_nuclear=1,
                spin_orbital_nuclear_magnetization=0,
                energy=1,
            )

        return Transition(
            label="0->1",
            level1=level1,
            level2=level2,
            einsteinA=1,
            multipole=request.param[0],
        ), request.param[1]

    @pytest.mark.parametrize(
        ("transition_polarization"),
        [
            ("M1", [0, 1, 0]),
            ("E1", [0, 0, 1]),
            ("E2", [1, 0, 0]),
        ],
        indirect=True,
    )
    def test_verify_pass(self, transition_polarization, verify_pass):
        transition, polarization = transition_polarization
        beam = Beam(
            transition=transition,
            detuning=0,
            rabi=1,
            phase=0,
            polarization=polarization,
            wavevector=[1, 0, 0],
            target=0,
        )

        verify_pass(beam)
