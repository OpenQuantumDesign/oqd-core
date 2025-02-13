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

from oqd_core.interface.analog import (
    AnalogCircuit,
    AnalogGate,
    Annihilation,
    Creation,
    Identity,
    PauliI,
    PauliX,
    PauliY,
    PauliZ,
)
from oqd_core.interface.atomic import (
    AtomicCircuit,
    Beam,
    Ion,
    Level,
    ParallelProtocol,
    Phonon,
    Pulse,
    SequentialProtocol,
    System,
    Transition,
)

########################################################################################
from oqd_core.interface.math import MathStr

########################################################################################

X, Y, Z, PI, A, C, LI = (
    PauliX(),
    PauliY(),
    PauliZ(),
    PauliI(),
    Annihilation(),
    Creation(),
    Identity(),
)


Hx = AnalogGate(hamiltonian=-(np.pi / 4) * X)
analog_circuit = AnalogCircuit()
analog_circuit.evolve(duration=1, gate=Hx)
analog_circuit.evolve(duration=1, gate=Hx)
analog_circuit.evolve(duration=1, gate=Hx)
analog_circuit.measure()

########################################################################################

downstate = Level(
    label="q0",
    principal=6,
    spin=1 / 2,
    orbital=0,
    nuclear=1 / 2,
    spin_orbital=1 / 2,
    spin_orbital_nuclear=0,
    spin_orbital_nuclear_magnetization=0,
    energy=2 * np.pi * 0,
)
upstate = Level(
    label="q1",
    principal=6,
    spin=1 / 2,
    orbital=0,
    nuclear=1 / 2,
    spin_orbital=1 / 2,
    spin_orbital_nuclear=1,
    spin_orbital_nuclear_magnetization=0,
    energy=2 * np.pi * 10,
)
estate = Level(
    label="e0",
    principal=5,
    spin=1 / 2,
    orbital=1,
    nuclear=1 / 2,
    spin_orbital=1 / 2,
    spin_orbital_nuclear=1,
    spin_orbital_nuclear_magnetization=-1,
    energy=2 * np.pi * 100,
)
estate2 = Level(
    label="e1",
    principal=5,
    spin=1 / 2,
    orbital=1,
    nuclear=1 / 2,
    spin_orbital=1 / 2,
    spin_orbital_nuclear=1,
    spin_orbital_nuclear_magnetization=1,
    energy=2 * np.pi * 110,
)

transitions = [
    Transition(
        label="q0->e0",
        level1="q0",
        level2="e0",
        einsteinA=1,
        multipole="E1",
    ),
    Transition(
        label="q0->e1",
        level1="q0",
        level2="e1",
        einsteinA=1,
        multipole="E1",
    ),
    Transition(
        label="q1->e0",
        level1="q1",
        level2="e0",
        einsteinA=1,
        multipole="E1",
    ),
    Transition(
        label="q1->e1",
        level1="q1",
        level2="e1",
        einsteinA=1,
        multipole="E1",
    ),
]

ion = Ion(
    mass=171,
    charge=1,
    position=[0, 0, 0],
    levels=[downstate, upstate, estate, estate2],
    transitions=transitions,
)

COM_x = Phonon(energy=0.1, eigenvector=[1, 0, 0])

system = System(
    ions=[
        ion,
    ],
    modes=[
        COM_x,
    ],
)

beam = Beam(
    transition="q0->e0",
    rabi=2 * np.pi * 1,
    detuning=0,
    phase=0,
    polarization=[1, 0, 0],
    wavevector=[0, 1, 0],
    target=0,
)

beam2 = Beam(
    transition="q0->e0",
    rabi=2 * np.pi * 5,
    detuning=2 * np.pi * 25,
    phase=0,
    polarization=[1, 0, 0],
    wavevector=[0, 1, 0],
    target=0,
)

beam3 = Beam(
    transition="q1->e0",
    rabi=2 * np.pi * 5,
    detuning=2 * np.pi * 25,
    phase=0,
    polarization=[1, 0, 0],
    wavevector=[0, 1, 0],
    target=0,
)

protocol1 = SequentialProtocol(
    sequence=[
        Pulse(beam=beam, duration=10),
    ]
)
protocol2 = ParallelProtocol(
    sequence=[
        Pulse(beam=beam2, duration=10),
        Pulse(beam=beam3, duration=10),
    ]
)

protocol = SequentialProtocol(
    sequence=[
        protocol1,
        protocol2,
    ]
)

atomic_circuit = AtomicCircuit(system=system, protocol=protocol)

########################################################################################


@pytest.mark.parametrize(
    "model",
    [
        MathStr(string="1"),
        MathStr(string="t"),
        MathStr(string="1+2"),
        MathStr(string="1*2"),
        MathStr(string="1**2"),
        MathStr(string="sin(t)"),
        MathStr(string="exp(t)"),
        X,
        Y @ Y @ X,
        X + Y + Z + PI,
        A + C + LI,
        Hx,
        analog_circuit,
        downstate,
        upstate,
        estate,
        estate2,
        transitions[0],
        transitions[1],
        transitions[2],
        transitions[3],
        ion,
        COM_x,
        system,
        beam,
        beam2,
        beam3,
        Pulse(beam=beam, duration=10),
        Pulse(beam=beam2, duration=10),
        Pulse(beam=beam3, duration=10),
        protocol1,
        protocol2,
        protocol,
        atomic_circuit,
    ],
)
def test_serialize_deserialize(model):
    assert model.model_validate(model.model_dump()) == model
    assert model.model_validate_json(model.model_dump_json()) == model
