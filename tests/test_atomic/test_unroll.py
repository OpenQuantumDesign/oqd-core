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

from oqd_core.compiler.atomic.canonicalize import unroll_label_pass

########################################################################################
from oqd_core.interface.atomic import (
    AtomicCircuit,
    Beam,
    Ion,
    Level,
    Pulse,
    SequentialProtocol,
    System,
    Transition,
)

########################################################################################


def test_unroll_string_labels():
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

    transitions = [
        Transition(
            label="q0->e0",
            level1="q0",
            level2="e0",
            einsteinA=1,
            multipole="E1",
        ),
    ]

    ion = Ion(
        mass=171,
        charge=1,
        position=[0, 0, 0],
        levels=[downstate, estate],
        transitions=transitions,
    )

    system = System(
        ions=[
            ion,
        ],
        modes=[],
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

    protocol = SequentialProtocol(
        sequence=[
            Pulse(beam=beam, duration=1),
        ]
    )

    atomic_circuit = AtomicCircuit(system=system, protocol=protocol)

    ########################################################################################

    unrolled_transitions = [
        Transition(
            label="q0->e0",
            level1=downstate,
            level2=estate,
            einsteinA=1,
            multipole="E1",
        ),
    ]

    unrolled_ion = Ion(
        mass=171,
        charge=1,
        position=[0, 0, 0],
        levels=[downstate, estate],
        transitions=unrolled_transitions,
    )

    unrolled_system = System(
        ions=[
            unrolled_ion,
        ],
        modes=[],
    )

    unrolled_beam = Beam(
        transition=unrolled_transitions[0],
        rabi=2 * np.pi * 1,
        detuning=0,
        phase=0,
        polarization=[1, 0, 0],
        wavevector=[0, 1, 0],
        target=0,
    )

    unrolled_protocol = SequentialProtocol(
        sequence=[
            Pulse(beam=unrolled_beam, duration=1),
        ]
    )

    unrolled_atomic_circuit = AtomicCircuit(
        system=unrolled_system, protocol=unrolled_protocol
    )

    assert unroll_label_pass(atomic_circuit) == unrolled_atomic_circuit


def test_unroll_tuple_labels():
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

    transitions = [
        Transition(
            label="q0->e0",
            level1="q0",
            level2="e0",
            einsteinA=1,
            multipole="E1",
        ),
    ]

    ion = Ion(
        mass=171,
        charge=1,
        position=[0, 0, 0],
        levels=[downstate, estate],
        transitions=transitions,
    )

    system = System(
        ions=[
            ion,
        ],
        modes=[],
    )

    beam = Beam(
        transition=("q0->e0", 0),
        rabi=2 * np.pi * 1,
        detuning=0,
        phase=0,
        polarization=[1, 0, 0],
        wavevector=[0, 1, 0],
        target=0,
    )

    protocol = SequentialProtocol(
        sequence=[
            Pulse(beam=beam, duration=1),
        ]
    )

    atomic_circuit = AtomicCircuit(system=system, protocol=protocol)

    ########################################################################################

    unrolled_transitions = [
        Transition(
            label="q0->e0",
            level1=downstate,
            level2=estate,
            einsteinA=1,
            multipole="E1",
        ),
    ]

    unrolled_ion = Ion(
        mass=171,
        charge=1,
        position=[0, 0, 0],
        levels=[downstate, estate],
        transitions=unrolled_transitions,
    )

    unrolled_system = System(
        ions=[
            unrolled_ion,
        ],
        modes=[],
    )

    unrolled_beam = Beam(
        transition=unrolled_transitions[0],
        rabi=2 * np.pi * 1,
        detuning=0,
        phase=0,
        polarization=[1, 0, 0],
        wavevector=[0, 1, 0],
        target=0,
    )

    unrolled_protocol = SequentialProtocol(
        sequence=[
            Pulse(beam=unrolled_beam, duration=1),
        ]
    )

    unrolled_atomic_circuit = AtomicCircuit(
        system=unrolled_system, protocol=unrolled_protocol
    )

    assert unroll_label_pass(atomic_circuit) == unrolled_atomic_circuit
