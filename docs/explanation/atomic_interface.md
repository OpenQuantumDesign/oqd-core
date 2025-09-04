The atomic interface expresses quantum information experiments in terms of light-matter interactions.

## System <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.atomic.system.System] </div>

The system describes the properties of the trapped-ion quantum device.

### Ion <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.atomic.system.Ion] </div>

An ion is described by its set of electronic energy levels. Each energy level has its associated quantum numbers:

- Principal quantum number
- Spin angular momentum, $S$
- Orbital angular momentum, $L$
- Spin-orbital angular momentum, $J = S + L$
- Nuclear angular momentum, $I$
- Spin-orbital-nuclear angular momentum, $F = J + I$
- Magnetization, $m_F$
- Energy, $E$

with the set of electronic energy levels, we assign two states to be the qubit states.

Manipulating the qubit states involves driving transitions between the qubit states of the ions, either directly or indirectly.

<!-- prettier-ignore -->
/// admonition | Example
    type: example

Definiition of an [`Ion`][oqd_core.interface.atomic.system.Ion] for $^{171}\mathrm{Yb}^+$:

```python

downstate = Level(
    label="q0",
    principal=6,
    spin=1/2,
    orbital=0,
    nuclear=1/2,
    spin_orbital=1/2,
    spin_orbital_nuclear=0,
    spin_orbital_nuclear_magnetization=0,
    energy=0,
)
upstate = Level(
    label="q1",
    principal=6,
    spin=1/2,
    orbital=0,
    nuclear=1/2,
    spin_orbital=1/2,
    spin_orbital_nuclear=1,
    spin_orbital_nuclear_magnetization=0,
    energy=2*pi*12.643e9,
)
estate = Level(
    label="e0",
    principal=5,
    spin=1/2,
    orbital=1,
    nuclear=1/2,
    spin_orbital=1/2,
    spin_orbital_nuclear=0,
    spin_orbital_nuclear_magnetization=0,
    energy=2*pi*811.52e12,
)

Yb171 = Ion(
    mass=171,
    charge=1,
    position=[0,0,0],
    levels=[
        downstate,
        upstate,
        estate,
    ],
    transitions=[
        Transition(
            label="q0->q1",
            level1=downstate,
            level2=upstate,
            einsteinA=...,
        ),
        Transition(
            label="q0->e0",
            level1=downstate,
            level2=estate,
            einsteinA=...,
        ),
        Transition(
            label="q1->e0",
            level1=upstate,
            level2=estate,
            einsteinA=...,
        ),
    ],
)
```

///

### Phonon <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.atomic.system.Phonon] </div>

In the trapped-ion system the system exhibits collective phonon modes, which are bosonic degrees of freedom.

These phonon modes are characterized by:

- Energy (eigenfrequency)
- Profile of the collective phonon mode in terms of the ions' motion (eigenvector)

<!-- prettier-ignore -->
/// admonition | Example
    type: example

Definition of the set of phonon modes for a trapped-ion system with a single ion:

```python
COM_x = Phonon(
    energy=2*pi*5e6,
    eigenvector=[1,0,0]
)
COM_y = Phonon(
    energy=2*pi*5e6,
    eigenvector=[0,1,0]
)
COM_z = Phonon(
    energy=2*pi*1e6,
    eigenvector=[0,0,1]
)
```

///

### Other

<!-- prettier-ignore -->
/// admonition | Not Implemented
    type: warning
The system is further described by a list of experimental parameters that require calibration to determine, e.g.:

- Maximum laser power
- Laser lock frequency
- etc.

These parameters will in the future be included in the [`System`][oqd_core.interface.atomic.system.System].

The [`System`][oqd_core.interface.atomic.system.System] will be retrieved from a calibration database to determine the current state of the system and the status of all calibrations required to run quantum experiments.

///

## Pulse Program <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.atomic.protocol.Protocol] </div>

The pulse program for a quantum experiment is described by a [`Protocol`][oqd_core.interface.atomic.protocol.Protocol]. The protocol defines the list of optical channels in the experiment and the real-time scheduling of pulses of the optical channels in order to perform the quantum experiment.

### Optical Channel <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.atomic.protocol.Beam] </div>

An optical channel is described by a [`Beam`][oqd_core.interface.atomic.protocol.Beam] with the following parameters:

- Transition of the ion for which to reference the Beam to.
- Rabi frequency to drive the referenced transition with.
- Detuning from the resonance of the referenced transition.
- Phase of the beam relative to the clock of the ion.
- Polarization of the beam.
- Wavevector of the beam.
- Target ion addressed by the beam.

<!-- prettier-ignore -->
/// admonition | Example
    type: example

Beam used to drive a microwave Rabi oscillation in the X-axis:

```python
microwave_beam = Beam(
    transition=Transition(level1=downstate,level2=upstate,...),
    rabi= 2*pi*1e6,
    detuning=0,
    phase=0,
    polarization=...
    wavevector=...
    target=0
)
```

///

<!-- prettier-ignore -->
/// admonition | Note
    type: note

The following parameters may be specified with the [math interface](#explanation/math_interface):

- Rabi frequency
- Detuning
- Phase
  ///

### Pulse <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.atomic.protocol.Pulse] </div>

A pulse turns on an optical channel for a duration of time.

<!-- prettier-ignore -->
/// admonition | Example
    type: example

Pulse that drives a microwave Rabi oscillation in the X-axis for a duration $T$:

```python
microwave_pulse = Pulse(
    beam=microwave_beam,
    duration=T,
)
```

///

#### Measurement

To perform a measurement, we not only have to turn on an optical channel, we also have to turn on a detector, this is handled by the [`MeasurePulse`][oqd_core.interface.atomic.protocol.MeasurePulse].

<!-- prettier-ignore -->
/// admonition | Example
    type: example

Pulse that drives a fluorescent transition and turns on the detector for a duration $T$:

```python
detection_pulse = MeasurePulse(
    beam=detection_beam,
    duration=T,
)
```

///

### Composition of Protocols

The pulse program for a quantum experiment is usually more complex than a pulse of a single beam. This is handled with [`SequentialProtocol`][oqd_core.interface.atomic.protocol.SequentialProtocol] and [`ParallelProtocol`][oqd_core.interface.atomic.protocol.ParallelProtocol].

/// tab | `SequentialProtocol`

//// html | div[style='float: right']
[![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.atomic.protocol.SequentialProtocol]
////

Sequential protocol applies a set of pulses or subprotocols sequentially in time.

<!-- prettier-ignore -->
//// admonition | Example
    type: example

The following protocol is for a Rabi flop and a measurement:

```python
microwave_beam = Beam(
    transition=Transition(level1=downstate,level2=upstate,...),
    rabi= 2*pi*1e6,
    detuning=0,
    phase=0,
    polarization=...
    wavevector=...
    target=0
)

detection_beam = Beam(
    transition=Transition(level1=upstate,level2=estate,...),
    rabi= 2*pi*1e6,
    detuning=0,
    phase=0,
    polarization=...
    wavevector=...
    target=0
)

protocol = SequentialProtocol(
    sequence=[
        Pulse(beam=raman1_beam,duration=T),
        Pulse(beam=raman2_beam,duration=100e-6)
        ]
    )
```

////
///

/// tab | `ParallelProtocol`

//// html | div[style='float: right']
[![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.atomic.protocol.ParallelProtocol]
////

Sequential protocol applies a set of pulses or subprotocols parallel in time.

<!-- prettier-ignore -->
//// admonition | Example
    type: example

The following protocol is for a two-photon Raman transition:

```python
raman1_beam = Beam(
    transition=Transition(level1=downstate,level2=estate,...),
    rabi= 2*pi*1e6,
    detuning=2*pi*1e9,
    phase=0,
    polarization=...
    wavevector=...
    target=0
)
raman2_beam = Beam(
    transition=Transition(level1=upstate,level2=estate,...),
    rabi= 2*pi*1e6,
    detuning=2*pi*1e9,
    phase=0,
    polarization=...
    wavevector=...
    target=0
)

protocol = ParallelProtocol(
    sequence=[
        Pulse(beam=raman1_beam,duration=T),
        Pulse(beam=raman2_beam,duration=T)
        ]
    )
```

////

///
