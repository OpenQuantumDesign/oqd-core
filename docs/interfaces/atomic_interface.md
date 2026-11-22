The atomic interface expresses quantum information experiments in terms of light-matter interactions.

## Ions

Ions are implemented through ion registers and index-based extraction.

- [`IonRegister`][oqd_core.interface.atomic.expr.IonRegister] creates an ion register
- [`AtomicList`][oqd_core.interface.atomic.expr.AtomicList] collects Atomic Expressions in a list

# Atomic Circuit

The [`AtomicCircuit`][oqd_core.interface.atomic.circuit.AtomicCircuit] is the top-level structure that contains atomic statements and control flow.

## Statements

### Atomic Operations

- [`Beam`][oqd_core.interface.atomic.expr.Beam] represents the optical channel.

- [`Pulse`][oqd_core.interface.atomic.expr.Pulse] applies a beam for a duration on a target ion.

- [`ParallelProtocol`][oqd_core.interface.atomic.statement.ParallelProtocol] composes pulses in a parallel fashion.

- [`SerialProtocol`][oqd_core.interface.atomic.statement.SerialProtocol] composes pulses in a serial fashion.

### Declarations

- [`Declaration`][oqd_core.interface.atomic.statement.Declaration] binds an expression result to a named identifier for later use.

### Control Flow

- [`IfElse`][oqd_core.interface.atomic.statement.IfElse] conditionally executes `then_branch` or `else_branch`.

- [`While`][oqd_core.interface.atomic.statement.While] implements the While loop when the condition is true.

- [`Break`][oqd_core.interface.atomic.statement.Break] exits the innermost loop.

- [`Continue`][oqd_core.interface.atomic.statement.Continue] skips to the next loop iteration.

## Usage

<!-- prettier-ignore -->
/// admonition | Example
    type: example

```py
from oqd_core.interface.atomic import AtomicCircuit, Beam, Access

circuit = AtomicCircuit()

mw = Beam(
    frequency=2.0,
    rabi=1.0,
    phase=0.0,
    polarization=[1.0, 0.0, 0.0],
    wavevector=[0.0, 0.0, 1.0],
)

ions = Access(name="ions")
circuit.pulse(beam=mw, duration=10e-6, target=ions, measured=False)
```

///
