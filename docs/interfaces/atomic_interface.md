The atomic interface expresses quantum information experiments in terms of light-matter interactions.

## Ions

Ions are implemented through ion registers and index-based extraction.

- [`IonRegister`][oqd_core.interface.atomic.expr.IonRegister] creates an ion register
- [`Extract`][oqd_core.interface.atomic.expr.Extract] accesses a target ion by index
- [`AtomicList`][oqd_core.interface.atomic.expr.AtomicList] collects Atomic Expressions in a list

## Math Expressions

Atomic parameters support math expressions.

### Math Primitives

- [`MathNum`][oqd_core.interface.atomic.expr.MathNum] represents numeric literals in atomic math expressions.
- [`MathImag`][oqd_core.interface.atomic.expr.MathImag] represents the imaginary unit.
- [`MathVar`][oqd_core.interface.atomic.expr.MathVar] represents compile-time variables in expressions.

### Math Operations

- Addition <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.atomic.expr.MathAdd] </div>
- Subtraction <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.atomic.expr.MathSub] </div>
- Multiplication <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.atomic.expr.MathMul] </div>
- Division <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.atomic.expr.MathDiv] </div>
- Exponentiation <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.atomic.expr.MathPow] </div>
- Named functions <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.atomic.expr.MathFunc] </div>

Compatible named functions include:
`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `exp`, `log`, `real`, `imag`, `conj`, `abs`, `heaviside`.

## Boolean Expressions

Boolean expressions are used for conditional and loop control flow. <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.atomic.expr.Bool] </div>

### Boolean Operations

- NOT <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.atomic.expr.BoolNot] </div>
- AND <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.atomic.expr.BoolAnd] </div>
- OR <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.atomic.expr.BoolOr] </div>
- Equal <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.atomic.expr.BoolEq] </div>
- Not equal <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.atomic.expr.BoolNotEq] </div>
- Less than <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.atomic.expr.BoolLessThan] </div>
- Less than or equal <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.atomic.expr.BoolLessThanEq] </div>
- Greater than <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.atomic.expr.BoolGreaterThan] </div>
- Greater than or equal <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.atomic.expr.BoolGreaterThanEq] </div>


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
from oqd_core.interface.atomic import AtomicCircuit, Beam

circuit = AtomicCircuit()

mw = Beam(
    frequency=2.0,
    rabi=1.0,
    phase=0.0,
    polarization=[1.0, 0.0, 0.0],
    wavevector=[0.0, 0.0, 1.0],
)

circuit.pulse(duration=10e-6, target=0, beam=mw, measured=False)
```

///
