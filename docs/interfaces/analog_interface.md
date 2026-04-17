The analog interface represents a quantum experiment in terms of time evolving Hamiltonians.

## Quantum Degrees of Freedom

In this analog interface, we allow for 2 different quantum degrees of freedom:

- Qubits consist of a pair of states (spin $\uparrow$ and spin $\downarrow$).
- Bosonic degrees of freedom form a fock space.

They are implemented using registers and index-based extraction:

- [`QuantumRegister`][oqd_core.interface.analog.expr.QuantumRegister] creates a qubit register
- [`ModeRegister`][oqd_core.interface.analog.expr.ModeRegister] creates a bosonic mode register
- [`Extract`][oqd_core.interface.analog.expr.Extract] accesses a target by index
- [`AnalogList`][oqd_core.interface.analog.expr.AnalogList] collects analog expressions in a list

## Analog Operators

/// tab | Pauli

The basis of operators for the qubits are the Pauli operators:

- $\sigma^I$ <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.PauliI] </div>
- $\sigma^x$ <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.PauliX] </div>
- $\sigma^y$ <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.PauliY] </div>
- $\sigma^z$ <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.PauliZ] </div>

///

/// tab | Ladder

The basis of operators for the bosonic degree of freedom are the ladder operators:

- $a$ <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.Annihilation] </div>
- $a^{\dagger}$ <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.Creation] </div>
- $I$ <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.Identity] </div>

///

### Operator Operations

The basis operators can be combined with the operations:

- Addition <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.OperatorAdd] </div>

- Subtraction <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.OperatorSub] </div>

- Multiplication <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.OperatorMul] </div>

- Tensor Product <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.OperatorKron] </div>

## Math Expressions

Analog parameters support math expressions.

### Math Primitives

- [`MathNum`][oqd_core.interface.analog.expr.MathNum] represents numeric literals in analog math expressions.
- [`MathImag`][oqd_core.interface.analog.expr.MathImag] represents the imaginary unit.
- [`MathVar`][oqd_core.interface.analog.expr.MathVar] represents compile-time variables in expressions.

### Math Operations

- Addition <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.MathAdd] </div>
- Subtraction <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.MathSub] </div>
- Multiplication <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.MathMul] </div>
- Division <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.MathDiv] </div>
- Exponentiation <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.MathPow] </div>
- Named functions <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.MathFunc] </div>

Compatible named functions include:
`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `exp`, `log`, `real`, `imag`, `conj`, `abs`, `heaviside`.

## Boolean Expressions

Boolean expressions are used for conditional and loop control flow. <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.Bool] </div>

### Boolean Operations

- NOT <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.BoolNot] </div>
- AND <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.BoolAnd] </div>
- OR <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.BoolOr] </div>
- Equal <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.BoolEq] </div>
- Not equal <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.BoolNotEq] </div>
- Less than <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.BoolLessThan] </div>
- Less than or equal <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.BoolLessThanEq] </div>
- Greater than <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.BoolGreaterThan] </div>
- Greater than or equal <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.BoolGreaterThanEq] </div>


# Analog Circuit

The [AnalogCircuit][oqd_core.interface.analog.circuit.AnalogCircuit] is the top level structure that contains analog statements and control flow.

## Statements

### Analog Operations

- [`Initialize`][oqd_core.interface.analog.expr.Initialize] initializes the specified targets before evolution.

- [`Evolve`][oqd_core.interface.analog.expr.Evolve] applies a Hamiltonian for a duration on the specified targets.

- [`Measure`][oqd_core.interface.analog.expr.Measure] performs measurement on the specified targets.

### Declarations

- [`Declaration`][oqd_core.interface.analog.statement.Declaration] binds an expression result to a named identifier for later use.

### Control Flow

- [`IfElse`][oqd_core.interface.analog.statement.IfElse] conditionally executes `then_branch` or `else_branch`.

- [`While`][oqd_core.interface.analog.statement.While] implements the While loop when the condition is true.

- [`Break`][oqd_core.interface.analog.statement.Break] exits the innermost loop.

- [`Continue`][oqd_core.interface.analog.statement.Continue] skips to the next loop iteration.

## Usage

<!-- prettier-ignore -->
/// admonition | Example
    type: example

```py
from oqd_core.interface.analog import AnalogCircuit, Initialize, Measure, Evolve

circuit = AnalogCircuit()
circuit.Initialize(targets=q)
circuit.Evolve(hamiltonian=H, duration=1, targets=q)
circuit.Measure(targets=q)
```

///
