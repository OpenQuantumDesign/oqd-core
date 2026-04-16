The analog interface represents a quantum experiment in terms of time evolving Hamiltonians.

## Quantum Degrees of Freedom

In this analog interface, we allow for 2 different quantum degrees of freedom:

/// tab | Qubits
Qubits consist of a pair of states (spin $\uparrow$ and spin $\downarrow$).
///
/// tab | Bosonic
Bosonic degrees of freedom form a fock space.
///

/// tab | Registers
//// html | div[style='float: right']
[![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.QuantumRegister]
////
//// html | div[style='float: right']
[![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.ModeRegister]
////
- [`QuantumRegister`][oqd_core.interface.analog.expr.QuantumRegister] creates a qubit register
- [`ModeRegister`][oqd_core.interface.analog.expr.ModeRegister] creates a bosonic mode register
///

## Operators

/// tab | Pauli

The basis of operators for the qubits are the Pauli operators:

- $\sigma^I$ <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.operator.PauliI] </div>
- $\sigma^x$ <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.operator.PauliX] </div>
- $\sigma^y$ <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.operator.PauliY] </div>
- $\sigma^z$ <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.operator.PauliZ] </div>

///

/// tab | Ladder

The basis of operators for the bosonic degree of freedom are the ladder operators:

- $a$ <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.operator.Annihilation] </div>
- $a^{\dagger}$ <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.operator.Creation] </div>
- $I$ <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.operator.Identity] </div>

///

### Operator Operations

The basis operators can be combined with the operations:

- Addition <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.OperatorAdd] </div>

- Subtraction <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.OperatorSub] </div>

- Multiplication <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.OperatorMul] </div>

- Tensor Product <div style="float:right"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.expr.OperatorKron] </div>

## Quantum Operations

- [`Initialize`][oqd_core.interface.analog.expr.Initialize]
- [`Evolve`][oqd_core.interface.analog.expr.Evolve]
- [`Measure`][oqd_core.interface.analog.expr.Measure]

`Evolve` applies a Hamiltonian for a duration on targets:
$$
U = e^{iHt}
$$

## Analog Circuit <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.analog.circuit.AnalogCircuit] </div>

The [AnalogCircuit][oqd_core.interface.analog.circuit.AnalogCircuit] is the top level structure that contains analog statements and control flow.

It supports:
- quantum operations (`initialize`, `evolve`, `measure`)
- declarations
- lists
- conditionals (`if`/`else`)
- loops (`while`)
- loop controls (`break`, `continue`)

## Usage

<!-- prettier-ignore -->
/// admonition | Example
    type: example

```py
circuit = AnalogCircuit()

circuit.initialize()
circuit.evolve(gate, duration=1)
circuit.measure()
```

///
