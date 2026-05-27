The atomic frontend provides tools to convert atomic source text into a parsed tree and to convert a parsed tree back into source text.

## Frontend Components

The atomic frontend is implemented in `src/oqd_core/frontend/atomic`.

### ANTLR4 Generated Files

These files are generated from the grammar `src/grammar/atomic` using ANTLR4:

- [`AtomicLexer.py`](../../src/oqd_core/frontend/atomic/AtomicLexer.py)
- [`AtomicParser.py`](../../src/oqd_core/frontend/atomic/AtomicParser.py)
- [`AtomicLexer.tokens`](../../src/oqd_core/frontend/atomic/AtomicLexer.tokens)
- [`AtomicParser.tokens`](../../src/oqd_core/frontend/atomic/AtomicParser.tokens)
- [`AtomicLexer.interp`](../../src/oqd_core/frontend/atomic/AtomicLexer.interp)
- [`AtomicParser.interp`](../../src/oqd_core/frontend/atomic/AtomicParser.interp)
- [`AtomicParserListener.py`](../../src/oqd_core/frontend/atomic/AtomicParserListener.py)
- [`AtomicParserVisitor.py`](../../src/oqd_core/frontend/atomic/AtomicParserVisitor.py)

For more details on the atomic grammar, see [`atomic_grammar.md`](../grammar/atomic_grammar.md).

### AST Builder

[`AtomicCircuitAST.py`](../../src/oqd_core/frontend/atomic/AtomicCircuitAST.py) contains the implementation of the AST Builder. The `parse_atomic` function uses the `AtomicASTBuilder` class to convert the atomic source text into an [`AtomicCircuit`][oqd_core.interface.atomic.circuit.AtomicCircuit].

### Serializer

[`serialize.py`](../../src/oqd_core/frontend/atomic/serialize.py) contains the implementation of the serializer. The `serialize_atomic` function uses the `SerializeAtomic` class to convert an [`AtomicCircuit`][oqd_core.interface.atomic.circuit.AtomicCircuit] into atomic source text.

## Usage of the AST Builder and Serializer

### Parse Atomic Source

<!-- prettier-ignore -->
/// admonition | Example
    type: example

```py
from oqd_core.frontend.atomic import parse_atomic

source = """
ions = ionreg(2)
mw = beam(2.0, 1.0, 0.0, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
pulse1 = pulse(mw, 1e-5, ions[0])
parallel {
pulse(mw, 5e-6, ions[0], false)
pulse(mw, 5e-6, ions[1], true)
}
"""

circuit = parse_atomic(source)
```

///

### Serialize an Atomic Circuit

<!-- prettier-ignore -->
/// admonition | Example
    type: example

```py
from oqd_core.frontend.atomic import parse_atomic, serialize_atomic

source = """
ions = ionreg(1)
mw = beam(2.0, 1.0, 0.0, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
pulse(mw, 1e-5, ions[0], true)
"""

circuit = parse_atomic(source)
serialized = serialize_atomic(circuit)
```

///

### Type Checker

The atomic type checker is implemented by the `AtomicTypeChecker` class in [`type_checker.py`](../../src/oqd_core/frontend/atomic/type_checker.py). The `AtomicTypeLattice` class defines a concrete lattice for atomic types with `leq`, `join`, and `meet` methods. The type checker builds a Control Flow Graph (CFG) from the typed AST, and runs the forward dataflow analysis on the CFG. The atomic Control Flow Graph (CFG) is implemented by the `AtomicCFGBuilder` class in [`cfg.py`](../../src/oqd_core/frontend/atomic/cfg.py). This CFGBuilder uses `CFGNode` and `SCCAnalysis` defined in the [analysis module](../../src/oqd_core/analysis/utils.py).

