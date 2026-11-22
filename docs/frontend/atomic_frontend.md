The atomic frontend provides tools to convert atomic source text into a typed Abstract Syntax Tree (AST) and to convert a typed AST back into source text.

## Frontend Components

The atomic frontend is implemented in `src/oqd_core/frontend/atomic`.

### ANTLR4 Generated Files

These files are generated from the grammar `src/grammar/atomic` using ANTLR4:

- [`AtomicLexer.py`](../../src/oqd_core/frontend/atomic/_generated/AtomicLexer.py)
- [`AtomicParser.py`](../../src/oqd_core/frontend/atomic/_generated/AtomicParser.py)
- [`AtomicLexer.tokens`](../../src/oqd_core/frontend/atomic/_generated/AtomicLexer.tokens)
- [`AtomicParser.tokens`](../../src/oqd_core/frontend/atomic/_generated/AtomicParser.tokens)
- [`AtomicLexer.interp`](../../src/oqd_core/frontend/atomic/_generated/AtomicLexer.interp)
- [`AtomicParser.interp`](../../src/oqd_core/frontend/atomic/_generated/AtomicParser.interp)
- [`AtomicParserListener.py`](../../src/oqd_core/frontend/atomic/_generated/AtomicParserListener.py)
- [`AtomicParserVisitor.py`](../../src/oqd_core/frontend/atomic/_generated/AtomicParserVisitor.py)

For more details on the atomic grammar, see [`atomic_grammar.md`](../grammar/atomic_grammar.md).

### AST Builder

[`AtomicASTBuilder.py`](../../src/oqd_core/frontend/atomic/AtomicASTBuilder.py) contains the implementation of the AST Builder. The `parse_atomic` function uses the `AtomicASTBuilder` class to convert the atomic source text into the Atomic AST.

### Error Listener

[`AtomicParserErrorListener.py`](../../src/oqd_core/frontend/atomic/AtomicParserErrorListener.py) implements a custom error listener class for the atomic interface.

### Serializer

[`serialize.py`](../../src/oqd_core/frontend/atomic/serialize.py) contains the implementation of the serializer. The `serialize_atomic` function uses the `SerializeAtomic` class to convert Atomic AST into atomic source text.

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

