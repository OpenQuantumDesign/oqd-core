The analog frontend provides tools to convert analog source text into a typed Abstract Syntax Tree (AST) and to convert a typed AST back into source text.

## Frontend Components

The analog frontend is implemented in `src/oqd_core/frontend/analog`.

### ANTLR4 Generated Files

These files are generated from the grammar `src/grammar/analog` using ANTLR4:

- [`AnalogLexer.py`](../../src/oqd_core/frontend/analog/AnalogLexer.py)
- [`AnalogParser.py`](../../src/oqd_core/frontend/analog/AnalogParser.py)
- [`AnalogLexer.tokens`](../../src/oqd_core/frontend/analog/AnalogLexer.tokens)
- [`AnalogParser.tokens`](../../src/oqd_core/frontend/analog/AnalogParser.tokens)
- [`AnalogLexer.interp`](../../src/oqd_core/frontend/analog/AnalogLexer.interp)
- [`AnalogParser.interp`](../../src/oqd_core/frontend/analog/AnalogParser.interp)
- [`AnalogParserListener.py`](../../src/oqd_core/frontend/analog/AnalogParserListener.py)
- [`AnalogParserVisitor.py`](../../src/oqd_core/frontend/analog/AnalogParserVisitor.py)

For more details on the analog grammar, see [`analog_grammar.md`](../grammar/analog_grammar.md).

### AST Builder

[`AnalogCircuitAST.py`](../../src/oqd_core/frontend/analog/AnalogCircuitAST.py) contains the implementation of the AST Builder. The `parse_analog` function uses the `AnalogASTBuilder` class to convert the analog source text into an [`AnalogCircuit`][oqd_core.interface.analog.circuit.AnalogCircuit]. It produces a typed AST of the source code.

### Serializer

[`serialize.py`](../../src/oqd_core/frontend/analog/serialize.py) contains the implementation of the serializer. The `serialize_analog` function uses the `SerializeAnalog` class to convert an [`AnalogCircuit`][oqd_core.interface.analog.circuit.AnalogCircuit] into analog source text.

## Usage of the AST Builder and Serializer

### Parse Analog Source

<!-- prettier-ignore -->
/// admonition | Example
    type: example

```py
from oqd_core.frontend.analog import parse_analog

source = """
q = qreg(2)
h = %X %@ %I
evolve(h, 1.0, q)
measure(q)
"""

circuit = parse_analog(source)
```

///

### Serialize an Analog Circuit

<!-- prettier-ignore -->
/// admonition | Example
    type: example

```py
from oqd_core.frontend.analog import parse_analog, serialize_analog

source = "q = qreg(1)\ninitialize(q)\nmeasure(q)\n"
circuit = parse_analog(source)
serialized = serialize_analog(circuit)
```

///

### Type Checker

The analog type checker is implemented by the `AnalogTypeChecker` class in [`type_checker.py`](../../src/oqd_core/frontend/analog/type_checker.py). The `AnalogTypeLattice` class defines a concrete lattice for analog types with `leq`, `join`, and `meet` methods. The type checker builds a Control Flow Graph (CFG) from the typed AST, and runs the forward dataflow analysis on the CFG. The analog Control Flow Graph (CFG) is implemented by the `AnalogCFGBuilder` class in [`cfg.py`](../../src/oqd_core/frontend/analog/cfg.py). This CFGBuilder uses `ControlFlowGraph`, `CFGNode`, and `SCCAnalysis` defined in the [analysis module](../../src/oqd_core/analysis/utils.py).


## Usage of the Type Checker

<!-- prettier-ignore -->
/// admonition | Example
    type: example

```py
from oqd_core.frontend.analog import parse_analog
from oqd_core.frontend.analog.cfg import AnalogCFGBuilder
from oqd_core.frontend.analog.type_checker import AnalogTypeChecker

source = """
q = qreg(2)
h = %X %@ %I
evolve(h, 1.0, q)
measure(q)
"""

circuit = parse_analog(source)
cfg = AnalogCFGBuilder().run(circuit)
type_checker = AnalogTypeChecker(cfg)
```

///

The output of the type checker is stored in `type_checker.result`.
