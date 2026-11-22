The analog frontend provides tools to convert analog source text into a typed Abstract Syntax Tree (AST) and to convert a typed AST back into source text.

## Frontend Components

The analog frontend is implemented in `src/oqd_core/frontend/analog`.

### ANTLR4 Generated Files

These files are generated from the grammar `src/grammar/analog` using ANTLR4:

- [`AnalogLexer.py`](../../src/oqd_core/frontend/analog/_generated/AnalogLexer.py)
- [`AnalogParser.py`](../../src/oqd_core/frontend/analog/_generated/AnalogParser.py)
- [`AnalogLexer.tokens`](../../src/oqd_core/frontend/analog/_generated/AnalogLexer.tokens)
- [`AnalogParser.tokens`](../../src/oqd_core/frontend/analog/_generated/AnalogParser.tokens)
- [`AnalogLexer.interp`](../../src/oqd_core/frontend/analog/_generated/AnalogLexer.interp)
- [`AnalogParser.interp`](../../src/oqd_core/frontend/analog/_generated/AnalogParser.interp)
- [`AnalogParserListener.py`](../../src/oqd_core/frontend/analog/_generated/AnalogParserListener.py)
- [`AnalogParserVisitor.py`](../../src/oqd_core/frontend/analog/_generated/AnalogParserVisitor.py)

For more details on the analog grammar, see [`analog_grammar.md`](../grammar/analog_grammar.md).

### AST Builder

[`AnalogASTBuilder.py`](../../src/oqd_core/frontend/analog/AnalogASTBuilder.py) contains the implementation of the AST Builder. The `parse_analog` function uses the `AnalogASTBuilder` class to convert the analog source text into the Analog AST.

### Error Listener

[`AnalogParserErrorListener.py`](../../src/oqd_core/frontend/analog/AnalogParserErrorListener.py) implements a custom error listener class for the analog interface.

### Serializer

[`serialize.py`](../../src/oqd_core/frontend/analog/serialize.py) contains the implementation of the serializer. The `serialize_analog` function uses the `SerializeAnalog` class to convert Analog AST into analog source text.

## Usage of the AST Builder and Serializer

### Parse Analog Source

<!-- prettier-ignore -->
/// admonition | Example
    type: example

```py
from oqd_core.frontend.analog import parse_analog

source = """
q = qreg(2)
h = %X @ %I
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

