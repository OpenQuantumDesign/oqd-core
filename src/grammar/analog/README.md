
# Grammar for Analog Interface

Make sure you're in the virtual environment, and sync your requirements.

```bash
uv sync
source .venv/bin/activate
```

## Generate the Lexer and Parser

Navigate into the directory which contains the grammar for the language. Run `antlr4` on the Lexer first, and then the Parser, as the Parser depends on the Lexer.

```bash
cd src/grammar/analog
antlr4 -Dlanguage=Python3 -visitor -listener -o ../../oqd_core/frontend/analog AnalogLexer.g4 
antlr4 -Dlanguage=Python3 -visitor -listener -o ../../oqd_core/frontend/analog AnalogParser.g4 
```

## Run the example

The example code is in `examples/analog/test.analog`. This file contains the expected syntax of the Parser.
Navigate to the project root directory:
```bash
cd ../../..
```

Run the Typer app:
```bash
# Run example
python3 -m oqd_core.frontend.analog.output -i examples/analog/test.analog -o examples/analog/test.ast
```

This example generates the Analog Circuit AST in `examples/analog/test.ast`.

