
# Grammar for Analog Interface

Make sure you're in the virtual environment, and sync your requirements.

```bash
uv sync
source .venv/bin/activate
```

## Generate the Lexer and Parser

Navigate into the directory which contains the grammar for the language. Run `antlr4` on the Lexer first, and then the Parser, as the Parser depends on the Lexer.

```bash
cd src/oqd_core/grammar/analog
antlr4 -Dlanguage=Python3 -visitor -listener -o ../../frontend/analog AnalogLexer.g4 
antlr4 -Dlanguage=Python3 -visitor -listener -o ../../frontend/analog AnalogParser.g4 
```

## Run the example

Navigate to the project root directory:
```bash
cd ../../../..
```

Run the Typer app:
```bash
# Run example
python3 -m oqd_core.frontend.analog.output -i examples/Analog/example.analog -o examples/Analog/example.out
```

This example saves the parsed tree in `examples/Analog/example.out`.

