
# Grammar for Atomic Interface

Make sure you're in the virtual environment, and sync your requirements.

```bash
uv sync
source .venv/bin/activate
```

## Generate the Lexer and Parser

Navigate into the directory which contains the grammar for the language. Run `antlr4` on the Lexer first, and then the Parser, as the Parser depends on the Lexer.

```bash
cd src/grammar/atomic
antlr4 -Dlanguage=Python3 -visitor -listener -o ../../oqd_core/frontend/atomic AtomicLexer.g4 
antlr4 -Dlanguage=Python3 -visitor -listener -o ../../oqd_core/frontend/atomic AtomicParser.g4 
```

## Run the example

The example code is in `examples/atomic/test.atomic`. This file contains the expected syntax of the Parser.
Navigate to the project root directory:
```bash
cd ../../..
```

Run the Typer app:
```bash
# Run example
python3 -m oqd_core.frontend.atomic.output -i examples/atomic/test.atomic -o examples/atomic/test.ast
```

This example generates the Atomic Circuit AST in `examples/atomic/test.ast`.

