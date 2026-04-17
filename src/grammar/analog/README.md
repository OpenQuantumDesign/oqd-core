
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

