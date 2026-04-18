The analog grammar defines the text syntax for the analog frontend.

It is implemented with ANTLR4 using:

- [`AnalogLexer.g4`](../../src/grammar/analog/AnalogLexer.g4) for tokens
- [`AnalogParser.g4`](../../src/grammar/analog/AnalogParser.g4) for grammar rules

## ANTLR4 Basics

ANTLR4 splits the language definition into two parts:

- A lexer grammar that converts characters into tokens
- A parser grammar that converts tokens into a parse tree

## Grammar Features

### Program and Blocks

- `program` is the container for the code
- `block` supports newline-separated statements and empty lines
- `statement` supports declarations, control flow, and expressions

### Statements and Control Flow

The grammar supports:

- Variable declarations with `declaration`
- Conditionals with `ifelse_stmt`
- Loops with `while_stmt`
- Loop control with `break_stmt` and `continue_stmt`

### Registers, Lists, and Indexing

The grammar includes register constructors and list extraction:

- `quantum_register`
- `mode_register`
- `analog_list`
- `analog_list_extract`

### Expressions and Operators

The grammar supports:

- Arithmetic precedence through `aexpr`, `mexpr`, `uexpr`, and `eexpr`
- Boolean logic with both word and symbolic forms (`and`/`&&`, `or`/`||`, `not`/`!`)
- Comparators (`==`, `!=`, `<`, `<=`, `>`, `>=`)
- Arithmetic operator tokens (`+`, `-`, `*`, `/`, `^`)
- Analog operator tokens (`%@`, `%+`, `%-`, `%*`)

### Literals, Functions, and Operators

The lexer defines:

- Integer and float literals (`INT`, `FLOAT`)
- Math variables (`MATH_VAR`) and imaginary literal (`IMAG`)
- Function names (`abs`, `sin`, `cos`, `tan`, `exp`, `log`, `sinh`, `cosh`, `tanh`, `atan`, `acos`, `asin`, `atanh`, `asinh`, `acosh`, `heaviside`, `conj`, `real`, `imag`, `atan2`)
- Quantum operator tokens (`%I`, `%X`, `%Y`, `%Z`, `%C`, `%A`, `%J`)

### Built-in Functions

The grammar supports the core analog statements through `initialize`, `evolve`, and `measure`:

- `initialize(targets)`
  - `targets`: target expression (`expr`)
- `measure(targets)`
  - `targets`: target expression (`expr`)
- `evolve(hamiltonian, duration, targets)`
  - `hamiltonian`: operator expression (`expr`)
  - `duration`: math expression (`aexpr`)
  - `targets`: target expression (`expr`)

## Generate Parser Files for Frontend

Run generation from the grammar directory and output generated files into `src/oqd_core/frontend/analog`. Run `antlr4` on the Lexer first, and then the Parser, as the Parser depends on the Lexer.

```bash
uv sync
source .venv/bin/activate
cd src/grammar/analog
antlr4 -Dlanguage=Python3 -visitor -listener -o ../../oqd_core/frontend/analog AnalogLexer.g4
antlr4 -Dlanguage=Python3 -visitor -listener -o ../../oqd_core/frontend/analog AnalogParser.g4
```

This generates lexer, parser, listener, and visitor artifacts used by the frontend parser pipeline.
