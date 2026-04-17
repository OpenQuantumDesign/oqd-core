The atomic grammar defines the text syntax for the atomic frontend.

It is implemented with ANTLR4 using:

- [`AtomicLexer.g4`](../../src/grammar/atomic/AtomicLexer.g4) for tokens
- [`AtomicParser.g4`](../../src/grammar/atomic/AtomicParser.g4) for grammar rules

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

- `ion_register`
- `atomic_list`
- `atomic_list_extract`

### Expressions and Operators

The grammar supports:

- Arithmetic precedence through `aexpr`, `mexpr`, `uexpr`, and `eexpr`
- Boolean logic with both word and symbolic forms (`and`/`&&`, `or`/`||`, `not`/`!`)
- Comparators (`==`, `!=`, `<`, `<=`, `>`, `>=`)
- Arithmetic operator tokens (`+`, `-`, `*`, `/`, `^`)

### Literals, Functions, and Operators

The lexer defines:

- Integer and float literals (`INT`, `FLOAT`)
- Math variables (`MATH_VAR`) and imaginary literal (`IMAG`)
- Function names (`abs`, `sin`, `cos`, `tan`, `exp`, `log`, `sinh`, `cosh`, `tanh`, `atan`, `acos`, `asin`, `atanh`, `asinh`, `acosh`, `heaviside`, `conj`, `real`, `imag`, `atan2`)

### Built-in Functions

The grammar supports the core atomic statements through `beam`, `pulse`, and `parallel`.

- `beam(frequency, rabi, phase, polarization, wavevector)`
  - `frequency`: math expression (`aexpr`)
  - `rabi`: math expression (`aexpr`)
  - `phase`: math expression (`aexpr`)
  - `polarization`: vector expression (`aexpr`),
  - `wavevector`: vector expression (`aexpr`),
- `pulse(beam, duration, target[, measured])`
  - `beam`: beam expression (`aexpr`),
  - `duration`: math expression (`aexpr`)
  - `target`: target expression (`aexpr`)
  - `measured`: boolean expression (`aexpr`), optional, defaults to `false`
- `parallel { ... }`
  - `...`: block (`block`) containing pulse statements to run in parallel

## Generate Parser Files for Frontend

Run generation from the grammar directory and output generated files into `src/oqd_core/frontend/atomic`.

```bash
uv sync
source .venv/bin/activate
cd src/grammar/atomic
antlr4 -Dlanguage=Python3 -visitor -listener -o ../../oqd_core/frontend/atomic AtomicLexer.g4
antlr4 -Dlanguage=Python3 -visitor -listener -o ../../oqd_core/frontend/atomic AtomicParser.g4
```

This generates lexer, parser, listener, and visitor artifacts used by the frontend parser pipeline.
