# Analog Compiler

The analog compiler is implemented in `src/oqd_core/compiler/analog`. The entry point is the `compile_analog_circuit` function in [`compile.py`](../../src/oqd_core/compiler/analog/compile.py). This function takes a `source` and returns the corresponding circuit and CFG. The input `source` can be a string, AnalogCircuit, or the CFG. The function optionally runs analysis with the `return_analysis` parameter, on the input `source` and returns the output.

For more details on the analog analysis, see [`analog_analysis.md`](../analysis/analog_analysis.md).

The compiler also provides the use of `@analog` as a decorator. This converts the source code of the function and converts them into Analog AST.

## Conversion

The `AnalogCFGBuilder` class defined in [`conversion.py`](../../src/oqd_core/compiler/analog/conversion.py) builds the CFG from an Analog AST. The `AnalogCFGtoAST` class in the same file converts a CFG into Analog AST. 

`PyASTtoAnalog` converts Python AST with Analog functions into Analog AST.

## Canonicalization Rules

Some simple math canonicalization passes are defined in [`canonicalization.py`](../../src/oqd_core/compiler/analog/canonicalization.py):

- `ReplaceSubtraction`
- `UnaryCombination`
- `CollectAssociativeOperands`
- `SortedOrder`

<!-- prettier-ignore -->
/// admonition | Example
    type: example
```py
from oqd_core.compiler.analog.passes.compile import compile_analog_circuit
source = """
q = qreg(2)
h = %X @ %I
evolve(h, 1.0, q)
measure(q)
"""
circuit, cfg = compile_analog_circuit(source)
```
