# Atomic Compiler

The atomic compiler is implemented in `src/oqd_core/compiler/atomic`. The entry point is the `compile_atomic_circuit` function in [`compile.py`](../../src/oqd_core/compiler/atomic/compile.py). This function takes a `source` and returns the corresponding circuit and CFG. The input `source` can be a string, AtomicCircuit, or the CFG. The function optionally runs analysis with the `return_analysis` parameter, on the input `source` and returns the output.

For more details on the atomic analysis, see [`atomic_analysis.md`](../analysis/atomic_analysis.md).

The compiler also provides the use of `@atomic` as a decorator. This feature is still under development and will be refined in future version.

## Conversion

The `AtomicCFGBuilder` class defined in [`conversion.py`](../../src/oqd_core/compiler/atomic/conversion.py) builds the CFG from an Atomic AST. The `AtomicCFGtoAST` class in the same file converts a CFG into Atomic AST. 

`PyASTtoAtomic` converts Python AST into Atomic AST. This feature is still under development and will be refined in future versions.

## Canonicalization Rules

Some simple math canonicalization passes are defined in [`canonicalization.py`](../../src/oqd_core/compiler/atomic/canonicalization.py):

- `ReplaceSubtraction`
- `UnaryCombination`
- `CollectAssociativeOperands`
- `SortedOrder`

<!-- prettier-ignore -->
/// admonition | Example
    type: example
```py
from oqd_core.compiler.atomic.passes.compile import compile_atomic_circuit
source = """
ions = ionreg(1)
mw = beam(2.0, 1.0, 0.0, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
pulse(mw, 1e-5, ions[0], true)
"""
circuit, cfg = compile_atomic_circuit(source)
```
