# Analysis

The atomic analysis pipeline is implemented in `src/oqd_core/analysis/atomic`.

## Control Flow Graph

The atomic Control Flow Graph (CFG) is implemented by the `AtomicCFGBuilder` class in [`cfg.py`](../../src/oqd_core/analysis/atomic/cfg.py). This CFGBuilder uses `ControlFlowGraph` and `Block` defined in the [analysis module](../../src/oqd_core/analysis/utils/control_flow.py).

## Type Checker

The atomic type checker is implemented by the `AtomicTypeChecker` class in [`type_checker.py`](../../src/oqd_core/analysis/atomic/type_checker.py). The `AtomicTypeLattice` class defines a concrete lattice for atomic types with `leq`, `join`, and `meet` methods. Type inference rules are implemented by the `AtomicSemantics` class in [`semantics.py`](../../src/oqd_core/analysis/atomic/semantics.py). The type checker runs forward dataflow analysis over the CFG.

Protocol blocks (`parallel` / `serial`) accept only pulse statements: `Pulse` nodes, pulse declarations, pulse variable references, and nested protocols. In `serial` blocks, pulse declarations update the local environment for later statements in the same block.

## Symbol Table

The atomic symbol table is implemented in [`symbol_table.py`](../../src/oqd_core/analysis/atomic/symbol_table.py). The `AtomicSymbolTableBuilder` class runs a second forward dataflow pass over the CFG, using the type checker output to track register and target dimensions. The result is an `AtomicSymbolTable` with register environments per block and a statement index.

`SymbolBinding.target_dim` is an `int` giving the total ion-target dimension (register size, extracted ion, or sum of list elements). Unlike the analog symbol table, atomic targets do not use a `(qreg, qmode)` tuple.

## Usage of the type checker

<!-- prettier-ignore -->
/// admonition | Example
    type: example

```py
from oqd_core.frontend.atomic import parse_atomic
from oqd_core.analysis.atomic import AtomicCFGBuilder, AtomicTypeChecker, AtomicSymbolTableBuilder

source = """
ions = ionreg(1)
mw = beam(2.0, 1.0, 0.0, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
pulse(mw, 1e-5, ions[0], true)
"""

circuit = parse_atomic(source)
cfg = AtomicCFGBuilder().run(circuit)
type_checker = AtomicTypeChecker(cfg)
symbol_table = AtomicSymbolTableBuilder(cfg, type_checker.dataflow_result).symbol_table
```

///

The type checker dataflow result is available at `type_checker.dataflow_result`.
