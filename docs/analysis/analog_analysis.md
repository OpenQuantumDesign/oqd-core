# Analysis

The analog analysis pipeline is implemented in `oqd_core/src/analysis/analog`. The [`Analyze`][oqd_core.analysis.analog.analyze.Analyze] class builds a Control Flow Graph (CFG), runs the type checker, and builds the symbol table. The result is stored in [`AnalogAnalysisResult`][oqd_core.analysis.analog.analyze.AnalogAnalysisResult] containing the CFG, type dataflow result, and symbol table.

## Control Flow Graph

The analog Control Flow Graph (CFG) is implemented by the `AnalogCFGBuilder` class in [`cfg.py`](../../src/oqd_core/analysis/analog/cfg.py). This CFGBuilder uses `ControlFlowGraph` and `Block` defined in the [analysis module](../../src/oqd_core/analysis/utils/control_flow.py).

## Type Checker

The analog type checker is implemented by the `AnalogTypeChecker` class in [`type_checker.py`](../../src/oqd_core/analysis/analog/type_checker.py). The `AnalogTypeLattice` class defines a concrete lattice for analog types with `leq`, `join`, and `meet` methods. Type inference rules are implemented by the `AnalogSemantics` class in [`semantics.py`](../../src/oqd_core/analysis/analog/semantics.py). The type checker runs forward dataflow analysis over the CFG.

## Symbol Table

The analog symbol table is implemented in [`symbol_table.py`](../../src/oqd_core/analysis/analog/symbol_table.py). The `AnalogSymbolTableBuilder` class runs a second forward dataflow pass over the CFG, using the type checker output to track register and target dimensions. The result is an `AnalogSymbolTable` with register environments per block and a statement index.


## Usage of the Type Checker

<!-- prettier-ignore -->
/// admonition | Example
    type: example

```py
from oqd_core.frontend.analog import parse_analog
from oqd_core.analysis.analog.cfg import AnalogCFGBuilder
from oqd_core.analysis.analog.type_checker import AnalogTypeChecker

source = """
q = qreg(2)
h = %X %@ %I
evolve(h, 1.0, q)
measure(q)
"""

circuit = parse_analog(source)
cfg = AnalogCFGBuilder().run(circuit)
type_checker = AnalogTypeChecker(cfg)
```

For a complete example:
```py
from oqd_core.frontend.analog import parse_analog
from oqd_core.analysis.analog import Analyze
source = """
q = qreg(2)
h = %X %@ %I
evolve(h, 1.0, q)
measure(q)
"""
circuit = parse_analog(source)
analysis = Analyze(circuit)
result = analysis.result

```
///

The output of the type checker is stored in `type_checker.result`. The type checker dataflow result is available at `result.dataflow_result`, and the symbol table is available at `result.symbol_table`.

