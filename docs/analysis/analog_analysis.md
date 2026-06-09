# Type Checker

The analog type checker is implemented by the `AnalogTypeChecker` class in [`type_checker.py`](../../src/oqd_core/analysis/analog/type_checker.py). The `AnalogTypeLattice` class defines a concrete lattice for analog types with `leq`, `join`, and `meet` methods. The type checker builds a Control Flow Graph (CFG) from the typed AST, and runs the forward dataflow analysis on the CFG. The analog Control Flow Graph (CFG) is implemented by the `AnalogCFGBuilder` class in [`cfg.py`](../../src/oqd_core/analysis/analog/cfg.py). This CFGBuilder uses `ControlFlowGraph` and `Block` defined in the [analysis module](../../src/oqd_core/analysis/utils/control_flow.py).

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

///

The output of the type checker is stored in `type_checker.result`.
