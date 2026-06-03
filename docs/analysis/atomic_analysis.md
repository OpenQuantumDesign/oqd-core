
### Type Checker

The atomic type checker is implemented by the `AtomicTypeChecker` class in [`type_checker.py`](../../src/oqd_core/analysis/atomic/type_checker.py). The `AtomicTypeLattice` class defines a concrete lattice for atomic types with `leq`, `join`, and `meet` methods. The type checker builds a Control Flow Graph (CFG) from the typed AST, and runs the forward dataflow analysis on the CFG. The atomic Control Flow Graph (CFG) is implemented by the `AtomicCFGBuilder` class in [`cfg.py`](../../src/oqd_core/analysis/atomic/cfg.py). This CFGBuilder uses `ControlFlowGraph` and `CFGNode` defined in the [analysis module](../../src/oqd_core/analysis/utils.py).


## Usage of the type checker

<!-- prettier-ignore -->
/// admonition | Example
    type: example

```py
from oqd_core.frontend.atomic import parse_atomic
from oqd_core.analysis.atomic.cfg import AtomicCFGBuilder
from oqd_core.analysis.atomic.type_checker import AtomicTypeChecker

source = """
ions = ionreg(1)
mw = beam(2.0, 1.0, 0.0, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
pulse(mw, 1e-5, ions[0], true)
"""

circuit = parse_atomic(source)
cfg = AtomicCFGBuilder().run(circuit)
type_checker = AtomicTypeChecker(cfg)
```

///

The output of the type checker is stored in `type_checker.result`.

## Strongly Connected Components (SCC)

The `AtomicSCC` class identifies the strongly connected components of the CFG to check for infinite loops in the program. The implementation follows [Tarjan's algorithm](https://www.geeksforgeeks.org/dsa/tarjan-algorithm-find-strongly-connected-components/).
