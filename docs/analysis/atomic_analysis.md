# Analysis

The atomic analysis pipeline is implemented in `src/oqd_core/analysis/atomic`.

## Control Flow Graph

The atomic Control Flow Graph (CFG) is implemented by the `AtomicCFGBuilder` class in [`cfg.py`](../../src/oqd_core/analysis/atomic/cfg.py). This CFGBuilder uses `ControlFlowGraph` and `Block` defined in the [analysis module](../../src/oqd_core/analysis/utils/control_flow.py).

## Type Checker

The atomic type checker is implemented by the `AtomicTypeChecker` class in [`type_checker.py`](../../src/oqd_core/analysis/atomic/type_checker.py). The `AtomicTypeLattice` class defines a concrete lattice for atomic types with `leq`, `join`, and `meet` methods. Type inference rules are implemented by the `AtomicSemantics` class in [`semantics.py`](../../src/oqd_core/analysis/atomic/semantics.py). The type checker runs forward dataflow analysis over the CFG.


## Usage of the type checker

<!-- prettier-ignore -->
/// admonition | Example
    type: example

```py
from oqd_core.frontend.atomic import parse_atomic
from oqd_core.analysis.atomic import AtomicCFGBuilder, AtomicTypeChecker

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

The output of the type checker is stored in `type_checker.dataflow_result`.
