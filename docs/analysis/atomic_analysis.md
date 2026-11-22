# Analysis

The atomic analysis pipeline is implemented in `src/oqd_core/analysis/atomic`. The different analysis provided for atomic interface are:

- `DominatorAnalysis` and `PostDominatorAnalysis` in [`dominator.py`](../../src/oqd_core/analysis/dominator.py)
- `AtomicTypeChecker` in [`type_checker.py`](../../src/oqd_core/analysis/atomic/type_checker.py)
- `ReachingDefinition` and `AvailableVariableAnalysis` in [`reaching_def.py`](../../src/oqd_core/analysis/atomic/reaching_def.py)
- `ConstantFolding` in [`fold.py`](../../src/oqd_core/analysis/atomic/fold.py)
- `AtomicBoundsChecker` in [`bounds_checker.py`](../../src/oqd_core/analysis/atomic/bounds_checker.py)

[`types.py`](../../src/oqd_core/analysis/atomic/types.py) contains the types used by the language, and the function signatures supported in the atomic interface.


## Usage of Atomic Analysis

<!-- prettier-ignore -->
/// admonition | Example
    type: example

```py
from oqd_core.compiler.atomic import compile_atomic_circuit

source = """
ions = ionreg(1)
mw = beam(2.0, 1.0, 0.0, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
pulse(mw, 1e-5, ions[0], true)
"""

circuit, cfg, analysis_results = compile_atomic_circuit(source, return_analysis=True)
```

///

