# Analysis

The analog analysis pipeline is implemented in `src/oqd_core/analysis/analog`. The different analysis provided for analog interface are:

- `DominatorAnalysis` and `PostDominatorAnalysis` in [`dominator.py`](../../src/oqd_core/analysis/dominator.py)
- `AnalogTypeChecker` in [`type_checker.py`](../../src/oqd_core/analysis/analog/type_checker.py)
- `ReachingDefinition` and `AvailableVariableAnalysis` in [`reaching_def.py`](../../src/oqd_core/analysis/analog/reaching_def.py)
- `ConstantFolding` in [`fold.py`](../../src/oqd_core/analysis/analog/fold.py)
- `DimensionChecker` in [`dim_checker.py`](../../src/oqd_core/analysis/analog/dim_checker.py)
- `AnalogBoundsChecker` in [`bounds_checker.py`](../../src/oqd_core/analysis/analog/bounds_checker.py)

[`types.py`](../../src/oqd_core/analysis/analog/types.py) contains the types used by the language, and the function signatures supported in the analog interface.

## Usage of Analog Analysis

<!-- prettier-ignore -->
/// admonition | Example
    type: example

```py
from oqd_core.compiler.analog import compile_analog_circuit

source = """
q = qreg(2)
h = %X @ %I
evolve(h, 1.0, q)
measure(q)
"""

circuit, cfg, analysis_results = compile_analog_circuit(source, return_analysis=True)
```

///

