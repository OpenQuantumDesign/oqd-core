# Atomic Compiler

The atomic compiler is implemented in `src/oqd_core/compiler/atomic`. The entry point is [`compile_atomic_circuit`][oqd_core.compiler.atomic.passes.compile.compile_atomic_circuit].

The compile pipeline:
- Canonicalize scalars and beams over the CFG via [`canonicalize_declarations_cfg`][oqd_core.compiler.atomic.cfg_passes.walk.canonicalize_declarations_cfg]
- Canonicalize nested protocols and relative time via [`canonicalize_protocol_cfg`][oqd_core.compiler.atomic.cfg_passes.protocol.canonicalize_protocol_cfg]
- Verify pulse target dimensions via [`verify_pulse_target_dim`][oqd_core.compiler.atomic.verify.passes.verify_pulse_target_dim]

## Compile Passes

<!-- prettier-ignore -->
::: oqd_core.compiler.atomic.passes.compile
    options:
        heading_level: 3
        members: [
            "compile_atomic_circuit",
        ]

## CFG Passes

<!-- prettier-ignore -->
::: oqd_core.compiler.atomic.cfg_passes.walk
    options:
        heading_level: 3
        members: [
            "canonicalize_expr",
            "canonicalize_beam",
            "iter_stmt_blocks",
            "canonicalize_scalars_cfg",
        ]
<!-- prettier-ignore -->
::: oqd_core.compiler.atomic.cfg_passes.protocol
    options:
        heading_level: 3
        members: [
            "canonicalize_protocol_cfg",
        ]
<!-- prettier-ignore -->
::: oqd_core.compiler.atomic.cfg_passes.resolve
    options:
        heading_level: 3
        members: [
            "resolve_scalar_expr",
            "resolve_beam_expr",
            "resolve_beam_ref",
            "resolve_pulse_expr",
            "resolve_pulse_ref",
            "resolve_protocol_pulses",
        ]

## Verification Passes

<!-- prettier-ignore -->
::: oqd_core.compiler.atomic.verify.passes
    options:
        heading_level: 3
        members: [
            "verify_pulse_target_dim",
        ]

## Atomic Math Passes

<!-- prettier-ignore -->
::: oqd_core.compiler.atomic.math.passes
    options:
        heading_level: 3
        members: [
            "canonicalize_math_expr",
            "evaluate_math_expr",
            "simplify_math_expr",
            "print_math_expr",
        ]

## Atomic Math Rewrite Rules

<!-- prettier-ignore -->
::: oqd_core.compiler.atomic.math.rules
    options:
        heading_level: 3
        members: [
            "DistributeMathExpr",
            "PartitionMathExpr",
            "ProperOrderMathExpr",
            "PruneMathExpr",
            "EvaluateMathExpr",
            "SimplifyMathExpr",
            "PrintMathExpr",
        ]

## Usage


<!-- prettier-ignore -->
/// admonition | Example
    type: example
```py
from oqd_core.frontend.atomic import parse_atomic
from oqd_core.analysis.atomic import AtomicCFGBuilder, AtomicTypeChecker, AtomicSymbolTableBuilder
from oqd_core.compiler.atomic.passes.compile import compile_atomic_circuit

source = """
ions = ionreg(1)
mw = beam(2.0, 1.0, 0.0, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
pulse(mw, 1e-5, ions[0], true)
"""

circuit = parse_atomic(source)
cfg = AtomicCFGBuilder().run(circuit)
type_checker = AtomicTypeChecker(cfg)
symbol_table = AtomicSymbolTableBuilder(cfg, type_checker.dataflow_result).symbol_table
circuit = compile_atomic_circuit(circuit, cfg, type_checker.dataflow_result, symbol_table)
```
