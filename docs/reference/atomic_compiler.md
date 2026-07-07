# Atomic Compiler

The atomic compiler is implemented in `src/oqd_core/compiler/atomic`. The entry point is [`compile_atomic_circuit`][oqd_core.compiler.atomic.passes.compile.compile_atomic_circuit].

The compile pipeline:
- Canonicalize scalars, beams, and pulse references over the CFG via [`canonicalize_scalars_cfg`][oqd_core.compiler.atomic.cfg_passes.scalar_env.canonicalize_scalars_cfg]
- Verify pulse durations are constant before protocol canonicalization via [`verify_constant_pulse_durations`][oqd_core.compiler.atomic.verify.passes.verify_constant_pulse_durations]
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
            "canonicalize_math_cfg",
            "canonicalize_math_block",
            "iter_stmt_blocks",
        ]
<!-- prettier-ignore -->
::: oqd_core.compiler.atomic.cfg_passes.scalar_env
    options:
        heading_level: 3
        members: [
            "canonicalize_scalars_cfg",
        ]
<!-- prettier-ignore -->
::: oqd_core.compiler.atomic.cfg_passes.protocol
    options:
        heading_level: 3
        members: [
            "canonicalize_protocol_cfg",
        ]

## Verification Passes

<!-- prettier-ignore -->
::: oqd_core.compiler.atomic.verify.passes
    options:
        heading_level: 3
        members: [
            "verify_constant_pulse_durations",
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

