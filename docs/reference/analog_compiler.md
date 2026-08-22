# Analog Compiler

The analog compiler is implemented in `src/oqd_core/compiler/analog`. The entry point is [`compile_analog_circuit`][oqd_core.compiler.analog.passes.compile.compile_analog_circuit], which takes an [`AnalogCircuit`][oqd_core.interface.analog.circuit.AnalogCircuit], a [`ControlFlowGraph`][oqd_core.analysis.utils.control_flow.ControlFlowGraph], and an `AnalogSymbolTable`[oqd_core.analysis.analog.symbol_table.AnalogSymbolTable].

The compile pipeline:
- Canonicalize operators over the CFG via [`canonicalize_operators_cfg`][oqd_core.compiler.analog.cfg_passes.walk.canonicalize_operators_cfg]
- Canonicalize math expressions over the CFG via [`canonicalize_math_cfg`][oqd_core.compiler.analog.cfg_passes.walk.canonicalize_math_cfg]
- Verify register access and Hamiltonian target dimensions
- Infer Hilbert space dimensions from canonicalized `Evolve` statements

## Compile Passes

<!-- prettier-ignore -->
::: oqd_core.compiler.analog.passes.compile
    options:
        heading_level: 3
        members: [
            "compile_analog_circuit",
        ]

## CFG Passes

<!-- prettier-ignore -->
::: oqd_core.compiler.analog.cfg_passes.walk
    options:
        heading_level: 3
        members: [
            "canonicalize_math_cfg",
            "canonicalize_math_block",
            "canonicalize_operators_cfg",
            "iter_stmt_blocks",
        ]

## Operator Analysis

<!-- prettier-ignore -->
::: oqd_core.compiler.analog.operator.dim
    options:
        heading_level: 3
        members: [
            "operator_dim",
            "coeff_and_op",
            "is_scalar_mul",
        ]
<!-- prettier-ignore -->
::: oqd_core.compiler.analog.operator.term_index
    options:
        heading_level: 3
        members: [
            "TermIndex",
            "term_index",
        ]

## Verification Passes

<!-- prettier-ignore -->
::: oqd_core.compiler.analog.verify.passes
    options:
        heading_level: 3
        members: [
            "verify_register_access_dim",
            "verify_hamiltonian_target_dim",
            "verify_analog_args_dim",
        ]



## Analog Math Passes

<!-- prettier-ignore -->

::: oqd_core.compiler.analog.math.passes
    options:
        heading_level: 3
        members: [
            "canonicalize_math_expr",
            "evaluate_math_expr",
            "simplify_math_expr",
            "print_math_expr",
        ]

## Analog Math Rewrite Rules

<!-- prettier-ignore -->
::: oqd_core.compiler.analog.math.rules
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
from oqd_core.frontend.analog import parse_analog
from oqd_core.analysis.analog import AnalogCFGBuilder, AnalogTypeChecker, AnalogSymbolTableBuilder
from oqd_core.compiler.analog.passes.compile import compile_analog_circuit
source = """
q = qreg(2)
h = %X %@ %I
evolve(h, 1.0, q)
measure(q)
"""
circuit = parse_analog(source)
cfg = AnalogCFGBuilder().run(circuit)
type_checker = AnalogTypeChecker(cfg)
symbol_table = AnalogSymbolTableBuilder(cfg, type_checker.dataflow_result).symbol_table
circuit, cfg, n_qreg, n_qmode = compile_analog_circuit(circuit, cfg, symbol_table)
```
