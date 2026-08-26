# Copyright 2024-2025 Open Quantum Design

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from oqd_core.analysis.analog.cfg import AnalogCFGBuilder
from oqd_core.analysis.analog.symbol_table import AnalogSymbolTableBuilder
from oqd_core.analysis.analog.type_checker import AnalogTypeChecker
from oqd_core.compiler.analog.cfg_passes.walk import (
    canonicalize_math_cfg,
    canonicalize_operators_cfg,
    iter_stmt_blocks,
)
from oqd_core.compiler.analog.error import AnalogCompilerError
from oqd_core.compiler.analog.math.passes import evaluate_math_expr
from oqd_core.compiler.analog.passes.compile import compile_analog_circuit
from oqd_core.frontend.analog.AnalogCircuitAST import parse_analog
from oqd_core.interface.analog import Declaration, Evolve, MathNum
from oqd_core.interface.analog.expr import OperatorAdd, OperatorMul


def build_inputs(program: str):
    circuit = parse_analog(program)
    cfg = AnalogCFGBuilder().run(circuit)
    type_checker = AnalogTypeChecker(cfg)
    symbol_table = AnalogSymbolTableBuilder(
        cfg, type_checker.dataflow_result
    ).symbol_table
    return circuit, cfg, symbol_table


def declaration_value(cfg, name):
    for _, block in iter_stmt_blocks(cfg):
        stmt = block.stmt
        if isinstance(stmt, Declaration) and stmt.name == name:
            return stmt.value
    raise KeyError(name)


## Compile ##

class TestAnalogCompile:
    @pytest.mark.parametrize(
        "program",
        [
            "r = qreg(2)\n evolve(%X %@ %I, 1.0, r)",
            "r = qreg(2)\n h = %X %@ %I %+ %Y %@ %I\n evolve(h, 1.0, r)",
        ],
    )
    def test_compile(self, program):
        circuit, cfg, symbol_table = build_inputs(program)
        compile_analog_circuit(circuit, cfg, symbol_table)

## CFG Passes ##

class TestAnalogCanonicalizeMathCfg:
        
    def test_evolve_duration(self):
        _, cfg, _ = build_inputs("r = qreg(2)\n evolve(%X, 1.0 + 1.0, r)")
        canonicalize_math_cfg(cfg)
        for _, block in iter_stmt_blocks(cfg):
            if isinstance(block.stmt, Evolve):
                assert evaluate_math_expr(block.stmt.duration) == 2
    
    def test_skips_operator_declarations(self):
        _, cfg, _ = build_inputs("h = %X %+ %Y")
        canonicalize_math_cfg(cfg)
        assert isinstance(declaration_value(cfg, "h"), OperatorAdd)


class TestAnalogCanonicalizeOperatorsCfg:
    def test_canonicalizes_operator_declarations(self):
        _, cfg, _ = build_inputs("h = %X %+ %Y")
        canonicalize_operators_cfg(cfg)
        h = declaration_value(cfg, "h")
        assert isinstance(h, OperatorAdd)
        assert isinstance(h.op1, OperatorMul)
        assert isinstance(h.op2, OperatorMul)
        assert h.op1.op1 == MathNum(value=1)
        assert h.op2.op1 == MathNum(value=1)

    def test_skips_math_declarations(self):
        _, cfg, _ = build_inputs("s = 2 * 3")
        canonicalize_operators_cfg(cfg)
        assert not isinstance(declaration_value(cfg, "s"), MathNum)

