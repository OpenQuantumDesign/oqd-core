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
from oqd_core.analysis.analog.symbol_table import (
    AnalogSymbolTableBuilder,
)
from oqd_core.analysis.analog.type_checker import AnalogTypeChecker
from oqd_core.analysis.analog.types import AnalogTypeError, TMReg, TQReg
from oqd_core.frontend.analog.AnalogCircuitAST import parse_analog
from oqd_core.interface.analog import Initialize

## Symbol Table ##

def build_symbol_table(program: str):
    circuit = parse_analog(program)
    cfg = AnalogCFGBuilder().run(circuit)
    type_checker = AnalogTypeChecker(cfg)
    symbol_table = AnalogSymbolTableBuilder(
        cfg, type_checker.dataflow_result
    ).symbol_table
    return symbol_table, circuit


class TestAnalogSymbolTable:
    def test_qreg_binding(self):
        symbol_table, circuit = build_symbol_table("r = qreg(3) \n initialize(r)")
        init = next(s for s in circuit.statements if isinstance(s, Initialize))
        env = symbol_table.in_env[symbol_table.stmt_index[id(init)]]
        assert env["r"].target_dim == 3
        assert env["r"].lattice_type is TQReg

    def test_qmode_binding(self):
        symbol_table, circuit = build_symbol_table("s = qmode(2) \n initialize(s)")
        init = next(s for s in circuit.statements if isinstance(s, Initialize))
        env = symbol_table.in_env[symbol_table.stmt_index[id(init)]]
        assert env["s"].target_dim == 2
        assert env["s"].lattice_type is TMReg

    def test_extract_binding(self):
        program = "r = qreg(2) \n q = r[0] \n initialize(q)"
        symbol_table, circuit = build_symbol_table(program)
        init = next(s for s in circuit.statements if isinstance(s, Initialize))
        env = symbol_table.in_env[symbol_table.stmt_index[id(init)]]
        assert env["q"].target_dim == 1
        
    def test_target_list_binding(self):
        program = (
            "r = qreg(3) \n"
            "target = [r[0], r[1], r[2]] \n"
            "initialize(target)"
        )
        symbol_table, circuit = build_symbol_table(program)
        init = next(s for s in circuit.statements if isinstance(s, Initialize))
        env = symbol_table.in_env[symbol_table.stmt_index[id(init)]]
        assert env["target"].target_dim == 3
        

## Control Flow Graph ##

class TestAnalogCFG:
    def test_analog_cfg(self):
        program = "r = qreg(3) \n x = 1"
        circuit = parse_analog(program)
        cfg = AnalogCFGBuilder().run(circuit)
        assert cfg is not None
        

## Type Checker ##

class TestAnalogTypeChecker:
    @pytest.mark.parametrize(
        "program",
        [   "r = qreg(2) \n initialize(r)",
            "r = qreg(2) \n measure(r)",
            "r = qreg(2) \n evolve(%X, 1.0, r)",
            "s = 5 * 4",
            "s = 5 + 2",
            "s = 5 - 2",
            "s = 6 / 2",
            "s = 2 ^ 3",
            "s = #omega + 1",
            "s = 1j * 2",
            "s = sin(1)",
            "s = atan2(1, 2)",
            "s = qmode(3) \n initialize(s)",
            "H = %X %* %I",
            "H = 2 %* %X",
            "H = %X %* 2",
            "H = %X %+ %Y",
            "H = %X %- %Y",
            "H = %X %@ %Y",
            "cond = true and false",
            "cond = true and false \n if (cond) {t = 0.2}",
            "cond = true or false \n while (cond) {t = 0.2}",
            "r = qreg(3) \n target = [r[0], r[1], r[2]] \n initialize(target)",
            "r = qreg(2) \n q = r[0] \n measure(q)",
            "s = qmode(2) \n m = s[0] \n initialize(m)",
            "s = qmode(3) \n evolve(%C %* %A, 1.0, s)",
            "c = 1 < 2",
            "c = 3 >= 2",
            "if (1 < 2) {x = 0}",
            "if (5 <= 4) {s = true}",
            "c = 1 == 2",
            "c = true != false",
            "c = not true",
            "x = 1 \n x = 2 \n y = x + 1",
            "n = 3 \n while (n > 0) { n = n - 1 }"
        ],
    )
    def test_analog_type_checker(self, program):
        circuit = parse_analog(program)
        cfg = AnalogCFGBuilder().run(circuit)
        AnalogTypeChecker(cfg)
        
    @pytest.mark.parametrize(
        "program",
        [   "initialize(r)",
            "measure(r)",
            "evolve(%X, 1.0, r)",
            "s = 5 \n r = qreg(3) \n target = [r[0], r[1], r[2], s] \n initialize(target)",
            "r = qreg(2) \n evolve(5, 1.0, r)",
            "r = qreg(2) \n evolve(%X, true, r)",
            "s = 5 \n initialize(s)",
            "s = 5 * true",
            "s = 5 + %I",
            "s = 5 - true",
            "s = %X / 2",
            "s = 2 ^ %I",
            "s = sin(true)",
            "s = cos(%X)",
            "s = atan2(1, true)",
            "s = 5 \n x = s[0]",
            "H = %X * %I",
            "H = %X %+ 5",
            "H = %X %@ 2",
            "cond = true and 4",
            "cond = 5 \n if (cond) {t = 0.2}",
            "cond = %I \n while (cond) {t = 0.2}",
            "c = true \n measure(c)",
            "c = 1 == true",
            "c = true != 5",
            "c = %X == %Y",
            "c = not 5",
            "c = ! %I",
            "c = 5 and true",
            "c = true or %I",
            "c = true < false",
            "c = true \n x = c[0]",
        ],
    )
    def test_analog_type_checker_error(self, program):
        circuit = parse_analog(program)
        with pytest.raises(AnalogTypeError):
            cfg = AnalogCFGBuilder().run(circuit)
            AnalogTypeChecker(cfg)



