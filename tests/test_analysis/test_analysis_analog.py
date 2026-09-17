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

from oqd_core.analysis.analog.bounds_checker import (
    AnalogBoundsChecker,
    AnalogOutOfBoundsError,
)
from oqd_core.analysis.analog.cfg import AnalogCFGBuilder
from oqd_core.analysis.analog.definite_assignment import (
    AnalogDefiniteAssignmentChecker,
    AnalogUndefinedVarError,
)
from oqd_core.analysis.analog.symbol_table import (
    AnalogSymbolTableBuilder,
)
from oqd_core.analysis.analog.type_checker import AnalogTypeChecker
from oqd_core.analysis.analog.types import AnalogTypeError
from oqd_core.analysis.utils.control_flow import Accumulator
from oqd_core.frontend.analog.AnalogCircuitAST import parse_analog

## Symbol Table ##

def build_symbol_table(program: str):
    circuit = parse_analog(program)
    cfg = AnalogCFGBuilder().run(circuit)
    cfg = Accumulator()(cfg)
    type_checker = AnalogTypeChecker(cfg)
    symbol_table = AnalogSymbolTableBuilder(
        cfg, type_checker.dataflow_result
    ).symbol_table
    return symbol_table, circuit


# class TestAnalogSymbolTable:
#     def test_qreg_binding(self):
#         symbol_table, circuit = build_symbol_table("r = qreg(3) \n initialize(r)")
#         init = next(s for s in circuit.statements if isinstance(s, Initialize))
#         env = symbol_table.in_env[symbol_table.stmt_index[id(init)]]
#         assert env["r"].target_dim == 3
#         assert env["r"].lattice_type is TQReg

#     def test_qmode_binding(self):
#         symbol_table, circuit = build_symbol_table("s = qmode(2) \n initialize(s)")
#         init = next(s for s in circuit.statements if isinstance(s, Initialize))
#         env = symbol_table.in_env[symbol_table.stmt_index[id(init)]]
#         assert env["s"].target_dim == 2
#         assert env["s"].lattice_type is TMReg

#     def test_extract_binding(self):
#         program = "r = qreg(2) \n q = r[0] \n initialize(q)"
#         symbol_table, circuit = build_symbol_table(program)
#         init = next(s for s in circuit.statements if isinstance(s, Initialize))
#         env = symbol_table.in_env[symbol_table.stmt_index[id(init)]]
#         assert env["q"].target_dim == 1
        
#     def test_target_list_binding(self):
#         program = (
#             "r = qreg(3) \n"
#             "target = [r[0], r[1], r[2]] \n"
#             "initialize(target)"
#         )
#         symbol_table, circuit = build_symbol_table(program)
#         init = next(s for s in circuit.statements if isinstance(s, Initialize))
#         env = symbol_table.in_env[symbol_table.stmt_index[id(init)]]
#         assert env["target"].target_dim == 3
        

## Control Flow Graph ##

class TestAnalogCFG:
    def test_analog_cfg(self):
        program = "r = qreg(3) \n x = 1"
        circuit = parse_analog(program)
        cfg = AnalogCFGBuilder().run(circuit)
        assert cfg is not None
        

## Definite Assignment ##

class TestAnalogAssignmentAnalysis:
    @pytest.mark.parametrize(
        "program",
        [   "r = qreg(2) \n initialize(r)",
            "r = qreg(2) \n initialize(r) \n measure(r)",
            "r = qreg(2) \n evolve(%X, 1.0, r[1])",
            "s = 5 * 4 \n t = s * 4",
            "s = 5 + 2 \n t = s + 4",
            "s = 5 - 2 \n t = s - 4",
            "s = 6 / 2 \n t = s / 3",
            "s = 2 ^ 3 \n t = s ^ 2",
            "pi = 3.14159 \n s = sin(pi)",
            "pi = 3.14159 \n s = atan2(1, pi)",
            "s = qmode(3) \n initialize(s[1])",
            "r = qreg(2) \n H = %X %* %I \n evolve(H, 1, r[0])",
            "r = qreg(2) \n H = %X %@ %Y \n evolve(H, 1, r)",
            "cond = true and false \n if (cond) {\n a = 0 \n }",
            "cond = true or false \n while (cond) {t = 0.2}",
            "r = qreg(3) \n target = [r[0], r[1], r[2]] \n initialize(target)",
            "s = qmode(2) \n evolve(%C %@ %A, 1.0, s)",
            "c = 1 < 2 \n if (c) { \n a = 2 \n }",
            "c = 3 >= 2 \n if (c) { \n c = true \n }",
            "a = true \n c = not a \n if (c) { \n c = a \n }",
            "x = 1 \n x = 2 \n y = x + 1",
            "n = 3 \n while (n > 0) { n = n - 1 }",
            "a = [1 , 2 , 3] \n b = a[1] \n e = [a, b]"
        ]
    )
    def test_analog_definite_assignment_checker(self, program):
        circuit = parse_analog(program)
        cfg = AnalogCFGBuilder().run(circuit)
        AnalogDefiniteAssignmentChecker(cfg)
    
    @pytest.mark.parametrize(
        "program",
        [   "initialize(r)",
            "measure(r)",
            "evolve(%X, 1.0, r)",
            "a = 0 \n b = a \n if (not a) { \n b = c \n}",
            "a = 0 \n b = a \n if (not c) { \n a = 1 \n}",
            "a = 0 \n b = a \n while (b) { \n b = c + 1 \n}",
            "a = 0 \n b = a \n while (d) { \n b = b + 1 \n}",
            "a = 0 \n b = a \n while (d) { \n b = d + 1 \n}",
            "a = [1, 2, a]",
            "a = 0 \n b = 2 \n c = [a, b, c]",
            "a = not b",
            "a = a == b",
            "a = %X \n b = %Y %* c",
            "a = 1 \n b = a + c",
            "a = a > 0",
            "a = 1 \n b = a < b",
            "pi = 3.14159 \n b = sin(pi * a)"
        ]
    )
    def test_analog_definite_assignment_checker_error(self, program):
        circuit = parse_analog(program)
        with pytest.raises(AnalogUndefinedVarError):
            cfg = AnalogCFGBuilder().run(circuit)
            AnalogDefiniteAssignmentChecker(cfg)


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
            "n = 3 \n while (n > 0) { n = n - 1 }",
        ],
    )
    def test_analog_type_checker(self, program):
        circuit = parse_analog(program)
        cfg = AnalogCFGBuilder().run(circuit)
        AnalogTypeChecker(cfg)
        
    @pytest.mark.parametrize(
        "program",
        [   "s = 5 \n r = qreg(3) \n target = [r[0], r[1], r[2], s] \n initialize(target)",
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


## Bounds Checker ##

class TestAnalogBoundsChecker:
    @pytest.mark.parametrize(
        "program",
        [   "r = qreg(2) \n [r[0], r[1]]",
            "r = qreg(3) \n initialize(r) \n measure(r)",
            "r = qreg(10) \n evolve(%X, 1.0, r[9])",
            "l = [5 * 4, 3 * 5] \n t = l[0]",
            "s = [5 + 2, 3] \n t = s[1]",
            "s = [5 - 2, 10] \n t = [1, 2, s]",
            "s = [6 / 2, 1] \n t = [s[0], 5]",
            "s = [[2 ^ 3], [4]] \n t = s[0] \n u = [s[1], t]",
            "pi = 3.14159 \n s = sin(pi) \n c = cos(pi) \n a = [s, c] \n tan(a[0])",
            "a = [1, 0] \n atan2(a[0], a[1])",
            "s = qmode(3) \n initialize(s[1])",
            "r = qreg(2) \n H = [%X %* %I] \n evolve(H[0], 1, r[0])",
            "cond = [true, false] \n if (cond[0]) {\n a = 0 \n }",
            # "cond = [true, false] \n while (cond[0]) {t = 0.2}",
            "c = [1 < 2, 2 <= 4, 4 == 5] \n if (c[1]) { \n a = 2 \n }",
            "c = [3 >= 2, 1 > 0]  \n if (c[0]) { \n c = true \n }",
            "a = true \n c = [a, not a] \n if (c[0]) { \n c = a \n }",
            "a = [1 , 2 , 3] \n b = a[1] \n e = [a, b] \n e[0]",
        ]
    )
    def test_analog_bounds_checker(self, program):
        circuit = parse_analog(program)
        cfg = AnalogCFGBuilder().run(circuit)
        AnalogBoundsChecker(cfg)
    
    @pytest.mark.parametrize(
        "program",
        [   "r = qreg(1) \n r[1]"
            "r = qreg(2) \n [r[0], r[2]]",
            "r = qreg(3) \n initialize(r[3]) \n measure(r)",
            "r = qreg(3) \n initialize(r) \n measure(r[4])",
            "r = qreg(10) \n evolve(%X, 1.0, r[10])",
            "l = [5 * 4, 3 * 5] \n t = l[3]",
            "s = [5 + 2, 3] \n t = s[4]",
            "s = [5 - 2, 10] \n t = [1, 2, s] \n u = t[4]",
            "s = [6 / 2, 1] \n t = [s[3], 5]",
            "s = [[2 ^ 3], [4]] \n t = s[0] \n u = [s[1], t[0]]",
            "pi = 3.14159 \n s = sin(pi) \n c = cos(pi) \n a = [s, c] \n tan(a[2])",
            "a = [1, 0] \n atan2(a[0], a[2])",
            "s = qmode(3) \n initialize(s[3])",
            "r = qreg(2) \n H = [%X %* %I] \n evolve(H[1], 1, r[0])",
            "cond = [true, false] \n if (cond[2]) {\n a = 0 \n }",
            "cond = [true, false] \n while (cond[2]) {t = 0.2}",
            "c = [1 < 2, 2 <= 4, 4 == 5] \n if (c[3]) { \n a = 2 \n }",
            "c = [3 >= 2, 1 > 0]  \n if (c[2]) { \n c = true \n }",
            "a = true \n c = [a, not a] \n if (c[2]) { \n c = a \n }",
            "a = [1 , 2 , 3] \n b = a[1] \n e = [a, b] \n e[3]",
        ]
    )
    def test_analog_bounds_checker_error(self, program):
        circuit = parse_analog(program)
        with pytest.raises(AnalogOutOfBoundsError):
            cfg = AnalogCFGBuilder().run(circuit)
            AnalogBoundsChecker(cfg)


## Accumulator ##

class TestAnalogAccumulator:
    @pytest.mark.parametrize(
        "program",
        [
            "1",
            "x = 0 \n while (true) { \n x = x + 2 \n if (x > 3) { \n break \n } \n else { \n a = 1\n } \n}",
            "if (5 == 2) { \n a = 1}",
            "a = 1 \n b = 2 \n r = qreg(3) \n p = 4 \n d = sin(3.141592)",
            "a = 5 \n b = 3 \n if (a > b) { \n if (b > 0) { \n initialize(targets) \n} \n else { \n measure(r) \n } \n }",
            "a = 2 \n b = 3 \n if (a > 2) { \n a = 3 \n if (a==b) { \n b = 2 \n } \n else { \n b = 5 \n } \n if (b < a) { \n b = 5 \n }\n } \n c = a + b \n c"
        ],
    )
    def test_analog_accumulator_simple(self, program):
        circuit = parse_analog(program)
        single_stmt_block_cfg = AnalogCFGBuilder().run(circuit)
        multiple_stmts_block_cfg = Accumulator()(single_stmt_block_cfg)
        assert(len(multiple_stmts_block_cfg.blocks) <= len(single_stmt_block_cfg.blocks))
    
    @pytest.mark.parametrize(
        "program",
        [
            "1",
            "x = 0 \n while (true) { \n x = x + 2 \n if (x > 3) { \n break \n } \n else { \n a = 1\n } \n}",
            "if (5 == 2) { \n a = 1}",
            "a = 5 \n if (a > 2) { \n if (a > 0) { \n initialize(targets) \n} \n else { \n measure(r) \n } \n }",
        ],
    )
    def test_analog_accumulator_does_nothing(self, program):
        circuit = parse_analog(program)
        single_stmt_block_cfg = AnalogCFGBuilder().run(circuit)
        multiple_stmts_block_cfg = Accumulator()(single_stmt_block_cfg)
        assert(len(multiple_stmts_block_cfg.blocks) == len(single_stmt_block_cfg.blocks))
    
    @pytest.mark.parametrize(
        "program",
        [
            "a = 0 \n x = 2",
            "a = 1 \n b = 2 \n r = qreg(3) \n p = 4 \n d = sin(3.141592)",
            "r = qreg(5) \n initialize(r) \n evolve(%X, 1, r[0])"
        ],
    )
    def test_analog_accumulator_single_block(self, program):
        circuit = parse_analog(program)
        single_stmt_block_cfg = AnalogCFGBuilder().run(circuit)
        assert(len(single_stmt_block_cfg.blocks) > 3)
        multiple_stmts_block_cfg = Accumulator()(single_stmt_block_cfg)
        assert(len(multiple_stmts_block_cfg.blocks) == 3)
    


