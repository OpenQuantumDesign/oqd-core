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

from oqd_core.analysis.atomic.cfg import AtomicCFGBuilder
from oqd_core.analysis.atomic.symbol_table import (
    AtomicSymbolError,
    AtomicSymbolTableBuilder,
)
from oqd_core.analysis.atomic.type_checker import AtomicTypeChecker
from oqd_core.analysis.atomic.types import AtomicTypeError
from oqd_core.compiler.atomic.verify.passes import verify_pulse_target_dim
from oqd_core.frontend.atomic.AtomicCircuitAST import parse_atomic
from oqd_core.interface.atomic import Pulse

## Symbol Table ##

def build_symbol_table(program: str):
    circuit = parse_atomic(program)
    cfg = AtomicCFGBuilder().run(circuit)
    type_checker = AtomicTypeChecker(cfg)
    symbol_table = AtomicSymbolTableBuilder(
        cfg, type_checker.dataflow_result
    ).symbol_table
    return symbol_table, cfg, circuit


class TestAtomicSymbolTable:
    def test_ionreg_binding(self):
        symbol_table, _, circuit = build_symbol_table(
            "r = ionreg(3)\n"
            "pulse(beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), 1e-5, r, true)"
        )
        pulse = next(s for s in circuit.statements if isinstance(s, Pulse))
        node_id = symbol_table.stmt_index[id(pulse)]
        assert symbol_table.in_env[node_id]["r"].target_dim == 3
        
    def test_extract_out_of_range(self):
        symbol_table, cfg, _ = build_symbol_table(
            "r = ionreg(2)\n"
            "pulse(beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), 1e-5, r[2], true)"
        )
        with pytest.raises(AtomicSymbolError):
            verify_pulse_target_dim(cfg, symbol_table)


## Control Flow Graph ##

class TestAtomicCFG:
    def test_atomic_cfg(self):
        program = "r = ionreg(3) \n x = 1"
        circuit = parse_atomic(program)
        cfg = AtomicCFGBuilder().run(circuit)
        assert cfg is not None
        

## Type Checker ##

class TestAtomicTypeChecker:
    @pytest.mark.parametrize(
        "program",
        [   "r = ionreg(2) \n pulse(beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), 1e-5, r, true)",
            "beam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])",
            "r = ionreg(2) \n beam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n \
            parallel {\n pulse(beam_mw, 5e-6, r[0])\n pulse(beam_mw, 5e-6, r[1])}",
            "s = 5 * 4",
            "s = 5 + 2",
            "s = 5 - 2",
            "s = 6 / 2",
            "s = 2 ^ 3",
            "s = #omega + 1",
            "s = 1j * 2",
            "s = sin(1)",
            "s = atan2(1, 2)",
            "cond = true and false",
            "cond = true and false \n if (cond) {t = 0.2}",
            "cond = true or false \n while (cond) {t = 0.2}",
            "r = ionreg(3) \n target = [r[0], r[1], r[2]]",
            "if (5 <= 4) {s = true}",
            "r = ionreg(2) \n beam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n \
            serial {\n parallel {\n pulse(beam_mw, 5e-6, r[0])\n pulse(beam_mw, 5e-6, r[1])\n }\n}",
            "r = ionreg(3) \n beam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n \
            p0 = pulse(beam_mw, 5e-6, r[0]) \n p1 = pulse(beam_mw, 5e-6, r[1]) \n \
            p2 = pulse(beam_mw, 5e-6, r[2]) \n parallel {\n p0\n serial {\n p1\n p2 \n}\n}",
            "r = ionreg(2) \n b = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n \
            pulse(b, 1e-5, r[0], false)",
            "r = ionreg(2) \n beam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n \
            parallel {\n serial {\n x = pulse(beam_mw, 5e-6, r[0])\n x\n }\n}",
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
    def test_atomic_type_checker(self, program):
        circuit = parse_atomic(program)
        cfg = AtomicCFGBuilder().run(circuit)
        AtomicTypeChecker(cfg)
        
    @pytest.mark.parametrize(
        "program",
        [   "r = ionreg(2) \n pulse(beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), true, r, true)",
            "s = 5 * true",
            "s = 5 - true",
            "s = sin(true)",
            "s = atan2(1, true)",
            "s = 5 \n x = s[0]",
            "cond = true and 4",
            "cond = 5 \n if (cond) {t = 0.2}",
            "s = 5 \n r = ionreg(3) \n target = [r[0], r[1], r[2], s] \n pulse(beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), 1e-5, target, true)",
            "r = ionreg(3) \n parallel {pulse(beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), true, ionreg(2), true) \n r[0]}",
            "r = ionreg(2) \n beam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n \
            parallel {\n serial {\n pulse(beam_mw, 5e-6, r[0])\n }\n r[1]\n}",
            "r = ionreg(2) \n beam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n \
            parallel {\n serial {\n x = 5\n }\n}",
            "r = true \n if (r) { \n r = 5} \n r = r + 2",
            "c = 1 == true",
            "c = true != 5",
            "c = not 5",
            "c = 5 and true",
            "c = true < false",
            "c = true \n x = c[0]",
        ],
    )
    def test_atomic_type_checker_error(self, program):
        circuit = parse_atomic(program)
        with pytest.raises(AtomicTypeError):
            cfg = AtomicCFGBuilder().run(circuit)
            AtomicTypeChecker(cfg)
        
