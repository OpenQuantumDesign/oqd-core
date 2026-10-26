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
from oqd_compiler_infrastructure import (
    CFG,
    CFGBlockAccumulator,
    Chain,
    RelabelCFGBlocks,
)

from oqd_core.compiler.atomic.conversion import AtomicCFGBuilder
from oqd_core.analysis.atomic.type_checker import AtomicTypeChecker
from oqd_core.frontend.atomic import parse_atomic


from oqd_core.analysis.atomic.types import (
    AtomicTypeError,
    TBool,
    TComplex,
    TFloat,
    TInt,
    TList,
    TIReg,
    TIRegElem,
    TPulse,
    TBeam,
)


@pytest.fixture()
def cfg_builder():

    passes = Chain(
        parse_atomic,
        AtomicCFGBuilder(),
        CFGBlockAccumulator(),
        RelabelCFGBlocks(),
    )

    return passes


@pytest.fixture()
def type_checker():
    return AtomicTypeChecker()



# class TestAtomicSymbolTable:
#     def test_ionreg_binding(self):
#         symbol_table, _, circuit = build_symbol_table(
#             "r = ionreg(3)\n"
#             "pulse(beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), 1e-5, r, true)"
#         )
#         pulse = next(s for s in circuit.statements if isinstance(s, Pulse))
#         node_id = symbol_table.stmt_index[id(pulse)]
#         assert symbol_table.in_env[node_id]["r"].target_dim == 3

#     def test_extract_out_of_range(self):
#         symbol_table, cfg, _ = build_symbol_table(
#             "r = ionreg(2)\n"
#             "pulse(beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), 1e-5, r[2], true)"
#         )
#         with pytest.raises(AtomicSymbolError):
#             verify_pulse_target_dim(cfg, symbol_table)


## Control Flow Graph ##


class TestAtomicCFG:
    def test_atomic_cfg(self):
        program = "r = ionreg(3) \n x = 1"
        circuit = parse_atomic(program)
        cfg = AtomicCFGBuilder()(circuit)
        assert cfg is not None


## Type Checker ##


class TestAtomicTypeChecker:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("x=1", {0: {}, 1: {"x": TInt}, 2: {"x": TInt}}),
            (
                "x=1 \n y = 2.0",
                {0: {}, 1: {"x": TInt, "y": TFloat}, 2: {"x": TInt, "y": TFloat}},
            ),
            (
                "x = 1 \n while (x<10) {x = x + 0.1}",
                {
                    0: {},
                    1: {"x": TInt},
                    2: {"x": TFloat},
                    3: {"x": TFloat},
                    4: {"x": TFloat},
                },
            ),
            (
                "x = 1 \n while (real(x)<10) {x = x + 0.1 * 1j}",
                {
                    0: {},
                    1: {"x": TInt},
                    2: {"x": TComplex},
                    3: {"x": TComplex},
                    4: {"x": TComplex},
                },
            ),
        ],
    )
    def test_atomic_type_checker(
        self,
        source,
        expected,
        cfg_builder,
        type_checker,
    ):
        cfg = cfg_builder(source)

        type_result = type_checker.analyze(cfg).out_states

        assert type_result == expected

    @pytest.mark.parametrize(
        "program",
        [
            "r = ionreg(2) \n pulse(beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), 1e-5, r, true)",
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
            "n = 3 \n while (n > 0) { n = n - 1 }",
        ],
    )
    def test_atomic_type_checker_pass(self, program):
        circuit = parse_atomic(program)
        cfg = AtomicCFGBuilder()(circuit)
        AtomicTypeChecker().analyze(cfg)

    @pytest.mark.parametrize(
        "program",
        [
            "r = ionreg(2) \n pulse(beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), true, r, true)",
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
            "parallel {\n x = 2}",
            "serial {\n x = 5}",
            "a = 5 \n parallel { \n a}",
            "a = 3 \n serial { \n a}",
            "parallel {\n serial { \n a = 4 * 5 \n b = 3 + 5}}",
            "serial {\n parallel { \n a = 4 * 5 \n b = 3 + 5}}",
        ],
    )
    def test_atomic_type_checker_error(self, program):
        circuit = parse_atomic(program)
        cfg = AtomicCFGBuilder()(circuit)
        with pytest.raises(AtomicTypeError):
            AtomicTypeChecker().analyze(cfg)
