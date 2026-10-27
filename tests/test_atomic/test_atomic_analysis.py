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

from oqd_core.atomic import (
    AtomicBoundsChecker,
    AtomicCFGBuilder,
    AtomicTypeChecker,
    AtomicTypeError,
    AvailableVariableAnalysis,
    AvailableVariableError,
    OutOfBoundsError,
    parse_atomic,
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


## Control Flow Graph ##


class TestAtomicCFG:
    def test_atomic_cfg(self):
        program = "r = ionreg(3) \n x = 1"
        circuit = parse_atomic(program)
        cfg = AtomicCFGBuilder()(circuit)
        assert cfg is not None


## Available Variables ##

class TestAtomicAssignmentAnalysis:
    @pytest.mark.parametrize(
        "program",
        [
            "r = ionreg(2)",
            "s = 5 * 4 \n t = s * 4",
            "s = 5 + 2 \n t = s + 4",
            "s = 5 - 2 \n t = s - 4",
            "s = 6 / 2 \n t = s / 3",
            "s = 2 ^ 3 \n t = s ^ 2",
            "pi = 3.14159 \n s = sin(pi)",
            "pi = 3.14159 \n s = atan2(1, pi)",
            "cond = true and false \n if (cond) {\n a = 0 \n }",
            "cond = true or false \n while (cond) {t = 0.2}",
            "c = 1 < 2 \n if (c) { \n a = 2 \n }",
            "c = 3 >= 2 \n if (c) { \n c = true \n }",
            "a = true \n c = not a \n if (c) { \n c = a \n }",
            "x = 1 \n x = 2 \n y = x + 1",
            "n = 3 \n while (n > 0) { n = n - 1 }",
            "a = [1 , 2 , 3] \n b = a[1] \n e = [a, b]",
            "beam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])",
            "r = ionreg(1) \n beam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n \
                p0 = pulse(beam_mw, 5e-6, r[0])",
            "a = [1, 2, 4] \n parallel {\n b = a[0] \n c = a[2] }",
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
        ],
    )
    def test_atomic_definite_assignment_checker(self, program):
        circuit = parse_atomic(program)
        cfg = AtomicCFGBuilder()(circuit)
        AvailableVariableAnalysis().analyze(cfg)

    @pytest.mark.parametrize(
        "program",
        [
            "a = 0 \n b = a \n if (not a) { \n b = c \n}",
            "a = 0 \n b = a \n if (not c) { \n a = 1 \n}",
            "a = 0 \n b = a \n while (b) { \n b = c + 1 \n}",
            "a = 0 \n b = a \n while (d) { \n b = b + 1 \n}",
            "a = 0 \n b = a \n while (d) { \n b = d + 1 \n}",
            "a = [1, 2, a]",
            "a = 0 \n b = 2 \n c = [a, b, c]",
            "a = not b",
            "a = a == b",
            "a = 1 \n b = a + c",
            "a = a > 0",
            "a = 1 \n b = a < b",
            "pi = 3.14159 \n b = sin(pi * a)",
            "a = 5 \n parallel {\n b = a + 5 \n c = b + 2}",
            "beam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], a)",
            "r = ionreg(1) \n beam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n \
                p0 = pulse(beam_mw, 5e-6, s[0])",
            "a = [1, 2, 4] \n parallel {\n b = a[0] \n serial {\n c = a[1] + b} }",
            "r = ionreg(3) \n target = [r[0], r[1], p[2]]",
            "if (5 <= 4) {s = b}",
            "r = ionreg(2) \n beam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n \
            serial {\n parallel {\n pulse(beam_mw, 5e-6, r[0])\n pulse(beam_w, 5e-6, r[1])\n }\n}",
            "r = ionreg(3) \n beam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n \
            p0 = pulse(beam_mw, 5e-6, r[0]) \n p1 = pulse(beam_mw, 5e-6, r[1]) \n \
            p2 = pulse(beam_mw, 5e-6, r[2]) \n parallel {\n p0\n serial {\n p1\n p \n}\n}",
            "r = ionreg(2) \n b = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n \
            pulse(b, 1e-5, w[0], false)",
            "r = ionreg(2) \n beam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n \
            parallel {\n serial {\n x = pulse(beam_mw, 5e-6, r[0])\n x\n } \n x\n}",
        ],
    )
    def test_atomic_definite_assignment_checker_error(self, program):
        circuit = parse_atomic(program)
        with pytest.raises(AvailableVariableError):
            cfg = AtomicCFGBuilder()(circuit)
            AvailableVariableAnalysis().analyze(cfg)

########################################################################################


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


## Bounds Checker ##

class TestAtomicBoundsChecker:
    @pytest.mark.parametrize(
        "program",
        [
            "r = ionreg(2)",
            "l = [5 * 4, 3 * 5] \n t = l[0]",
            "s = [5 + 2, 3] \n t = s[1]",
            "s = [5 - 2, 10] \n t = [1, 2, s]",
            "s = [6 / 2, 1] \n t = [s[0], 5]",
            "pi = 3.14159 \n s = sin(pi) \n c = cos(pi) \n a = [s, c] \n tan(a[0])",
            "a = [1, 0] \n atan2(a[0], a[1])",
            "cond = [true, false] \n if (cond[0]) {\n a = 0 \n }",
            "cond = [true, false] \n while (cond[0]) {t = 0.2}",
            "c = [1 < 2, 2 <= 4, 4 == 5] \n if (c[1]) { \n a = 2 \n }",
            "c = [3 >= 2, 1 > 0]  \n if (c[0]) { \n c = true \n }",
            "a = true \n c = [a, not a] \n if (c[0]) { \n c = a \n }",
            "a = [1 , 2 , 3] \n b = a[1] \n e = [a, b] \n e[0]",
            "r = ionreg(3) \n target = [r[0], r[1], r[2]]",
            "r = ionreg(3) \n b = [beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), \
            beam(1e6, 0.5, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])] \n \
            p0 = pulse(b[0], 5e-6, r[0]) \n p1 = pulse(b[1], 5e-6, r[1])",
            "r = ionreg(2) \n b = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n \
            c = beam(1e6, 0.5, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n parallel {\n \
            serial {\n x = pulse(b, 5e-6, r[0])\n } \n pulse(c, 4e-06, r[1]\n}",
            "r = ionreg(2) \n beam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n \
            serial {\n parallel {\n pulse(beam_mw, 5e-6, r[0])\n pulse(beam_mw, 5e-6, r[1])\n }\n}",
            "r = ionreg(3) \n beam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n \
            p0 = pulse(beam_mw, 5e-6, r[0]) \n p1 = pulse(beam_mw, 5e-6, r[1]) \n \
            p2 = pulse(beam_mw, 5e-6, r[2]) \n parallel {\n p0\n serial {\n p1\n p2 \n}\n}",
            "r = ionreg(2) \n b = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n \
            pulse(b, 1e-5, r[0], false)",
            "r = ionreg(2) \n beam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n \
            parallel {\n serial {\n x = pulse(beam_mw, 5e-6, r[0])\n x\n }\n}",
        ],
    )
    def test_atomic_bounds_checker(self, program):
        circuit = parse_atomic(program)
        cfg = AtomicCFGBuilder()(circuit)
        AtomicBoundsChecker().analyze(cfg)

    @pytest.mark.parametrize(
        "program",
        [
            "r = ionreg(1) \n r[1] \n r = ionreg(2) \n [r[0], r[2]]",
            "l = [5 * 4, 3 * 5] \n t = l[3]",
            "s = [5 + 2, 3] \n t = s[4]",
            "s = [5 - 2, 10] \n t = [1, 2, s] \n u = t[4]",
            "s = [6 / 2, 1] \n t = [s[3], 5]",
            "pi = 3.14159 \n s = sin(pi) \n c = cos(pi) \n a = [s, c] \n tan(a[2])",
            "a = [1, 0] \n atan2(a[0], a[2])",
            "cond = [true, false] \n if (cond[2]) {\n a = 0 \n }",
            "cond = [true, false] \n while (cond[2]) {t = 0.2}",
            "c = [1 < 2, 2 <= 4, 4 == 5] \n if (c[3]) { \n a = 2 \n }",
            "c = [3 >= 2, 1 > 0]  \n if (c[2]) { \n c = true \n }",
            "a = true \n c = [a, not a] \n if (c[2]) { \n c = a \n }",
            "a = [1 , 2 , 3] \n b = a[1] \n e = [a, b] \n e[3]",
            "r = ionreg(3) \n target = [r[0], r[1], r[3]]",
            "r = ionreg(3) \n b = [beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), \
            beam(1e6, 0.5, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])] \n \
            p0 = pulse(b[0], 5e-6, r[0]) \n p1 = pulse(b[3], 5e-6, r[1])",
            # "r = ionreg(2) \n b = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n \
            # c = beam(1e6, 0.5, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n parallel {\n \
            # serial {\n x = pulse(b, 5e-6, r[0])\n } \n pulse(c, 4e-06, r[3]\n}",
            "r = ionreg(2) \n beam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n \
            serial {\n parallel {\n pulse(beam_mw, 5e-6, r[0])\n pulse(beam_mw, 5e-6, r[2])\n }\n}",
            "r = ionreg(3) \n beam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n \
            p0 = pulse(beam_mw, 5e-6, r[0]) \n p1 = pulse(beam_mw, 5e-6, r[1]) \n \
            p2 = pulse(beam_mw, 5e-6, r[3]) \n parallel {\n p0\n serial {\n p1\n p2 \n}\n}",
            "r = ionreg(2) \n b = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n \
            pulse(b, 1e-5, r[4], false)",
            "r = ionreg(2) \n beam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \n \
            parallel {\n serial {\n x = pulse(beam_mw, 5e-6, r[2])\n x\n }\n}",
        ],
    )
    def test_atomic_bounds_checker_error(self, program):
        circuit = parse_atomic(program)
        with pytest.raises(OutOfBoundsError):
            cfg = AtomicCFGBuilder()(circuit)
            AtomicBoundsChecker().analyze(cfg)

