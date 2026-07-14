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
from oqd_core.compiler.atomic.cfg_passes.protocol import canonicalize_protocol_circuit
from oqd_core.compiler.atomic.cfg_passes.walk import (
    canonicalize_declarations_cfg,
    iter_stmt_blocks,
)
from oqd_core.compiler.atomic.passes.compile import compile_atomic_circuit
from oqd_core.frontend.atomic.AtomicCircuitAST import parse_atomic
from oqd_core.interface.atomic import (
    Declaration,
    Pulse,
    MathNum,
    ParallelProtocol,
    SerialProtocol,
)

REGISTER = "r = ionreg(2)\n"
BEAM = "b = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])\n"

def build_inputs(program: str):
    circuit = parse_atomic(program)
    cfg = AtomicCFGBuilder().run(circuit)
    type_checker = AtomicTypeChecker(cfg)
    symbol_table = AtomicSymbolTableBuilder(
        cfg, type_checker.dataflow_result
    ).symbol_table
    return circuit, cfg, symbol_table, type_checker.dataflow_result


def declaration_value(cfg, name):
    for _, block in iter_stmt_blocks(cfg):
        stmt = block.stmt
        if isinstance(stmt, Declaration) and stmt.name == name:
            return stmt.value
    raise KeyError(name)


## Compile ##

class TestAtomicCompile:
    @pytest.mark.parametrize(
        "program",
        [
            REGISTER + BEAM + "pulse(b, 1e-5, r, true)",
            REGISTER + BEAM + "parallel {\n pulse(b, 5e-6, r[0])\n pulse(b, 5e-6, r[1])\n}",
            REGISTER + BEAM + "serial {\n pulse(b, 5e-6, r[0])\n pulse(b, 5e-6, r[1])\n}",
            REGISTER + BEAM + "parallel {\n serial {\n pulse(b, 5e-6, r[0])\n pulse(b, 5e-6, r[1])\n}\n}",
        ],
    )
    def test_compile(self, program):
        circuit, cfg, symbol_table, type_result = build_inputs(program)
        compile_atomic_circuit(circuit, cfg, symbol_table, type_result)
    
    def test_rebuilds_cfg_after_protocol(self):
        program = (
            REGISTER + BEAM
            + "parallel {\n serial {\n pulse(b, 5e-6, r[0])\n pulse(b, 5e-6, r[1])\n}\n}"
        )
        circuit, cfg, symbol_table, type_result = build_inputs(program)
        _, new_cfg = compile_atomic_circuit(circuit, cfg, symbol_table, type_result)
        assert new_cfg is not cfg


## CFG Passes ##

class TestAtomicCanonicalizeDeclarationsCfg:
    def test_beam_fields_canonicalized(self):
        _, cfg, _, type_result = build_inputs(
            "b = beam(1e6 + 1e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])"
        )
        canonicalize_declarations_cfg(cfg, type_result)
        assert declaration_value(cfg, "b").frequency == MathNum(value=2e6)

    def test_pulse_duration_canonicalized(self):
        _, cfg, _, type_result = build_inputs(
            REGISTER + BEAM + "p = pulse(b, 1e-5 + 1e-5, r[0])"
        )
        canonicalize_declarations_cfg(cfg, type_result)
        assert declaration_value(cfg, "p").duration == MathNum(value=2e-5)
    

class TestAtomicCanonicalizeProtocol:
    def test_nested_serial_in_parallel(self):
        program = (
            REGISTER + BEAM
            + "parallel {\n serial {\n pulse(b, 5e-6, r[0])\n pulse(b, 5e-6, r[1])\n}\n}"
        )
        circuit, _, _, _ = build_inputs(program)
        canonicalize_protocol_circuit(circuit)
        stmt = circuit.statements[-1]
        assert isinstance(stmt, SerialProtocol)
        assert all(isinstance(p, ParallelProtocol) for p in stmt.pulses)

    def test_serial_protocol_pulses_remain_pulses(self):
        program = REGISTER + BEAM + "serial {\n pulse(b, 5e-6, r[0])\n pulse(b, 5e-6, r[1])\n}"
        circuit, _, _, _ = build_inputs(program)
        canonicalize_protocol_circuit(circuit)
        stmt = circuit.statements[-1]
        assert all(isinstance(p, ParallelProtocol) for p in stmt.pulses)
        assert all(isinstance(p.pulses[0], Pulse) for p in stmt.pulses)
