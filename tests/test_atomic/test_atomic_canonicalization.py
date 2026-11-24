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
    CFGBlockAccumulator,
    Chain,
    FixedPoint,
    Post,
    RelabelCFGBlocks,
    InplacePost,
    InplacePre,
)

from oqd_core.atomic import (
    parse_atomic,
    AtomicCFGBuilder,
    Constant,
    Add,
    Extract,
    Sub,
    Mul,
    Pow,
    Pos,
    Pulse,
    Neg,
    Access,
    Declaration,
    SerialProtocol,
    ParallelProtocol,
)

from oqd_core.analysis.dominator import PostDominatorAnalysis
from oqd_core.analysis.atomic.fold import ConstantFolding, ConstantFoldingRule, ConstantDurationRule

from oqd_core.compiler.atomic.canonicalization import (
    ReplaceSubtraction,
    UnaryCombination,
    CollectAssociativeOperands,
    SortOperands,
    ResolveNestedProtocol,
)
from oqd_core.compiler.atomic import AtomicCFGtoAST


class TestAtomicCanonicalization:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("a = 1 \n b = a + 2 + 3 + a + 5", Declaration(name="b", value=Add(exprs=[Constant(value=2), Constant(value=3), Constant(value=5), Access(name="a"), Access(name="a")]))),
            ("a = 2 \n b = 5 \n c = b + a + 5", Declaration(name="c", value=Add(exprs=[Constant(value=5), Access(name="b"), Access(name="a")]))),
            ("a = 1 \n b = 5 \n c = 4 \n d = b + 3 + c + 2 + 5 + a", Declaration(name="d", value=Add(exprs=[Constant(value=3), Constant(value=2), Constant(value=5), Access(name="b"), Access(name="c"), Access(name="a")]))),
        ],
    )
    def test_atomic_sort_operands(
        self, 
        source,
        expected,
    ):
        circuit = parse_atomic(source)
        cfg = AtomicCFGBuilder()(circuit)
        cfg = Post(SortOperands())(cfg)
        assert cfg[len(cfg)-2].stmts[0] == expected
    
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("b = 2 + (3 + 5)", Declaration(name="b", value=Add(exprs=[Constant(value=2), Constant(value=3), Constant(value=5)]))),
            ("b = 2 * (3 * 5)", Declaration(name="b", value=Mul(exprs=[Constant(value=2), Constant(value=3), Constant(value=5)]))),
            ("b = 2 * (3 * -5)", Declaration(name="b", value=Neg(expr=Mul(exprs=[Constant(value=2), Constant(value=3), Constant(value=5)])))),
            ("b = 2 + +(3 + 5)", Declaration(name="b", value=Add(exprs=[Constant(value=2), Constant(value=3), Constant(value=5)]))),
            ("b = 2 + -(3 + 5)", Declaration(name="b", value=Add(exprs=[Constant(value=2), Neg(expr=Constant(value=3)), Neg(expr=Constant(value=5))]))),
        ],
    )
    def test_atomic_collect_assoc_operands(
        self, 
        source,
        expected,
    ):
        circuit = parse_atomic(source)
        cfg = AtomicCFGBuilder()(circuit)
        cfg = FixedPoint(Post(CollectAssociativeOperands()))(cfg)
        assert cfg[len(cfg)-2].stmts[0] == expected
        
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("b = 2 + +3", Declaration(name="b", value=Add(exprs=[Constant(value=2), Constant(value=3)]))),
            ("b = 2 + -3", Declaration(name="b", value=Add(exprs=[Constant(value=2), Neg(expr=Constant(value=3))]))),
            ("b = +2 + +3", Declaration(name="b", value=Add(exprs=[Constant(value=2), Constant(value=3)]))),
            ("b = +2 + -3", Declaration(name="b", value=Add(exprs=[Constant(value=2), Neg(expr=Constant(value=3))]))),
        ],
    )
    def test_atomic_unary_combination(
        self, 
        source,
        expected,
    ):
        circuit = parse_atomic(source)
        cfg = AtomicCFGBuilder()(circuit)
        cfg = FixedPoint(Post(UnaryCombination()))(cfg)
        assert cfg[len(cfg)-2].stmts[0] == expected
    
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("b = 2 - 3", Declaration(name="b", value=Add(exprs=[Constant(value=2), Neg(expr=Constant(value=3))]))),
            ("b = 2 - 3 - 4", Declaration(name="b", value=Add(exprs=[Constant(value=2), Neg(expr=Constant(value=3)), Neg(expr=Constant(value=4))]))),
            ("b = 2 / 3 / 2", Declaration(name="b", value=Mul(exprs=[Constant(value=2), Pow(exprs=[Constant(value=3), Constant(value=-1)]), Pow(exprs=[Constant(value=2), Constant(value=-1)])]))),
            ("b = 2 / (3 - 4) / 2", Declaration(name="b", value=Mul(exprs=[Constant(value=2), Pow(exprs=[Add(exprs=[Constant(value=3), Neg(expr=Constant(value=4))]), Constant(value=-1)]), Pow(exprs=[Constant(value=2), Constant(value=-1)])]))),
        ],
    )
    def test_atomic_replace_sub(
        self, 
        source,
        expected,
    ):
        circuit = parse_atomic(source)
        cfg = AtomicCFGBuilder()(circuit)
        cfg = FixedPoint(Post(ReplaceSubtraction()))(cfg)
        assert cfg[len(cfg)-2].stmts[0] == expected
    
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("b = 1 \n c = b - 2 + (3 + +5)", Declaration(name="c", value=Add(exprs=[Constant(value=3), Constant(value=5), Access(name="b"), Neg(expr=Constant(value=2))]))),
            ("b = 1 \n c = +b * (3 * -5) / 2 / 3", Declaration(name="c", value=Neg(expr=Mul(exprs=[Constant(value=3), Constant(value=5), Access(name="b"), Pow(exprs=[Constant(value=2), Constant(value=-1)]), Pow(exprs=[Constant(value=3), Constant(value=-1)])])))),
            ("b = 1 \n c = -b * (3 * -5) / (2 - b - 4) / 3", Declaration(name='c', value=Mul(exprs=[Constant(value=3), Constant(value=5), Access(name='b'), Pow(exprs=[Add(exprs=[Constant(value=2), Neg(expr=Access(name='b')), Neg(expr=Constant(value=4))]), Constant(value=-1)]), Pow(exprs=[Constant(value=3), Constant(value=-1)])]))),
        ],
    )
    def test_atomic_canonicalization_all(
        self, 
        source,
        expected,
    ):
        circuit = parse_atomic(source)
        cfg = AtomicCFGBuilder()(circuit)
        passes = Chain(
            SortOperands(),
            CollectAssociativeOperands(),
            UnaryCombination(),
            ReplaceSubtraction(),
        )
        cfg = FixedPoint(Post(passes))(cfg)
        assert cfg[len(cfg)-2].stmts[0] == expected
    
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("r = ionreg(2) \nbeam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \npulse(beam_mw, 1.0, r[0])", \
                Pulse(beam=Access(name="beam_mw"), duration=Constant(value=1.0), target=Extract(value=Access(name="r"), index=0), measured=Constant(value=False))),
            ("r = ionreg(3) \nbeam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \nb = beam(1e6, 0.5, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \
                \nparallel { \npulse(beam_mw, 0.0001, r[0]) \npulse(b, 0.0002, r[0]) \n }", \
                SerialProtocol(pulses=[ParallelProtocol(pulses=[Pulse(beam=Access(name='beam_mw'), duration=Constant(value=0.0001), \
                target=Extract(value=Access(name='r'), index=Constant(value=0)), measured=Constant(value=False)), Pulse(beam=Access(name='b'), \
                duration=Constant(value=0.0001), target=Extract(value=Access(name='r'), index=Constant(value=0)), measured=Constant(value=False))]), \
                Pulse(beam=Access(name='b'), duration=Constant(value=0.0001), target=Extract(value=Access(name='r'), index=Constant(value=0)), measured=Constant(value=False))])),
            ("r = ionreg(3) \nbeam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \nb = beam(1e6, 0.5, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \
                \nparallel { \npulse(beam_mw, 0.0001, r[0]) \nserial { \npulse(b, 0.0002, r[0]) \n} \n }", \
                SerialProtocol(pulses=[ParallelProtocol(pulses=[Pulse(beam=Access(name='beam_mw'), duration=Constant(value=0.0001), \
                target=Extract(value=Access(name='r'), index=Constant(value=0)), measured=Constant(value=False)), Pulse(beam=Access(name='b'), \
                duration=Constant(value=0.0001), target=Extract(value=Access(name='r'), index=Constant(value=0)), measured=Constant(value=False))]), \
                Pulse(beam=Access(name='b'), duration=Constant(value=0.0001), target=Extract(value=Access(name='r'), index=Constant(value=0)), measured=Constant(value=False))])),
            ("r = ionreg(3) \nbeam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \nb = beam(1e6, 0.5, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \
                \nparallel { \nserial { \npulse(beam_mw, 0.0001, r[0]) \npulse(b, 0.0002, r[0]) \n} \npulse(b, 0.0001, r[0]) \n}", \
                SerialProtocol(pulses=[ParallelProtocol(pulses=[Pulse(beam=Access(name='beam_mw'), duration=Constant(value=0.0001), \
                target=Extract(value=Access(name='r'), index=Constant(value=0)), measured=Constant(value=False)), Pulse(beam=Access(name='b'), \
                duration=Constant(value=0.0001), target=Extract(value=Access(name='r'), index=Constant(value=0)), measured=Constant(value=False))]), \
                Pulse(beam=Access(name='b'), duration=Constant(value=0.0002), target=Extract(value=Access(name='r'), index=Constant(value=0)), measured=Constant(value=False))])),
        ],
    )
    def test_atomic_canonicalization_nested_protocol(
        self,
        source,
        expected,
    ):
        circuit = parse_atomic(source)
        cfg = AtomicCFGBuilder()(circuit)
        cfg = FixedPoint(InplacePost(ResolveNestedProtocol()))(cfg)
        assert cfg[len(cfg)-2].stmts[0] == expected
    
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("r = ionreg(2) \nbeam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \npulse(beam_mw, 1.0, r[0])", \
                Pulse(beam=Access(name="beam_mw"), duration=Constant(value=1.0), target=Extract(value=Access(name="r"), index=0), measured=Constant(value=False))),
            ("r = ionreg(3) \nbeam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \nb = beam(1e6, 0.5, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) \
                \ndur = 0.0001 \nparallel { \npulse(beam_mw, dur, r[0]) \npulse(b, dur * 2, r[0]) \n }", \
                SerialProtocol(pulses=[ParallelProtocol(pulses=[Pulse(beam=Access(name='beam_mw'), duration=Constant(value=0.0001), \
                target=Extract(value=Access(name='r'), index=Constant(value=0)), measured=Constant(value=False)), Pulse(beam=Access(name='b'), \
                duration=Constant(value=0.0001), target=Extract(value=Access(name='r'), index=Constant(value=0)), measured=Constant(value=False))]), \
                Pulse(beam=Access(name='b'), duration=Constant(value=0.0001), target=Extract(value=Access(name='r'), index=Constant(value=0)), measured=Constant(value=False))])),
        ],
    )
    def test_atomic_canonicalization_nested_protocol_with_fold(
        self,
        source,
        expected,
    ):
        circuit = parse_atomic(source)
        cfg = AtomicCFGBuilder()(circuit)

        fold_checker = ConstantFolding()
        fold_res = fold_checker.analyze(cfg)
        fold_pass = FixedPoint(InplacePre(ConstantFoldingRule(fold_res=fold_res)))
        cfg = fold_pass(cfg)

        cfg = FixedPoint(InplacePost(ResolveNestedProtocol()))(cfg)
        assert cfg[len(cfg)-2].stmts[0] == expected


    