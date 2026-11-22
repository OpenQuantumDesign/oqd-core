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
)

from oqd_core.analog import (
    parse_analog,
    AnalogCFGBuilder,
    Constant,
    Add,
    Sub,
    Mul,
    Pow,
    Pos,
    Neg,
    Access,
    Declaration
)

from oqd_core.compiler.analog.canonicalization import (
    ReplaceSubtraction,
    UnaryCombination,
    CollectAssociativeOperands,
    SortOperands,
)


class TestAnalogCanonicalization:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("a = 1 \n b = a + 2 + 3 + a + 5", Declaration(name="b", value=Add(exprs=[Constant(value=2), Constant(value=3), Constant(value=5), Access(name="a"), Access(name="a")]))),
            ("a = 2 \n b = 5 \n c = b + a + 5", Declaration(name="c", value=Add(exprs=[Constant(value=5), Access(name="b"), Access(name="a")]))),
            ("a = 1 \n b = 5 \n c = 4 \n d = b + 3 + c + 2 + 5 + a", Declaration(name="d", value=Add(exprs=[Constant(value=3), Constant(value=2), Constant(value=5), Access(name="b"), Access(name="c"), Access(name="a")]))),
        ],
    )
    def test_analog_sort_operands(
        self, 
        source,
        expected,
    ):
        circuit = parse_analog(source)
        cfg = AnalogCFGBuilder()(circuit)
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
    def test_analog_collect_assoc_operands(
        self, 
        source,
        expected,
    ):
        circuit = parse_analog(source)
        cfg = AnalogCFGBuilder()(circuit)
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
    def test_analog_unary_combination(
        self, 
        source,
        expected,
    ):
        circuit = parse_analog(source)
        cfg = AnalogCFGBuilder()(circuit)
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
    def test_analog_replace_sub(
        self, 
        source,
        expected,
    ):
        circuit = parse_analog(source)
        cfg = AnalogCFGBuilder()(circuit)
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
    def test_analog_canonicalization_all(
        self, 
        source,
        expected,
    ):
        circuit = parse_analog(source)
        cfg = AnalogCFGBuilder()(circuit)
        passes = Chain(
            SortOperands(),
            CollectAssociativeOperands(),
            UnaryCombination(),
            ReplaceSubtraction(),
        )
        cfg = FixedPoint(Post(passes))(cfg)
        assert cfg[len(cfg)-2].stmts[0] == expected
    


