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

from typing import Union

import pytest
from oqd_compiler_infrastructure import ConversionRule, Post, RewriteRule, WalkBase

from oqd_core.compiler.analog.utils import PrintOperator
from oqd_core.compiler.analog.operator.dim import operator_dim
from oqd_core.compiler.analog.error import AnalogCompilerError
from oqd_core.compiler.analog.operator.dim import scalar_mul

########################################################################################
from oqd_core.interface.analog.expr import (
    Annihilation,
    Creation,
    Identity,
    OperatorExpr,
    PauliI,
    PauliX,
    PauliY,
    PauliZ,
    MathNum,
    OperatorMul,
)
from helpers import parse_math

########################################################################################


X, Y, Z, PI, A, C, LI = (
    PauliX(),
    PauliY(),
    PauliZ(),
    PauliI(),
    Annihilation(),
    Creation(),
    Identity(),
)


def apply_pass(
    operator: OperatorExpr,
    rule: Union[ConversionRule, RewriteRule] = PrintOperator(verbose=True),
    walk_method: WalkBase = Post,
    reverse: bool = False,
):
    return walk_method(rule, reverse=reverse)(operator)


#######################################################


class TestRealFinalStringVerbosePrintOp:
    """Testing the final string produced with the actual for real number expressions"""

    def setup_method(self):
        self._rule = PrintOperator(verbose=True)

    def test_pauli_add_V(self):
        """Testing Vanilla Pauli Additions"""
        operator = PauliX() + PauliY() + PauliZ()
        actual = "(PauliX() + PauliY()) + PauliZ()"
        assert apply_pass(operator=operator, rule=self._rule) == actual

    def test_pauli_operator_multiply_V(self):
        """Testing Vanilla Pauli Multiplication  with verbose print"""
        operator = PauliX() * PauliY() * PauliZ()
        actual = "(PauliX() * PauliY()) * PauliZ()"
        assert apply_pass(operator=operator, rule=self._rule) == actual

    def test_pauli_scalar_left_multiplication_V(self):
        """Testing scalar Pauli left Multiplication"""
        operator = 3 * PauliX()
        actual = "(3) * (PauliX())"
        assert apply_pass(operator=operator, rule=self._rule) == actual

    def test_pauli_scalar_left_multiple_multiplication_V(self):
        """Testing scalar Pauli left Multiplication with several terms"""
        operator = 3 * 5 * PauliX()
        actual = "(15) * (PauliX())"
        assert apply_pass(operator=operator, rule=self._rule) == actual

    def test_pauli_scalar_right_multiplication_V(self):
        """Testing scalar Pauli right Multiplication"""
        operator = PauliX() * 3
        actual = "(3) * (PauliX())"
        assert apply_pass(operator=operator, rule=self._rule) == actual

    def test_pauli_scalar_right_multiple_multiplication_V(self):
        """Testing scalar Pauli right Multiplication"""
        operator = PauliX() * 3 * 5
        actual = "(5) * ((3) * (PauliX()))"
        assert apply_pass(operator=operator, rule=self._rule) == actual

    def test_pauli_scalar_multiplication_with_addition_V(self):
        """Testing scalar Pauli Multiplication with addition"""
        operator = 3 * PauliX() + 5 * PauliI()
        actual = "((3) * (PauliX())) + ((5) * (PauliI()))"
        assert apply_pass(operator=operator, rule=self._rule) == actual

    def test_pauli_scalar_multiplication_nested_l1_V(self):
        """Testing scalar Pauli Multiplication with nested operations"""
        operator = (PauliX() * 3) @ PauliY() + (5 * PauliZ()) @ (2 * PauliI())
        actual = "(((3) * (PauliX())) @ PauliY()) + (((5) * (PauliZ())) @ ((2) * (PauliI())))"
        assert apply_pass(operator=operator, rule=self._rule) == actual

    def test_pauli_scalar_multiplication_nested_l2_V(self):
        """Testing scalar Pauli Multiplication with more nested operations with verbose print"""
        operator = (PauliX() * 3) @ ((3 * PauliY() + 7 * PauliY()) - (3 * PauliZ())) + (
            5 * 5 * PauliZ()
        ) @ (2 * PauliI())
        actual = "(((3) * (PauliX())) @ ((((3) * (PauliY())) + ((7) * (PauliY()))) - ((3) * (PauliZ())))) + (((25) * (PauliZ())) @ ((2) * (PauliI())))"
        assert apply_pass(operator=operator, rule=self._rule) == actual

    def test_pauli_scalar_string_nested_float_combination_V(self):
        """Testing scalar Pauli multiplication with combination of string and (int, float) with verbose print"""
        operator = scalar_mul(parse_math("3 * 4 * #t"), PauliX())
        actual = "((3 * 4) * #t) * (PauliX())"
        assert apply_pass(operator=operator, rule=self._rule) == actual

    def test_pauli_kron_with_addition_V(self):
        """Testing tensor product with addition for Pauli with verbose print"""
        operator = PauliX() @ (PauliI() + PauliZ()) @ PauliY()
        actual = "(PauliX() @ (PauliI() + PauliZ())) @ PauliY()"
        assert apply_pass(operator=operator, rule=self._rule) == actual

    def test_pauli_string_left_multiplication_V(self):
        """Testing string Pauli left Multiplication"""
        operator = scalar_mul(parse_math("sin(#t)"), PauliX())
        actual = "(sin(#t)) * (PauliX())"
        assert apply_pass(operator=operator, rule=self._rule) == actual

    def test_pauli_string_right_multiplication_V(self):
        """Testing string Pauli right Multiplication"""
        operator = scalar_mul(parse_math("cos(#t)"), PauliX())
        actual = "(cos(#t)) * (PauliX())"
        assert apply_pass(operator=operator, rule=self._rule) == actual

    def test_pauli_string_left_right_multiplication_V(self):
        """Testing string Pauli left and right Multiplication with verbose print"""
        operator = scalar_mul(
            parse_math("cos(#t)"),
            scalar_mul(parse_math("sin(#t)"), PauliX()),
        )
        actual = "(cos(#t)) * ((sin(#t)) * (PauliX()))"
        assert apply_pass(operator=operator, rule=self._rule) == actual

    def test_pauli_string_left_right_multiplication_nested_V(self):
        """Testing string Pauli left and right nested Multiplication  with verbose print"""
        operator = scalar_mul(
            parse_math("cos(#t)"),
            scalar_mul(parse_math("sin(#t + 3 * tan(#t))"), PauliX()),
        )
        actual = "(cos(#t)) * ((sin(#t + (3 * tan(#t)))) * (PauliX()))"
        assert apply_pass(operator=operator, rule=self._rule) == actual

    def test_pauli_minus_string_V(self):
        """Pauli scalar multiplication with string and with negative sign"""
        operator = scalar_mul(parse_math("2 * 4 * #t"), -PauliX())
        actual = "((2 * 4) * #t) * ((-1) * (PauliX()))"
        assert apply_pass(operator=operator, rule=self._rule) == actual

    def test_pauli_multiple_minus_V(self):
        """Pauli scalar multiplication with string and with double negative  with verbose print"""
        operator = 2 * -(-PauliX())
        actual = "(2) * ((-1) * ((-1) * (PauliX())))"
        assert apply_pass(operator=operator, rule=self._rule) == actual

    @pytest.mark.skip(reason="Not Implemented")
    def test_pauli_multiple_minus_nested_V(self):
        raise NotImplementedError


class TestComplexFinalStringVerbosePrintOp:
    """Testing the final string produced with the actual for complex number expressions"""

    def setup_method(self):
        self._rule = PrintOperator(verbose=True)

    def test_pauli_left_img_V(self):
        """Testing Pauli with left img with verbose print"""
        operator = 2j * PauliX()
        actual = "(0.0 + (1j * 2.0)) * (PauliX())"
        assert apply_pass(operator=operator, rule=self._rule) == actual

    def test_pauli_right_img_V(self):
        """Testing Pauli with right img  with verbose print"""
        operator = PauliX() * 2j
        actual = "(0.0 + (1j * 2.0)) * (PauliX())"
        assert apply_pass(operator=operator, rule=self._rule) == actual

    def test_pauli_nested_img_V(self):
        """Testing Pauli with nested img with verbose print"""
        operator = PauliX() * 2j * (PauliI() + 8j * PauliY())
        actual = "((0.0 + (1j * 2.0)) * (PauliX())) * (PauliI() + ((0.0 + (1j * 8.0)) * (PauliY())))"
        assert apply_pass(operator=operator, rule=self._rule) == actual

