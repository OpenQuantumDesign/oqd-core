from midstack.interface.analog import *
from midstack.compiler.analog.verify.canonicalization import *
from midstack.compiler.analog.error import CanonicalFormError
from midstack.compiler.rule import RewriteRule
from midstack.compiler.walk import Walk, Post, Pre
import unittest
from midstack.interface.math import *
from unittest_prettify.colorize import (
    colorize,
    GREEN,
    BLUE,
    RED,
    MAGENTA,
)

X, Y, Z, I, A, C, LI = (
    PauliX(),
    PauliY(),
    PauliZ(),
    PauliI(),
    Annihilation(),
    Creation(),
    Identity(),
)


def test_function(
    operator: Operator, rule: RewriteRule, walk_method: Walk, reverse: bool
):
    walk_method(rule, reverse=reverse)(operator)


class CanonicalFormErrors(unittest.TestCase):

    def assertCanonicalFormErrorRaised(
        self,
        operator: Operator,
        rule: RewriteRule,
        walk_method: Walk = Post,
        reverse: bool = False,
    ):
        with self.assertRaises(CanonicalFormError) as context:
            test_function(
                operator=operator, rule=rule, walk_method=walk_method, reverse=reverse
            )
        print(context.exception)

    def assertCanonicalFormErrorNotRaised(
        self,
        operator,
        rule: RewriteRule,
        walk_method: Walk = Post,
        reverse: bool = False,
    ):
        with self.assertRaises(AssertionError) as context:
            self.assertCanonicalFormErrorRaised(
                operator=operator, rule=rule, walk_method=walk_method, reverse=reverse
            )
        print(context.exception)


@colorize(color=BLUE)
class TestCanonicalizationVerificationOperatorDistribute(
    CanonicalFormErrors, unittest.TestCase
):
    maxDiff = None

    def __init__(self, methodName: str = "runTest") -> None:
        self._rule = CanVerOperatorDistribute()
        super().__init__(methodName)

    def test_pauli_simple_fail(self):
        """Simple failure with pauli"""
        op = ((2 * X) + Y) @ Z
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_pauli_ladder_simple_fail(self):
        """Simple failure with pauli and ladder"""
        op = ((2 * X) + A) @ Z
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_pauli_simple_pass(self):
        """Simple pass with pauli"""
        op = (2 * X) @ Z + Y @ Z
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_pauli_ladder_simple_pass(self):
        """Simple pass with pauli"""
        op = (2 * X) @ Z + Y @ Z + 2 * (A * A * A * C) * (C * A * A)
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_pauli_nested_fail(self):
        """Simple failure with nested pauli"""
        op = Y * (X @ (X @ X * (X + X)))
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_pauli_nested_pass(self):
        """Simple pass with nested pauli"""
        op = (
            1 * (I @ (A * A))
            + 3 * (X @ (A * A))
            + 7 * (Y @ (A * A))
            + (Z @ (A * A))
            + 7 * (Z @ (A * C))
        )
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_nested_multiplication_pass(self):
        """Nested multiplication passing test"""
        op = 3 * (2 * (2 + 7) * 7 * (X @ Y @ Y @ Z @ (A * A * C * LI))) + (
            2j + 7
        ) * 7 * (X @ Z @ X @ Z @ (A * A * C * LI * A * A))
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_complex_nested_pass(self):
        """Complicated pass with nested pauli and ladder"""
        op = (
            1 * (I @ (A * A))
            + 3 * (X @ (A * A))
            + 7 * (Y @ (A * A))
            + (Z @ ((A * A * A * A * A * A * C * A * C * C * C)))
            + 7 * (Z @ (A * C))
        )
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_complex_nested_fail(self):
        """Complicated fail with nested pauli and ladder"""
        op = (
            1 * (I @ A * A)
            + 3 * (X @ A * A)
            + 7 * (Y @ A * A)
            + (Z @ (A * A * A * A * A * A * C + A * C * C * C))
            + 7 * (Z @ A * C)
        )
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_complex_pauli_fail(self):
        """Complicated test with complex numers"""
        op = 2 * (X @ (7 * Y * (1j + 2) * Y)) + 6 * (
            Z @ (-3j * Y)
        )  # 2*(X@(7*Y*(1j)*Y)) + 6*(Z@(-3j*Y))
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_addition_pauli_scalar_multiple_simple(self):
        """Pauli addition with scalar multiple passes"""
        op = 3 * (X + Y)
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_addition_pauli_scalar_multiple_nested(self):
        """Pauli addition with scalar multiple passes with nested operations"""
        op = 2j * (5 * (3 * (X + Y)) + 3 * (Z @ A * A * C * A))
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_subtraction_pauli(self):
        """Pauli Subtarction Fail"""
        op = X * (X - Y)
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_multiplication_OperatorKron_distribution(self):
        """Multiplication of OperatorKron should fail as op can be further simplified by distribution"""
        op = (X @ C) * (Y @ LI)
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_multiplication_OperatorScalarMul_distribution_v1(self):
        """Multiplication of OperatorScalarMul should just as op cannot be further simplified by distribution (needs MathExpr) v1"""
        op = (X) * (2 * Y)
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_multiplication_OperatorScalarMul_distribution_v2(self):
        """Multiplication of OperatorScalarMul should just as op cannot be further simplified by distribution (needs MathExpr) v2"""
        op = (X @ C) * (2 * (Y @ LI))
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_multiplication_OperatorScalarMul_distribution_v3(self):
        """Multiplication of OperatorScalarMul should just as op cannot be further simplified by distribution (needs MathExpr) v3"""
        op = (2 * (X @ C)) * (Y @ LI)
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_multiplication_OperatorScalarMul_distribution_v3(self):
        """Multiplication of OperatorScalarMul should just as op cannot be further simplified by distribution (needs MathExpr) v3"""
        op = (2 * (X @ C)) * (2 * (Y @ LI))
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_multiplication_OperatorScalarMul_distribution_v4(self):
        """Multiplication of OperatorScalarMul should just as op cannot be further simplified by distribution (needs MathExpr) v4"""
        op = (Y @ LI) * (2 * (X @ C)) * (Y @ LI)
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_multiplication_OperatorScalarMul_distribution_pass(self):
        """Error raised as op can be further simplified (i.e. specifically (X@C)*(Y@LI))"""
        op = (X @ C) * (Y @ LI) * (2 * (X @ C)) * (Y @ LI)
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)


@colorize(color=BLUE)
class TestCanonicalizationVerificationGatherMathExpr(
    CanonicalFormErrors, unittest.TestCase
):
    maxDiff = None

    def __init__(self, methodName: str = "runTest") -> None:
        self._rule = CanVerGatherMathExpr()
        super().__init__(methodName)

    def test_pauli_simple_pass(self):
        """Simple Pauli passing"""
        op = 2 * (X @ Z) + Y @ Z
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_pauli_simple_fail(self):
        """Simple Pauli failing"""
        op = (2 * X) @ Z + Y @ Z
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_pauli_simple_fail_one_term(self):
        """Simple Pauli failing with 1 term"""
        op = X * ((2 + 3) * X)
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_pauli_complicated_pass(self):
        """Complicated test passing with pauli and ladders"""
        op = 2j * (X @ (A * A * C) @ (Y @ (A * C * A * A * C * LI))) + (-1) * (
            A * C * A * A * C * LI
        )
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_pauli_complicated_fail(self):
        """Complicated test failing with pauli and ladders"""
        op = 2j * (X @ (A * A * C) @ (Y @ (A * C * A * A * C * LI))) + (-1) * (
            A * C * A * (1 * A) * C * LI
        )
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_addition_pauli_scalar_multiple_nested_fail(self):
        """Pauli addition fail with scalar multiple passes with nested operations"""
        op = 2j * (5 * (3 * (X + Y)) + 3 * (Z @ A * A * C * A))
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_pauli_nested_ops_fail(self):
        """Nested scalar multiplications of Paulis fail"""
        op = 3 * (3 * (3 * (X * Y)))
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_pauli_ladder_nested_ops_pass(self):
        """Nested scalar multiplications pass"""
        op = (3 * 3 * 3) * ((X * Y) @ (A * C))
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)


@colorize(color=BLUE)
class TestCanonicalizationVerificationProperOrder(
    CanonicalFormErrors, unittest.TestCase
):
    maxDiff = None

    def __init__(self, methodName: str = "runTest") -> None:
        self._rule = CanVerProperOrder()
        super().__init__(methodName)

    def test_simple_pauli_pass(self):
        """Simple tensor product in proper order"""
        op = (X @ Y) @ Z
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_simple_pauli_pass(self):
        """Simple tensor product not in proper order"""
        op = X @ (Y @ Z)
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_simple_pauli_with_other_ops_pass(self):
        """Simple tensor product with other operations in proper order"""
        op = (X @ (2 * Y + Z @ (A * C * A))) @ Z
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_simple_pauli_with_other_ops_pass(self):
        """Simple tensor product with other operations not in proper order"""
        op = X @ ((2 * Y + Z @ (A * C * A)) @ Z)
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_nested_paulis_v1(self):
        """Nested structure with nested addition not in proper order"""
        op = I + (X + (2 * Y) + Z)
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_nested_paulis_pass_v2(self):
        """Nested structure with nested addition in proper order"""
        op = (I + X) + ((2 * Y) + Z)
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_nested_paulis_pass_with_brackets(self):
        """Nested structure with nested addition in proper order with brackets"""
        op = ((I + X) + (2 * Y)) + Z
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_nested_paulis_pass_without_brackets(self):
        """Nested structure with nested addition in proper order with brackets (i.e. using Python default)"""
        op = I + X + (2 * Y) + Z
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_nested_tensor_prod_paulis_multiplication_fail(self):
        """Nested structure with nested multiplication (4j*(2...)) not in proper order"""
        op = (A @ C) @ (4j * (2 * (X + (2 * Y) + Z)))
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_nested_tensor_prod_paulis_multiplication_pass(self):
        """Nested structure with nested multiplication (4j*...) in proper order"""
        op = (A @ C) @ ((4j * 2) * (X + (2 * Y) + Z))
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)


@colorize(color=BLUE)
class TestCanonicalizationVerificationPauliAlgebra(
    CanonicalFormErrors, unittest.TestCase
):
    maxDiff = None

    def __init__(self, methodName: str = "runTest") -> None:
        self._rule = CanVerPauliAlgebra()
        super().__init__(methodName)

    def test_simple_pass(self):
        """Simple Pauli Pass"""
        op = X @ X + Z @ X
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_simple_fail(self):
        """Simple Pauli Fail"""
        op = X @ X + Z @ (X * X)
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_nested_pass(self):
        """Simple nested Pauli test pass"""
        op = I @ I @ X @ (X - Z) @ (A * A) + Z @ (X @ (X @ (X + Z))) @ (
            A * C * A * C * C
        )
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_nested_fail(self):
        """Simple nested Pauli test fail because of X * I"""
        op = I @ I @ X @ (X - Z) @ (A * A) + Z @ (X @ (X @ ((X * I) + Z))) @ (
            A * C * A * C * C
        )
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_assumption_pass(self):
        """Showing test passing due to assumption"""
        op = X * (2 * Y)
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_assumption_pass_v2(self):
        """Showing test passing due to assumption: GatherMathExpr needs to be done before further PauliAlgebra can be done"""
        op = X * ((1j) * Y)
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)


@colorize(color=BLUE)
class TestCanonicalizationVerificationGatherPauli(
    CanonicalFormErrors, unittest.TestCase
):
    maxDiff = None

    def __init__(self, methodName: str = "runTest") -> None:
        self._rule = CanVerGatherPauli()
        super().__init__(methodName)

    def test_simple_pass(self):
        """Simple pass"""
        op = X @ X @ Y @ (A * C * LI) @ A
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_simple_fail(self):
        """Simple fail"""
        op = X @ X @ Y @ (A * C * LI) @ A @ I
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_complicated_addition_pass(self):
        """Complicated Addition pass"""
        op = X @ X @ Y @ (A * C * LI) @ A + I @ I @ I @ I @ C @ C  ##  X@(A*A)@A+I@C@C
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_simple_adddition_fail(self):
        """Simple Addition fail"""
        op = X @ X @ Y @ (A * C * LI) @ A @ I + I @ I @ I @ I @ C @ C
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_simple_addition_pass(self):
        """Simple Addition pass"""
        op = X @ (A * A) @ A + I @ C @ C
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_assumption_addition_pass(self):
        """Assumption Single Nested Addition pass"""
        op = (Y + I) @ A @ X
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_assumtion_nested_addition_pass(self):
        """Assumption double Nested Addition pass"""
        op = (Y + I) @ A @ (Z + I)  # +(I+Z)@C@C@(X+(Z+I)) ##  X@(A*A)@A+I@C@C
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_assumption_simple_nested_addition_pass(self):
        """Error not found when Ladder @ PauliAddition"""
        op = A @ (Z + I)  # +(I+Z)@C@C@(X+(Z+I)) ##  X@(A*A)@A+I@C@C
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_assumption_simple_nested_subtraction_pass(self):
        """Error not found when Ladder @ PauliSubtraction"""
        op = A @ (Z + I)  # +(I+Z)@C@C@(X+(Z+I)) ##  X@(A*A)@A+I@C@C
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    # def test_assumption_simple_nested_product_fail(self):
    #     """Error found when Ladder @ PauliMultiplication"""
    #     op = A@(Z*I)
    #     self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_ladder_prod_fail(self):
        """ladder multiplication fail"""
        op = (A * C * A) @ X
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_ladder_prod_pass(self):
        """ladder multiplication pass"""
        op = X @ (A * C * A)
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)


@colorize(color=BLUE)
class TestCanonicalizationVerificationNormalOrder(
    CanonicalFormErrors, unittest.TestCase
):
    maxDiff = None

    def __init__(self, methodName: str = "runTest") -> None:
        self._rule = CanVerNormalOrder()
        super().__init__(methodName)

    def test_simple_pass(self):
        """Simple pass"""
        op = X @ X @ Y @ (C * A) @ A
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_simple_fail(self):
        """Simple fail"""
        op = X @ X @ Y @ (A * C) @ A
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_simple_pass_only_ladders(self):
        """Simple pass with only ladders"""
        op = (C * C * C * A) @ C @ LI
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_simple_fail_only_ladders(self):
        """Simple fail with only ladders"""
        op = (C * C * C * A) @ C @ LI
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_simple_addition_pass(self):
        """Simple pass with addition"""
        op = X @ X @ Y @ (C * A) @ A + I @ I @ I @ C @ C
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_simple_addition_fail(self):
        """Simple fail with addition"""
        op = X @ X @ Y @ (C * A * C) @ A + I @ I @ I @ C @ C
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_simple_starting_with_ladder_pass_v1(self):
        """Starting with ladder pass  v1"""
        op = (C * A) @ (Z + I)
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_simple_starting_with_ladder_fail_v1(self):
        """Starting with ladder fail  v1"""
        op = (C * A * C) @ (Z + I)
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_simple_starting_with_single_ladder_pass_v1(self):
        """Starting with single ladder pass  v1"""
        op = C @ (Z + I)
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_simple_starting_with_single_ladder_tensors(self):
        """Starting with ladder pass  v1"""
        op = C @ A @ C @ LI @ A @ C @ (Z + I) @ X @ Y
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_assumption_only_ladder_addition_without_distribution(self):
        """Error not detected when we do only ladder operations without distribution of ladders"""
        op = A * (A + C)
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_assumption_only_ladder_addition_with_distribution(self):
        """Error detected when we do only ladder operation with distribution of ladders"""
        op = (A * A) + (A * C)
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_terminal_ladder_normal_order(self):
        """Test if A*C (terminal gives error)"""
        op = (A * C) * LI
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_just_terminal_ladder_normal_order(self):
        """Test if just A*C (terminal gives error)"""
        op = A * C
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)


@colorize(color=BLUE)
class TestCanonicalizationVerificationPruneIdentity(
    CanonicalFormErrors, unittest.TestCase
):
    maxDiff = None

    def __init__(self, methodName: str = "runTest") -> None:
        self._rule = CanVerPruneIdentity()
        super().__init__(methodName)

    def test_simple_pass(self):
        """Simple pass"""
        op = X @ X @ Y @ (C * A) @ A
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_simple_addition_pass(self):
        """Simple pass"""
        op = A @ LI @ LI + C @ C @ LI
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_simple_fail(self):
        """Simple pass"""
        op = X @ X @ Y @ (C * A * LI) @ A
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_show_proper_order_not_needed(self):
        """Showing that expression does not need to be in proper order"""
        op = A * (C * A * (A * C * A * (C * C) * LI))
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)


@colorize(color=BLUE)
class TestCanonicalizationVerificationSortedOrder(
    CanonicalFormErrors, unittest.TestCase
):
    maxDiff = None

    def __init__(self, methodName: str = "runTest") -> None:
        self._rule = CanVerSortedOrder()
        super().__init__(methodName)

    def test_simple_pass(self):
        """Simple Pass"""
        op = X + (4 * Y) + Z
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_simple_fail_2_terms(self):
        """Simple fail with 2 terms"""
        op = X + I
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_simple_fail(self):
        """Simple Fail"""
        op = X + (4 * Y) + I
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_nested_pass(self):
        """Nested Pass"""
        op = (X @ Y + (2 * (3j) * (X @ Z))) + (Y @ I) + (Z @ I)
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_nested_fail(self):
        """Nested fail because of I@Y. This tests if it works with OperatorScalarMul"""
        op = (X @ Y + (2 * (3j) * (I @ Y))) + (Y @ I) + (Z @ I)
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_simple_fail_identical_operators(self):
        """Fail with identical operators"""
        op = X @ I @ Z + 2 * X @ I @ Z
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_assumtion_pass(self):
        """Hamiltonian needs to be distributed"""
        op = (I + X) @ Z
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_simple_pass_ladder(self):
        """Simple Pass ladder"""
        op = C * A + (C * A * A * A)
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_simple_fail_ladder(self):
        """Simple fail ladder"""
        op = (C * A * A * A) + (C * A)
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_simple_pass_ladder_3_terms(self):
        """Simple Pass ladder 3 terms"""
        op = C * A + (C * A * A * A) + (C * C * A * A)
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_simple_fail_ladder_3_terms(self):
        """Simple fail ladder 3 terms"""
        op = C * A + (C * C * A * A) + (C * A * A * A)
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_simple_pass_ladder_pauli(self):
        """Simple pass ladder-pauli 3 terms: Pauli's are given precedence in order of importance"""
        op = X @ (C * A) + Y @ (C * C * A * A) + Z @ (C * A * A * A)
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_simple_fail_ladder_pauli(self):
        """Simple fail ladder-pauli 3 terms: Pauli's are given precedence in order of importance"""
        op = X @ (C * A) + I @ (C * A * A * A) + Z @ (C * C * A * A)
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_scalar_mul_pass(self):
        """Simple pass with scalar mul"""
        op = X + Y + 2 * 2 * Z
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_scalar_mul_fail(self):
        """Simple fail with scalar mul because of repeated operator"""
        op = X + Y + 2 * 2 * Z + Z
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_scalar_mul_pass_with_ladder(self):
        """Simple pass with scalar mul with ladders"""
        op = 2 * (X @ X @ (C * A * A)) + 2 * (Y @ Y @ (C * A)) + 3 * (Z @ Z @ (C * A))
        self.assertCanonicalFormErrorNotRaised(operator=op, rule=self._rule)

    def test_scalar_mul_fail_with_ladder(self):
        """Simple fail with scalar mul with ladders because ladders are not sorted"""
        op = 2 * (X @ X @ (C * A * A)) + 2 * (X @ X @ (C * A)) + 3 * (Z @ Z @ (C * A))
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_scalar_mul_fail_duplicate(self):
        """Simple fail with duplicates"""
        op = X + X
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_scalar_mul_fail_duplicate_more_terms(self):
        """Error Not raised with duplicates with more terms"""
        op = I @ X + X @ X + X @ Y + X @ Z + Y @ X + Y @ Y + Y @ Z + I @ X
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_scalar_mul_fail_duplicate_complex(self):
        """Simple fail with duplicates with ladder"""
        op = (
            2 * (X @ X @ (C * A * A))
            + 9 * (X @ X @ (C * A * A))
            + 3 * (Z @ Z @ (C * A))
        )
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)

    def test_nested_fail_duplicate(self):
        """Nested fail due to duplicate operators"""
        op = (X @ Y + (2 * (3j) * (X @ Y))) + (Y @ I) + (Z @ I)
        self.assertCanonicalFormErrorRaised(operator=op, rule=self._rule)


@colorize(color=BLUE)
class TestCanonicalizationVerificationScaleTerms(
    CanonicalFormErrors, unittest.TestCase
):
    maxDiff = None

    def __init__(self, methodName: str = "runTest") -> None:
        self._rule = CanVerScaleTerm()
        self._walk_method = Pre
        super().__init__(methodName)

    def test_simple_addition_pass(self):
        """Addition pass"""
        op = 2 * (X @ Z) + 3 * (Y @ I) + 2 * (Z @ Z)
        self.assertCanonicalFormErrorNotRaised(
            operator=op, rule=self._rule, walk_method=self._walk_method
        )

    def test_simple_addition_fail(self):
        """Addition fail"""
        op = 2 * (X @ Z) + (Y @ I) + 2 * (Z @ Z)
        self.assertCanonicalFormErrorRaised(
            operator=op, rule=self._rule, walk_method=self._walk_method
        )

    def test_single_pass(self):
        """Single term pass"""
        op = 2 * (X @ Z)
        self.assertCanonicalFormErrorNotRaised(
            operator=op, rule=self._rule, walk_method=self._walk_method
        )

    def test_single_fail(self):
        """Single Term fail"""
        op = X @ Z
        self.assertCanonicalFormErrorRaised(
            operator=op, rule=self._rule, walk_method=self._walk_method
        )

    def test_complicated_addition_pass(self):
        """Complicated addition pass"""
        op = (
            2 * (X @ Z @ Z @ (C * A))
            + 3 * Z
            + 2 * (Y @ Z)
            + 1 * A
            + 1 * (C * C)
            + 1 * I
            + 1 * Z
        )
        self.assertCanonicalFormErrorNotRaised(
            operator=op, rule=self._rule, walk_method=self._walk_method
        )

    def test_simple_addition_fail(self):
        """Complicated addition fail"""
        op = (
            2 * (X @ Z @ Z @ (C * A))
            + Z
            + 2 * (Y @ Z)
            + 1 * A
            + 1 * (C * C)
            + 1 * I
            + 1 * Z
        )
        self.assertCanonicalFormErrorRaised(
            operator=op, rule=self._rule, walk_method=self._walk_method
        )


if __name__ == "__main__":
    unittest.main()
