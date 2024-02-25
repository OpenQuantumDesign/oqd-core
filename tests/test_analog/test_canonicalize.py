from quantumion.interface.analog import *
from quantumion.compiler.analog.canonicalize import *
from quantumion.compiler.analog.base import *
from rich import print as pprint
import unittest
from quantumion.interface.math import *
from unittest_prettify.colorize import (
    colorize,
    GREEN,
    BLUE,
    RED,
    MAGENTA,
)
X, Y, Z, I, A, C, LI = PauliX(), PauliY(), PauliZ(), PauliI(), Annihilation(), Creation(), Identity()

def test_function(operator: Operator, visitor = PrintOperator()):
    return operator.accept(visitor=visitor)

class CanonicalFormErrors(unittest.TestCase):

    def assertCanonicalFormErrorRaised(self, operator, visitor = CanonicalizationVerificationOperator()):
        with self.assertRaises(CanonicalFormError) as context:
            test_function(operator = operator, visitor=visitor)
        print(context.exception)
    
    def assertCanonicalFormErrorNotRaised(self, operator, visitor = CanonicalizationVerificationOperator()):
        with self.assertRaises(AssertionError) as context:
            self.assertCanonicalFormErrorRaised(operator=operator, visitor=visitor)
        print(context.exception)

@colorize(color=MAGENTA)
class TestCanonicalizationVerification(CanonicalFormErrors, unittest.TestCase):
    maxDiff = None

    def test_simple_addition_true(self):
        """Simple addition test of canonical operator"""
        op =  1*I + 2*X
        #self.assertTrue(test_function(operator = op, visitor=CanonicalizationVerificationAddition()))
        self.assertCanonicalFormErrorNotRaised(operator=op)

    def test_simple_addition_tensor_prod_true(self):
        """Simple addition test of canonical operator with tensor product"""
        op =  1*(I@Y) + 2*(X@I)
        #self.assertTrue(test_function(operator = op, visitor=CanonicalizationVerificationAddition()))
        self.assertCanonicalFormErrorNotRaised(operator=op)

    def test_simple_addition_false(self):
        """Simple addition test of non canonical operator"""
        op =  1*I + X
        self.assertCanonicalFormErrorRaised(operator=op)

    def test_complicated_addition_true(self):
        """Complicated addition test of canonical operator"""
        op =  1*(I @ (A*A)) + 3*(X @ (A*A)) + 7*(Y @ (A*A)) + 6* (Z @ (A*A)) + 7 * (Z @ (A*C))
        self.assertCanonicalFormErrorNotRaised(operator=op)

    @unittest.skip("The issue of brackets seems to be resolved")
    def test_complicated_addition_passing_test_missing_bracket(self):
        # Note: removing some brackets gives errors because of tree structure what to do
        # This can be source of some ambiguity. 
        """Complicated addition test of canonical operator showing passing test with missing brackets"""
        op =  1*(I @ (A*A)) + 3*(X @ (A*A)) + 7*(Y @ A*A) + 6* (Z @ (A*A)) + 7 * (Z @ (A*C))
        self.assertCanonicalFormErrorNotRaised(operator=op)

    def test_complicated_addition_pass_with_all_bracket(self):
        """Complicated addition test of canonical operator without missing brackets"""
        op =  1*(I @ (A*A)) + 3*(X @ (A*A)) + 7*(Y @ (A*A)) + 6* (Z @ (A*A)) + 7 * (Z @ (A*C))
        self.assertCanonicalFormErrorNotRaised(operator=op)

    def test_complicated_addition_false(self):
        """Complicated addition test of non canonical operator"""
        op =  1*(I @ A*A) + 3*(X @ A*A) + 7*(Y @ A*A) + (Z @ A*A) + 7 * (Z @ A*C)
        self.assertCanonicalFormErrorRaised(operator=op)

    def test_pauli_tensor_and_ladder_multiplication_true(self):
        """Complicated pauli tensor product and ladder multiplication"""
        op = 2*(I @ (A*C) @ X @ (C*A*A*A*C*LI*A)) 
        self.assertCanonicalFormErrorNotRaised(operator=op)

    def test_pauli_tensor_and_ladder_multiplication_false_v1(self):
        """Complicated pauli tensor product and ladder multiplication with not in canonical form v1"""
        op = 2*(I @ (A*X) @ X @ (C*A*A*A*C*LI*A)) 
        self.assertCanonicalFormErrorRaised(operator=op)

    def test_pauli_tensor_and_ladder_multiplication_false_v2(self):
        """Complicated pauli tensor product and ladder multiplication with not in canonical form v2"""
        op = 2*(I @ (A*C) @ X @ (C*A*A*A*C*LI*A) @ (X * Y)) 
        self.assertCanonicalFormErrorRaised(operator=op)

    def test_pauli_tensor_and_ladder_multiplication_false_v3(self):
        """Complicated pauli tensor product and ladder multiplication with not in canonical form v3"""
        op = 2*(I @ (A*C) @ X @ (C*A*A*2*A*C*LI*A)) 
        self.assertCanonicalFormErrorRaised(operator=op)

    def test_scalar_operator_product_with_pauli(self): # produces `Incorrect canonical scalar operator multiplication` as expected
        """Nested product of Paulis"""
        op = (3*(3*(3*(A*A)))) + (2*(C*C))
        self.assertCanonicalFormErrorRaised(operator=op)

    def test_scalar_operator_product_with_ladder(self): # produces `Incorrect canonical scalar operator multiplication`` as expected
        """Nested product of Ladders"""
        op = (3*(3*(3*(X*Y)))) + (2*(X*Z))
        self.assertCanonicalFormErrorRaised(operator=op)
    
    def test_subtraction(self):
        op = 3*X - 2*Y
        self.assertCanonicalFormErrorRaised(operator=op)

    @unittest.skip("Not implemenyed yet")
    def test_proper(self):
        op = X @ (Y @ Z)
        self.assertCanonicalFormErrorRaised(operator=op)


@colorize(color=BLUE)
class TestCanonicalizationVerificationOperatorDistribute(CanonicalFormErrors, unittest.TestCase):
    maxDiff = None

    def test_pauli_simple_fail(self):
        """Simple failure with pauli"""
        op = ((2*X) + Y) @ Z
        self.assertCanonicalFormErrorRaised(operator=op, visitor=CanonicalizationVerificationOperatorDistribute())

    def test_pauli_ladder_simple_fail(self):
        """Simple failure with pauli and ladder"""
        op = ((2*X) + A) @ Z
        self.assertCanonicalFormErrorRaised(operator=op, visitor=CanonicalizationVerificationOperatorDistribute())

    def test_pauli_simple_pass(self):
        """Simple pass with pauli"""
        op = (2*X)@Z + Y@Z
        self.assertCanonicalFormErrorNotRaised(operator=op, visitor=CanonicalizationVerificationOperatorDistribute())

    def test_pauli_ladder_simple_pass(self):
        """Simple pass with pauli"""
        op = (2*X)@Z + Y@Z + 2*(A*A*A*C)*(C*A*A)
        self.assertCanonicalFormErrorNotRaised(operator=op, visitor=CanonicalizationVerificationOperatorDistribute())

    def test_pauli_nested_fail(self):
        """Simple failure with nested pauli"""
        op = Y*(X@(X@X*(X+X)))
        self.assertCanonicalFormErrorRaised(operator=op, visitor=CanonicalizationVerificationOperatorDistribute())

    def test_pauli_nested_pass(self):
        """Simple pass with nested pauli"""
        op  = 1*(I @ A*A) + 3*(X @ A*A) + 7*(Y @ A*A) + (Z @ A*A) + 7 * (Z @ A*C)
        self.assertCanonicalFormErrorNotRaised(operator=op, visitor=CanonicalizationVerificationOperatorDistribute())

    def test_nested_multiplication_pass(self):
        """Nested multiplication passing test"""
        op = 3*(2*(2+7)*7*(X@Y@Y@Z@(A*A*C*LI))) + (2j+7)*7*(X@Z@X@Z@(A*A*C*LI*A*A))
        self.assertCanonicalFormErrorNotRaised(operator=op, visitor=CanonicalizationVerificationOperatorDistribute())

    def test_complex_nested_pass(self):
        """Complicated pass with nested pauli and ladder"""
        op  = 1*(I @ A*A) + 3*(X @ A*A) + 7*(Y @ A*A) + (Z @ A*A*A*A*A*A*C+A*C*C*C) + 7 * (Z @ A*C)
        self.assertCanonicalFormErrorNotRaised(operator=op, visitor=CanonicalizationVerificationOperatorDistribute())

    def test_complex_nested_fail(self):
        """Complicated fail with nested pauli and ladder"""
        op  = 1*(I @ A*A) + 3*(X @ A*A) + 7*(Y @ A*A) + (Z @ (A*A*A*A*A*A*C+A*C*C*C)) + 7 * (Z @ A*C)
        self.assertCanonicalFormErrorRaised(operator=op, visitor=CanonicalizationVerificationOperatorDistribute())

    def test_complex_pauli_fail(self):
        """Complicated test with complex numers"""
        op = 2*(X@(7*Y*(1j+2)*Y)) + 6*(Z@(-3j*Y)) #2*(X@(7*Y*(1j)*Y)) + 6*(Z@(-3j*Y)) 
        self.assertCanonicalFormErrorNotRaised(operator=op, visitor=CanonicalizationVerificationOperatorDistribute())

    def test_addition_pauli_scalar_multiple_simple(self):
        """Pauli addition with scalar multiple passes"""
        op = 3* (X + Y)
        self.assertCanonicalFormErrorRaised(operator=op, visitor=CanonicalizationVerificationOperatorDistribute())

    def test_addition_pauli_scalar_multiple_nested(self):
        """Pauli addition with scalar multiple passes with nested operations"""
        op = 2j*(5*(3* (X + Y)) + 3* (Z @ A*A*C*A))
        self.assertCanonicalFormErrorRaised(operator=op, visitor=CanonicalizationVerificationOperatorDistribute())

    def test_subtraction_pauli(self):
        """Pauli Subtarction Fail"""
        op = X * (X - Y)
        self.assertCanonicalFormErrorRaised(operator=op, visitor=CanonicalizationVerificationOperatorDistribute())

@colorize(color=BLUE)
class TestCanonicalizationVerificationGatherMathExpr(CanonicalFormErrors, unittest.TestCase):
    maxDiff = None

    def test_pauli_simple_pass(self):
        """Simple Pauli passing"""
        op = 2*(X@Z) + Y@Z
        self.assertCanonicalFormErrorNotRaised(operator=op, visitor=CanonicalizationVerificationGatherMathExpr())

    def test_pauli_simple_fail(self):
        """Simple Pauli failing"""
        op = (2*X)@Z + Y@Z
        self.assertCanonicalFormErrorRaised(operator=op, visitor=CanonicalizationVerificationGatherMathExpr())

    def test_pauli_simple_fail_one_term(self):
        """Simple Pauli failing with 1 term"""
        op = X * ((2+3)*X)
        self.assertCanonicalFormErrorRaised(operator=op, visitor=CanonicalizationVerificationGatherMathExpr())

    def test_pauli_complicated_pass(self):
        """Complicated test passing with pauli and ladders"""
        op = 2j*(X @ (A * A * C) @ (Y @ (A*C*A*A*C*LI))) + (-1)*(A*C*A*A*C*LI)
        self.assertCanonicalFormErrorNotRaised(operator=op, visitor=CanonicalizationVerificationGatherMathExpr())

    def test_pauli_complicated_fail(self):
        """Complicated test failing with pauli and ladders"""
        op = 2j*(X @ (A * A * C) @ (Y @ (A*C*A*A*C*LI))) + (-1)*(A*C*A*(1*A)*C*LI)
        self.assertCanonicalFormErrorRaised(operator=op, visitor=CanonicalizationVerificationGatherMathExpr())

    def test_addition_pauli_scalar_multiple_nested_fail(self):
        """Pauli addition fail with scalar multiple passes with nested operations"""
        op = 2j*(5*(3* (X + Y)) + 3* (Z @ A*A*C*A))
        self.assertCanonicalFormErrorRaised(operator=op, visitor=CanonicalizationVerificationGatherMathExpr())

    def test_pauli_nested_ops_fail(self):
        """Nested scalar multiplications of Paulis fail"""
        op = (3*(3*(3*(X*Y))))
        self.assertCanonicalFormErrorRaised(operator=op, visitor=CanonicalizationVerificationGatherMathExpr())

    def test_pauli_ladder_nested_ops_pass(self):
        """Nested scalar multiplications pass"""
        op = (3 * 3 * 3) * ((X * Y) @ (A*C))
        self.assertCanonicalFormErrorNotRaised(operator=op, visitor=CanonicalizationVerificationGatherMathExpr())

@colorize(color=BLUE)
class TestCanonicalizationVerificationProperOrder(CanonicalFormErrors, unittest.TestCase):
    maxDiff = None

    def test_simple_pauli_pass(self):
        """Simple tensor product in proper order"""
        op = (X @ Y) @ Z
        self.assertCanonicalFormErrorNotRaised(operator=op, visitor=CanonicalizationVerificationProperOrder())

    def test_simple_pauli_pass(self):
        """Simple tensor product not in proper order"""
        op = X @ (Y @ Z)
        self.assertCanonicalFormErrorRaised(operator=op, visitor=CanonicalizationVerificationProperOrder())

    def test_simple_pauli_with_other_ops_pass(self):
        """Simple tensor product with other operations in proper order"""
        op = (X @ (2*Y + Z@(A*C*A))) @ Z
        self.assertCanonicalFormErrorNotRaised(operator=op, visitor=CanonicalizationVerificationProperOrder())

    def test_simple_pauli_with_other_ops_pass(self):
        """Simple tensor product with other operations not in proper order"""
        op = X @ ((2*Y + Z@(A*C*A)) @ Z)
        self.assertCanonicalFormErrorRaised(operator=op, visitor=CanonicalizationVerificationProperOrder())

    def test_nested_paulis_v1(self):
        """Nested structure with nested addition not in proper order"""
        op = I + (X + (2*Y)+Z)
        self.assertCanonicalFormErrorRaised(operator=op, visitor=CanonicalizationVerificationProperOrder())

    def test_nested_paulis_pass_v2(self):
        """Nested structure with nested addition in proper order"""
        op = (I + X) + ((2*Y) + Z)
        self.assertCanonicalFormErrorRaised(operator=op, visitor=CanonicalizationVerificationProperOrder())

    def test_nested_paulis_pass_with_brackets(self):
        """Nested structure with nested addition in proper order with brackets"""
        op = ((I + X) + (2*Y))+Z
        self.assertCanonicalFormErrorNotRaised(operator=op, visitor=CanonicalizationVerificationProperOrder())
        
    def test_nested_paulis_pass_without_brackets(self):
        """Nested structure with nested addition in proper order with brackets (i.e. using Python default)"""
        op = I + X + (2*Y) + Z
        self.assertCanonicalFormErrorNotRaised(operator=op, visitor=CanonicalizationVerificationProperOrder())

    def test_nested_tensor_prod_paulis_multiplication_fail(self):
        """Nested structure with nested multiplication (4j*(2...)) not in proper order"""
        op = (A @ C) @ (4j*(2 * (X + (2*Y)+Z)))
        self.assertCanonicalFormErrorRaised(operator=op, visitor=CanonicalizationVerificationProperOrder())

    def test_nested_tensor_prod_paulis_multiplication_pass(self):
        """Nested structure with nested multiplication (4j*...) in proper order"""
        op = (A @ C) @ ((4j* 2) * (X + (2*Y)+Z))
        self.assertCanonicalFormErrorNotRaised(operator=op, visitor=CanonicalizationVerificationProperOrder())

if __name__ == '__main__':
    unittest.main()
    #node = 2 * PauliX() @ (2 * PauliY() * 3) @ (MathStr(string='5*t') * PauliZ()) + (2 * PauliY() +  3 * PauliY()) @ (MathStr(string='5*t') * PauliZ())
    # X, Y, Z, I = PauliX(), PauliY(), PauliZ(), PauliI()
    # node = X  @ Y @ Annihilation()@ Z
    # pprint(node)
    # pprint(node.accept(SeparatePauliLadder()))
    # #pprint(node.accept(GatherMathExpr()))
    # pprint(node.accept(DeNestOpMulKron()).accept(DeNestOpMulKron()).accept(GatherMathExpr()).accept(GatherMathExpr()).accept(GatherMathExpr()).accept(ProperKronOrder()).accept(SeparatePauliLadder()).accept(PrintOperator()))