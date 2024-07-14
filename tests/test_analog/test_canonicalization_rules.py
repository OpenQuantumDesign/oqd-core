from quantumion.interface.analog import *
from quantumion.compilerv2.canonical_verification import *
from quantumion.compiler.analog.verify import *
from quantumion.compiler.analog.error import *
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

def test_function(operator: Operator, rule : RewriteRule, walk_method = Post):
    return FixedPoint(walk_method(rule))(operator)


@colorize(color=BLUE)
class TestOperatorDistribute(unittest.TestCase):
    maxDiff = None

    def __init__(self, methodName: GREEN = "runTest") -> None:
        super().__init__(methodName)
        self.rule = OperatorDistribute()

    def test_simple(self):
        """Simple test"""
        op = X@(X+Y)
        expected = X@X + X@Y
        self.assertEqual(test_function(operator = op, rule = self.rule), expected)


if __name__ == '__main__':
    unittest.main()