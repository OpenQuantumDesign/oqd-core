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

from oqd_core.interface.analog.expr import (
    Annihilation,
    Creation,
    Identity,
    Ladder,
    MathAdd,
    MathDiv,
    MathFunc,
    MathImag,
    MathMul,
    MathNum,
    MathPow,
    MathSub,
    MathVar,
    OperatorAdd,
    OperatorKron,
    OperatorMul,
    OperatorSub,
    OperatorTerminal,
    Pauli,
    PauliI,
    PauliX,
    PauliY,
    PauliZ,
    Access,
)
from oqd_core.compiler.analog.error import AnalogCompilerError

MATH_EXPR_TYPES = (
    MathNum,
    MathImag,
    MathVar,
    MathAdd,
    MathSub,
    MathMul,
    MathDiv,
    MathPow,
    MathFunc,
)

OPERATOR_EXPR_TYPES = (
    Pauli,
    PauliI,
    PauliX,
    PauliY,
    PauliZ,
    Annihilation,
    Creation,
    Identity,
    OperatorAdd,
    OperatorMul,
    OperatorKron,
    OperatorSub,
)


def scalar_mul(scalar, op):
    return OperatorMul(op1=scalar, op2=op)


def is_scalar_mul(node) -> bool:
    if isinstance(node, OperatorMul):
        if (isinstance(node.op1, MATH_EXPR_TYPES) and isinstance(node.op2, OPERATOR_EXPR_TYPES)) or \
            (isinstance(node.op2, MATH_EXPR_TYPES) and isinstance(node.op1, OPERATOR_EXPR_TYPES)):
            return True
    return False


def coeff_and_op(node):
    if is_scalar_mul(node):
        if isinstance(node.op1, MATH_EXPR_TYPES):
            return node.op1, node.op2
        return node.op2, node.op1
    if isinstance(node, OPERATOR_EXPR_TYPES):
        return MathNum(value=1), node
    return None, None


def factor_dim(node):
    if isinstance(node, (Pauli, OperatorMul, Ladder)):
        return 1
    if isinstance(node, OperatorKron):
        d1 = factor_dim(node.op1)
        d2 = factor_dim(node.op2)
        return d1 + d2


def term_dim(expr):
    _, op = coeff_and_op(expr)
    if op is None:
        return None
    if isinstance(op, (OperatorTerminal, OperatorMul, OperatorKron)):
        return factor_dim(op)


def operator_dim(expr):
    
    ref = None
    curr = expr
    
    while isinstance(curr, OperatorAdd):
        dim = term_dim(curr.op2)
        if ref is None:
            ref = dim
        elif ref != dim:
            raise AnalogCompilerError("Incorrect Hilbert space dimension")
        curr = curr.op1
        
    last = term_dim(curr)
    if ref is None:
        return last
    if ref != last:
        raise AnalogCompilerError("Incorrect Hilbert space dimension")
    return ref
            
