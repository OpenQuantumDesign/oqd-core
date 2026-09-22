# Copyright 2024-2025 Open Quantum Design

# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import re

from oqd_compiler_infrastructure import ConversionRule, Post

from oqd_core.interface.analog import (
    Access,
    Add,
    AnalogCircuit,
    AnalogExpr,
    AnalogList,
    And,
    Annihilation,
    Break,
    BuiltinCall,
    CastAnalogExpr,
    Complex,
    Constant,
    Continue,
    Creation,
    Declaration,
    Div,
    Eq,
    Evolve,
    Extract,
    Geq,
    Gt,
    Identity,
    IfElse,
    Initialize,
    Kron,
    Leq,
    Lt,
    Measure,
    ModeRegister,
    Mul,
    Neg,
    Neq,
    Not,
    Or,
    PauliI,
    PauliX,
    PauliY,
    PauliZ,
    Pos,
    Pow,
    QuantumRegister,
    RuntimeVar,
    Sub,
    While,
    Xor,
)
from oqd_core.interface.analog.expr import BinaryOp, UnaryOp

########################################################################################

ARITH_OP_MAPPING = {
    Pow: (1, "^"),
    Pos: (2, "+"),
    Neg: (2, "-"),
    Not: (2, "!"),
    Mul: (3, "*"),
    Div: (3, "/"),
    Kron: (3, "@"),
    Add: (4, "+"),
    Sub: (4, "-"),
    Lt: (5, "<"),
    Leq: (5, "<="),
    Gt: (5, ">"),
    Geq: (5, ">="),
    Eq: (6, "=="),
    Neq: (6, "!="),
    And: (7, "&&"),
    Xor: (8, "^^"),
    Or: (9, "||"),
}


class SerializeAnalog(ConversionRule):
    def _parenthesize_precedence(self, expr, inner_type, outer_type):
        inner_precedence = ARITH_OP_MAPPING.get(inner_type, (0, ""))[0]
        outer_precedence = ARITH_OP_MAPPING.get(outer_type, (0, ""))[0]

        if inner_precedence > outer_precedence:
            return f"({expr})"

        return expr

    def generic_map(self, model, operands):
        return str(model)

    def map_AnalogCircuit(self, model: AnalogCircuit, operands):
        statements = operands["statements"]
        return "\n".join(statements) + "\n"

    def map_Declaration(self, model: Declaration, operands):
        return f"{operands['name']} = {operands['value']}"

    def map_While(self, model: While, operands):
        body = "\n".join(operands["body"]) + "\n"
        return f"while ({operands['condition']}) {{\n{body}}}"

    def map_IfElse(self, model: IfElse, operands):
        then_branch = "\n".join(operands["then_branch"])
        else_branch = operands["else_branch"]
        if else_branch:
            else_branch = "\n".join(else_branch)
            return f"if ({operands['condition']}) {{\n{then_branch}\n}} else {{\n{else_branch}\n}}"
        return f"if ({operands['condition']}) {{\n{then_branch}\n}}"

    def map_Break(self, model: Break, operands):
        return "break"

    def map_Continue(self, model: Continue, operands):
        return "continue"

    def map_Evolve(self, model: Evolve, operands):
        return f"evolve({operands['hamiltonian']}, {operands['duration']}, {operands['targets']})"

    def map_Measure(self, model: Measure, operands):
        return f"measure({operands['targets']})"

    def map_Initialize(self, model: Initialize, operands):
        return f"initialize({operands['targets']})"

    def map_AnalogList(self, model: AnalogList, operands):
        return "[" + ", ".join(operands["values"]) + "]"

    def map_Extract(self, model: Extract, operands):
        return f"{operands['access']}[{operands['index']}]"

    def map_Access(self, model: Access, operands):
        return operands["name"]

    def map_QuantumRegister(self, model: QuantumRegister, operands):
        return f"qreg({operands['size']})"

    def map_ModeRegister(self, model: ModeRegister, operands):
        return f"qmode({operands['size']})"

    def map_RuntimeVar(self, model: RuntimeVar, operands):
        return operands["name"]

    def map_Constant(self, model: Constant, operands):
        return operands["value"]

    def map_bool(self, model: bool, operands):
        return str(model).lower()

    def map_Complex(self, model: Complex, operands):
        real = operands["real"]
        imag = operands["imag"]

        if model.imag == 0:
            return f"{real}r"

        if model.real == 0:
            return f"{imag}j"

        return f"({real}r{imag}j)"

    def map_BuiltinCall(self, model: BuiltinCall, operands):
        func = model.func
        return f"{func}({', '.join(operands['args'])})"

    def map_BinaryOp(self, model: BinaryOp, operands):
        expr1 = self._parenthesize_precedence(
            operands["expr1"], model.expr1.__class__, model.__class__
        )
        expr2 = self._parenthesize_precedence(
            operands["expr2"], model.expr2.__class__, model.__class__
        )

        return f"{expr1} {ARITH_OP_MAPPING[model.__class__][1]} {expr2}"

    def map_UnaryOp(self, model: UnaryOp, operands):
        expr = self._parenthesize_precedence(
            operands["expr"], model.expr.__class__, model.__class__
        )

        return f"{ARITH_OP_MAPPING[model.__class__][1]}{expr}"

    def map_PauliI(self, model: PauliI, operands):
        return "%I"

    def map_PauliX(self, model: PauliX, operands):
        return "%X"

    def map_PauliY(self, model: PauliY, operands):
        return "%Y"

    def map_PauliZ(self, model: PauliZ, operands):
        return "%Z"

    def map_Creation(self, model: Creation, operands):
        return "%C"

    def map_Annihilation(self, model: Annihilation, operands):
        return "%A"

    def map_Identity(self, model: Identity, operands):
        return "%J"


########################################################################################

serialize_analog = Post(SerializeAnalog())
