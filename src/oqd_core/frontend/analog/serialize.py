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

from oqd_compiler_infrastructure import ConversionRule, Post

from oqd_core.interface.analog import (
    Access,
    Add,
    AnalogCircuit,
    AnalogList,
    And,
    Annihilation,
    Break,
    BuiltinCall,
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
from oqd_core.interface.analog.expr import BinaryOp, Pauli, UnaryOp

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
    def __init__(self, indent=2):
        super().__init__()
        self.indent = indent

    def generic_map(self, model, operands):
        return str(model)

    def _indent_block(self, body):
        body_str = "\n".join(body)
        body_str = "\n".join(
            map(lambda x: " " * self.indent + x, body_str.splitlines())
        )

        if body_str:
            body_str = "\n" + body_str + "\n"

        return body_str

    def _parenthesize_precedence(self, expr, inner_type, outer_type):
        inner_precedence = ARITH_OP_MAPPING.get(inner_type, (0, ""))[0]
        outer_precedence = ARITH_OP_MAPPING.get(outer_type, (0, ""))[0]

        if inner_precedence > outer_precedence:
            return f"({expr})"

        return expr

    def map_AnalogCircuit(self, model: AnalogCircuit, operands):
        statements = operands["statements"]

        return "\n".join(statements) + "\n"

    def map_Declaration(self, model: Declaration, operands):
        return f"{operands['name']} = {operands['value']}"

    def map_While(self, model: While, operands):
        body_str = self._indent_block(operands["body"])
        return f"while ({operands['condition']}) {{{body_str}}}"

    def map_IfElse(self, model: IfElse, operands):
        then_str = self._indent_block(operands["then_branch"])
        else_str = self._indent_block(operands["else_branch"])

        return f"if ({operands['condition']}) {{{then_str}}}\n" + (
            f"else {{{else_str}}}" if else_str else ""
        )

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
        exprs = [
            self._parenthesize_precedence(s, expr.__class__, model.__class__)
            for s, expr in zip(operands["exprs"], model.exprs)
        ]

        return f" {ARITH_OP_MAPPING[model.__class__][1]} ".join(exprs)

    def map_UnaryOp(self, model: UnaryOp, operands):
        expr = self._parenthesize_precedence(
            operands["expr"], model.expr.__class__, model.__class__
        )

        return f"{ARITH_OP_MAPPING[model.__class__][1]}{expr}"

    def map_Pauli(self, model: Pauli, operands):
        args = (model.level1, model.level2, model.dim)
        args_string = (operands["level1"], operands["level2"], operands["dim"])

        pauli = {PauliX: "%X", PauliY: "%Y", PauliZ: "%Z", PauliI: "%I"}[
            model.__class__
        ]

        if args[2] != 2:
            return f"{pauli}({', '.join(args_string)})"

        if args[:2] != [0, 1]:
            return f"{pauli}({', '.join(args_string[:2])})"

        return pauli

    def map_Creation(self, model: Creation, operands):
        return "%C"

    def map_Annihilation(self, model: Annihilation, operands):
        return "%A"

    def map_Identity(self, model: Identity, operands):
        return "%J"


########################################################################################

serialize_analog = Post(SerializeAnalog())
