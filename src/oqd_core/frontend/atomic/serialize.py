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

from oqd_core.interface.atomic import (
    Access,
    Add,
    And,
    AtomicCircuit,
    AtomicList,
    Beam,
    Break,
    BuiltinCall,
    Complex,
    Constant,
    Continue,
    Declaration,
    Div,
    Eq,
    Extract,
    Geq,
    Gt,
    IfElse,
    IonRegister,
    Leq,
    Lt,
    Mul,
    Neg,
    Neq,
    Not,
    Or,
    ParallelProtocol,
    Pos,
    Pow,
    Pulse,
    RuntimeVar,
    SerialProtocol,
    Sub,
    While,
    Xor,
)
from oqd_core.interface.atomic.expr import BinaryOp, UnaryOp

########################################################################################

__all__ = ["SerializeAtomic", "serialize_atomic"]

########################################################################################

ARITH_OP_MAPPING = {
    Pow: (1, "^"),
    Pos: (2, "+"),
    Neg: (2, "-"),
    Not: (2, "!"),
    Mul: (3, "*"),
    Div: (3, "/"),
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

class SerializeAtomic(ConversionRule):
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

    def map_AtomicCircuit(self, model: AtomicCircuit, operands):
        statements = operands["statements"]
        return "\n".join(statements) + "\n"

    ## Statements ##

    def map_ParallelProtocol(self, model: ParallelProtocol, operands):
        body_str = self._indent_block(operands["pulses"])
        return f"parallel {{{body_str}}}"

    def map_SerialProtocol(self, model: SerialProtocol, operands):
        body_str = self._indent_block(operands["pulses"])
        return f"serial {{{body_str}}}"

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

    ## Expressions ##

    def map_Beam(self, model: Beam, operands):
        frequency = operands["frequency"]
        rabi = operands["rabi"]
        phase = operands["phase"]
        polarization = operands["polarization"]
        wavevector = operands["wavevector"]
        return f"beam({frequency}, {rabi}, {phase}, {polarization}, {wavevector})"

    def map_Pulse(self, model: Pulse, operands):
        beam = operands["beam"]
        duration = operands["duration"]
        measured = operands["measured"]
        target = operands["target"]
        return f"pulse({beam}, {duration}, {target}, {measured})"

    def map_AtomicList(self, model: AtomicList, operands):
        return "[" + ", ".join(operands["values"]) + "]"

    def map_Extract(self, model: Extract, operands):
        return f"{operands['value']}[{operands['index']}]"

    def map_Access(self, model: Access, operands):
        return operands["name"]

    def map_IonRegister(self, model: IonRegister, operands):
        return f"ionreg({operands['size']})"

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

        return f"{real}r{imag}j"

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

########################################################################################

serialize_atomic = Post(SerializeAtomic())
