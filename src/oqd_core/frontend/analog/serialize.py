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

from __future__ import annotations

from oqd_compiler_infrastructure import ConversionRule, Post

from oqd_core.interface.analog import (
    Access,
    AnalogCircuit,
    AnalogList,
    Bool,
    BoolAnd,
    BoolEq,
    BoolGreaterThan,
    BoolGreaterThanEq,
    BoolLessThan,
    BoolLessThanEq,
    BoolNot,
    BoolNotEq,
    BoolOr,
    Break,
    Continue,
    Declaration,
    Evolve,
    Extract,
    IfElse,
    Initialize,
    MathAdd,
    MathDiv,
    MathFunc,
    MathImag,
    MathMul,
    MathNum,
    MathPow,
    MathSub,
    MathVar,
    Measure,
    ModeRegister,
    QuantumRegister,
    While,
)
from oqd_core.interface.analog.expr import (
    Annihilation,
    Creation,
    Identity,
    OperatorAdd,
    OperatorKron,
    OperatorMul,
    OperatorSub,
    PauliI,
    PauliX,
    PauliY,
    PauliZ,
)

########################################################################################


class SerializeAnalog(ConversionRule):
    def generic_map(self, model, operands):
        if model is None or isinstance(model, (str, int, float, bool)):
            return model
        raise TypeError(f"Unsupported node: {model}")

    def map_AnalogCircuit(self, model: AnalogCircuit, operands):
        statements = operands["statements"]
        return "\n".join(statements) + "\n"

    ## Statements ##

    def map_Declaration(self, model: Declaration, operands):
        return f"{operands['name']} = {operands['value']};"

    def map_While(self, model: While, operands):
        body = "\n".join(operands["body"])
        return f"while ({operands['condition']}) {{\n{body}}}"

    def map_IfElse(self, model: IfElse, operands):
        then_branch = "\n".join(operands["then_branch"])
        else_branch = operands["else_branch"]
        if else_branch:
            else_branch = "\n".join(else_branch)
            return f"if ({operands['condition']}) {{\n{then_branch}\n}} else {{\n{else_branch}\n}}"
        return f"if ({operands['condition']}) {{\n{then_branch}\n}}"

    def map_Break(self, model: Break, operands):
        return "break;"

    def map_Continue(self, model: Continue, operands):
        return "continue;"

    ## Expressions ##

    def map_Evolve(self, model: Evolve, operands):
        return f"evolve({operands['hamiltonian']}, {operands['duration']}, {operands['targets']});"

    def map_Measure(self, model: Measure, operands):
        return f"measure({operands['targets']});"

    def map_Initialize(self, model: Initialize, operands):
        return f"initialize({operands['targets']});"

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

    ## Math ##

    def map_MathVar(self, model: MathVar, operands):
        return operands["name"]

    def map_MathNum(self, model: MathNum, operands):
        value = ""
        if isinstance(model.value, bool):
            value = str(int(model.value))
        if isinstance(model.value, (int, float)):
            value = str(model.value)
        return value

    def map_MathImag(self, model: MathImag, operands):
        return "1j"

    def map_MathFunc(self, model: MathFunc, operands):
        func = model.func
        expr = operands["expr"]
        if func == "atan2":
            return f"atan2({expr[0]}, {expr[1]})"
        return f"{func}({expr})"

    def map_MathAdd(self, model: MathAdd, operands):
        return f"{operands['expr1']} + {operands['expr2']}"

    def map_MathSub(self, model: MathSub, operands):
        expr = operands["expr2"]
        if isinstance(model.expr2, (MathAdd, MathSub)):
            expr = f"({expr})"
        return f"{operands['expr1']} - {expr}"

    def map_MathMul(self, model: MathMul, operands):
        if isinstance(model.expr1, MathNum) and model.expr1.value == -1:
            inner = operands["expr2"]
            if isinstance(model.expr2, (MathAdd, MathSub)):
                inner = f"({inner})"
            return f"-{inner}"
        left = operands["expr1"]
        right = operands["expr2"]
        if isinstance(model.expr1, (MathAdd, MathSub)):
            left = f"({left})"
        if isinstance(model.expr2, (MathAdd, MathSub)):
            right = f"({right})"
        return f"{left} * {right}"

    def map_MathDiv(self, model: MathDiv, operands):
        left = operands["expr1"]
        right = operands["expr2"]
        if isinstance(model.expr1, (MathAdd, MathSub)):
            left = f"({left})"
        if isinstance(model.expr2, (MathAdd, MathSub)):
            right = f"({right})"
        return f"{left} / {right}"

    def map_MathPow(self, model: MathPow, operands):
        left = operands["expr1"]
        right = operands["expr2"]
        if isinstance(model.expr1, (MathAdd, MathSub, MathMul, MathDiv)):
            left = f"({left})"
        if isinstance(model.expr2, (MathAdd, MathSub, MathMul, MathDiv)):
            right = f"({right})"
        return f"{left} ^ {right}"

    ## Bool ##

    def map_Bool(self, model: Bool, operands):
        return "true" if model.value else "false"

    def map_BoolNot(self, model: BoolNot, operands):
        inner = operands["expr"]
        if isinstance(
            model.expr,
            (
                BoolAnd,
                BoolOr,
                BoolNot,
                BoolEq,
                BoolNotEq,
                BoolLessThan,
                BoolLessThanEq,
                BoolGreaterThan,
                BoolGreaterThanEq,
            ),
        ):
            inner = f"({inner})"
        return f"not {inner}"

    def map_BoolAnd(self, model: BoolAnd, operands):
        left = operands["expr1"]
        right = operands["expr2"]
        if isinstance(model.expr1, BoolOr):
            left = f"({left})"
        if isinstance(model.expr2, BoolOr):
            right = f"({right})"
        return f"{left} and {right}"

    def map_BoolOr(self, model: BoolOr, operands):
        return f"{operands['expr1']} or {operands['expr2']}"

    def map_BoolEq(self, model: BoolEq, operands):
        return f"{operands['expr1']} == {operands['expr2']}"

    def map_BoolNotEq(self, model: BoolNotEq, operands):
        return f"{operands['expr1']} != {operands['expr2']}"

    def map_BoolLessThan(self, model: BoolLessThan, operands):
        return f"{operands['expr1']} < {operands['expr2']}"

    def map_BoolLessThanEq(self, model: BoolLessThanEq, operands):
        return f"{operands['expr1']} <= {operands['expr2']}"

    def map_BoolGreaterThan(self, model: BoolGreaterThan, operands):
        return f"{operands['expr1']} > {operands['expr2']}"

    def map_BoolGreaterThanEq(self, model: BoolGreaterThanEq, operands):
        return f"{operands['expr1']} >= {operands['expr2']}"

    ## Analog Operators ##

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

    def map_OperatorAdd(self, model: OperatorAdd, operands):
        return f"{operands['op1']} %+ {operands['op2']}"

    def map_OperatorSub(self, model: OperatorSub, operands):
        right = operands["op2"]
        if isinstance(model.op2, (OperatorAdd, OperatorSub)):
            right = f"({right})"
        return f"{operands['op1']} %- {right}"

    def map_OperatorMul(self, model: OperatorMul, operands):
        left = operands["op1"]
        right = operands["op2"]
        if isinstance(model.op1, (OperatorAdd, OperatorSub, OperatorKron)):
            left = f"({left})"
        if isinstance(model.op2, (OperatorAdd, OperatorSub, OperatorKron)):
            right = f"({right})"
        return f"{left} %* {right}"

    def map_OperatorKron(self, model: OperatorKron, operands):
        left = operands["op1"]
        right = operands["op2"]
        if isinstance(model.op1, (OperatorAdd, OperatorSub, OperatorMul)):
            left = f"({left})"
        if isinstance(model.op2, (OperatorAdd, OperatorSub, OperatorMul)):
            right = f"({right})"
        return f"{left} %@ {right}"


########################################################################################

serialize_analog = Post(SerializeAnalog())
