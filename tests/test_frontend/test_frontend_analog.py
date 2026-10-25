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

from oqd_core.frontend.analog import parse_analog, serialize_analog
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
    Pow,
    QuantumRegister,
    RuntimeVar,
    Sub,
    While,
)

########################################################################################


def test_program():
    circuit = parse_analog("")
    assert isinstance(circuit, AnalogCircuit)
    assert circuit.statements == []


## Declarations ##


class TestAnalogDeclarations:
    def test_int(self):
        circuit = parse_analog("x = 1")
        assert len(circuit.statements) == 1
        decl = circuit.statements[0]
        assert isinstance(decl, Declaration)
        assert decl.name == "x"
        assert decl.value == Constant(value=1)

    def test_float(self):
        circuit = parse_analog("pi = 3.14")
        decl = circuit.statements[0]
        assert isinstance(decl, Declaration)
        assert decl.name == "pi"
        assert decl.value == Constant(value=3.14)

    def test_math_var(self):
        circuit = parse_analog("omega = #omega")
        decl = circuit.statements[0]
        assert decl.value == RuntimeVar(name="#omega")

    def test_imag(self):
        circuit = parse_analog("z = 1j")
        decl = circuit.statements[0]
        assert decl.value == Constant(value=Complex(real=0, imag=1))

    def test_quantum_register(self):
        circuit = parse_analog("r = qreg(2)")
        decl = circuit.statements[0]
        assert decl.value == QuantumRegister(size=2)

    def test_mode_register(self):
        circuit = parse_analog("s = qmode(3)")
        decl = circuit.statements[0]
        assert decl.value == ModeRegister(size=3)

    def test_list(self):
        circuit = parse_analog("list = [1, 2, 3]")
        decl = circuit.statements[0]
        assert isinstance(decl.value, AnalogList)
        assert decl.value.values == [
            Constant(value=1),
            Constant(value=2),
            Constant(value=3),
        ]

    def test_list_extract(self):
        circuit = parse_analog("r = qreg(2)\nq0 = r[0]")
        decl = circuit.statements[1]
        print(decl)
        assert isinstance(decl.value, Extract)
        assert decl.value.value == Access(name="r")
        assert decl.value.index == Constant(value=0)


## Math Expressions ##


class TestAnalogMathExpressions:
    def test_addition(self):
        circuit = parse_analog("x = 1 + 2")
        decl = circuit.statements[0]
        print(decl)
        assert decl.value == Add(exprs=[Constant(value=1), Constant(value=2)])

    def test_subtraction(self):
        circuit = parse_analog("x = 5 - 3")
        decl = circuit.statements[0]
        assert decl.value == Sub(exprs=[Constant(value=5), Constant(value=3)])

    def test_multiplication(self):
        circuit = parse_analog("x = 2 * 3")
        decl = circuit.statements[0]
        assert decl.value == Mul(exprs=[Constant(value=2), Constant(value=3)])

    def test_division(self):
        circuit = parse_analog("x = 6 / 2")
        decl = circuit.statements[0]
        assert decl.value == Div(exprs=[Constant(value=6), Constant(value=2)])

    def test_power(self):
        circuit = parse_analog("x = 2^3")
        decl = circuit.statements[0]
        assert decl.value == Pow(exprs=[Constant(value=2), Constant(value=3)])

    def test_negation(self):
        circuit = parse_analog("x = -1")
        decl = circuit.statements[0]
        assert decl.value == Neg(expr=Constant(value=1))

    def test_nested_expression(self):
        circuit = parse_analog("x = 2 * 3 + 1")
        decl = circuit.statements[0]
        expected = Add(exprs=[
            Mul(exprs=[Constant(value=2), Constant(value=3)]),
            Constant(value=1),
            ]
        )
        assert decl.value == expected

    def test_paranthesis_expression(self):
        circuit = parse_analog("x = 2 * (3 + 1)")
        decl = circuit.statements[0]
        expected = Mul(exprs=[
            Constant(value=2),
            Add(exprs=[Constant(value=3), Constant(value=1)]),
            ]
        )
        assert decl.value == expected

    @pytest.mark.parametrize(
        "func_name",
        [
            "sin",
            "cos",
            "tan",
            "exp",
            "log",
            "abs",
            "sinh",
            "cosh",
            "tanh",
            "atan",
            "acos",
            "asin",
            "atanh",
            "asinh",
            "acosh",
            "conj",
            "heaviside",
            "real",
            "imag",
        ],
    )
    def test_unary_math_function(self, func_name):
        circuit = parse_analog(f"x = {func_name}(1)")
        decl = circuit.statements[0]
        assert isinstance(decl.value, BuiltinCall)
        assert decl.value.func == func_name
        assert decl.value.args == [Constant(value=1)]

    def test_atan2(self):
        circuit = parse_analog("x = atan2(1, 2)")
        decl = circuit.statements[0]
        assert isinstance(decl.value, BuiltinCall)
        assert decl.value.func == "atan2"
        assert decl.value.args == [Constant(value=1), Constant(value=2)]


## Bool Expressions ##


class TestAnalogBool:
    def test_true(self):
        circuit = parse_analog("x = true")
        assert circuit.statements[0].value == Constant(value=True)

    def test_false(self):
        circuit = parse_analog("x = false")
        assert circuit.statements[0].value == Constant(value=False)

    @pytest.mark.parametrize(
        "op, cls",
        [
            ("==", Eq),
            ("!=", Neq),
            ("<=", Leq),
            ("<", Lt),
            ("and", And),
            ("&&", And),
            ("or", Or),
            ("||", Or),
            (">=", Geq),
            (">", Gt),
        ],
    )
    def test_comparison(self, op, cls):
        circuit = parse_analog(f"x = 1 {op} 2")
        decl = circuit.statements[0]
        assert isinstance(decl, Declaration)
        assert isinstance(decl.value, cls)

    @pytest.mark.parametrize(
        "op, cls",
        [
            ("not", Not),
            ("!", Not),
        ],
    )
    def test_not(self, op, cls):
        circuit = parse_analog(f"a = true \n x = {op} a")
        decl = circuit.statements[1]
        assert isinstance(decl, Declaration)
        assert isinstance(decl.value, cls)


## Analog Operators ##


class TestAnalogOperators:
    def test_pauli_x(self):
        circuit = parse_analog("op = %X")
        assert circuit.statements[0].value == PauliX()

    def test_pauli_y(self):
        circuit = parse_analog("op = %Y")
        assert circuit.statements[0].value == PauliY()

    def test_pauli_z(self):
        circuit = parse_analog("op = %Z")
        assert circuit.statements[0].value == PauliZ()

    def test_pauli_i(self):
        circuit = parse_analog("op = %I")
        assert circuit.statements[0].value == PauliI()

    def test_creation(self):
        circuit = parse_analog("op = %C")
        assert circuit.statements[0].value == Creation()

    def test_annihilation(self):
        circuit = parse_analog("op = %A")
        assert circuit.statements[0].value == Annihilation()

    def test_identity(self):
        circuit = parse_analog("op = %J")
        assert circuit.statements[0].value == Identity()

    def test_operator_add(self):
        circuit = parse_analog("op = %X + %Y")
        assert circuit.statements[0].value == Add(exprs=[PauliX(), PauliY()])

    def test_operator_sub(self):
        circuit = parse_analog("op = %X - %Y")
        assert circuit.statements[0].value == Sub(exprs=[PauliX(), PauliY()])

    def test_operator_mul(self):
        circuit = parse_analog("op = %X * %Y")
        assert circuit.statements[0].value == Mul(exprs=[PauliX(), PauliY()])

    def test_operator_kron(self):
        circuit = parse_analog("op = %X @ %Y")
        assert circuit.statements[0].value == Kron(exprs=[PauliX(), PauliY()])


## Statements ##


class TestAnalogStatements:
    @pytest.fixture
    def register(self):
        return "r = qreg(2)\n"

    def test_initialize(self, register):
        circuit = parse_analog(register + "initialize(r)")
        statement = circuit.statements[1]
        assert isinstance(statement, Initialize)
        assert statement.targets == Access(name="r")

    def test_measure(self, register):
        circuit = parse_analog(register + "measure(r)")
        statement = circuit.statements[1]
        assert isinstance(statement, Measure)
        assert statement.targets == Access(name="r")

    def test_evolve(self, register):
        circuit = parse_analog(register + "evolve(%X, 1.0, r)")
        statement = circuit.statements[1]
        assert isinstance(statement, Evolve)
        assert statement.hamiltonian == PauliX()
        assert statement.duration == Constant(value=1.0)
        assert statement.targets == Access(name="r")

    def test_multiple_statements(self, register):
        program = "\n".join(
            [
                register,
                "initialize(r)",
                "evolve(%X, 1.0, r)",
                "measure(r)",
            ]
        )
        circuit = parse_analog(program)
        assert len(circuit.statements) == 4
        assert isinstance(circuit.statements[0], Declaration)
        assert isinstance(circuit.statements[1], Initialize)
        assert isinstance(circuit.statements[2], Evolve)
        assert isinstance(circuit.statements[3], Measure)


## Control Flow Statements ##


class TestAnalogControlFlow:
    def test_if(self):
        program = "x = 1\n if (x > 0) {\n y = 2\n}"
        circuit = parse_analog(program)
        ifelse = circuit.statements[1]
        assert isinstance(ifelse, IfElse)
        assert isinstance(ifelse.condition, Gt)
        assert len(ifelse.then_branch) == 1
        assert ifelse.else_branch == []

    def test_if_else(self):
        program = "x = 1\n if (x > 0) {\n y = 2\n} \n else {\n y = 3\n}"
        circuit = parse_analog(program)
        ifelse = circuit.statements[1]
        assert isinstance(ifelse, IfElse)
        assert isinstance(ifelse.condition, Gt)
        assert len(ifelse.then_branch) == 1
        assert len(ifelse.else_branch) == 1

    def test_while_statement(self):
        program = "n = 3\nwhile (n > 0) {\n    n = n - 1\n}"
        circuit = parse_analog(program)
        while_statement = circuit.statements[1]
        assert isinstance(while_statement, While)
        assert isinstance(while_statement.condition, Gt)
        assert len(while_statement.body) == 1

    def test_break_loop(self):
        program = "while (true) {\n break\n}"
        circuit = parse_analog(program)
        while_statement = circuit.statements[0]
        assert isinstance(while_statement.body[0], Break)

    def test_continue_loop(self):
        program = "while (true) {\n continue\n}"
        circuit = parse_analog(program)
        while_statement = circuit.statements[0]
        assert isinstance(while_statement.body[0], Continue)

    def test_break_outside_loop(self):
        with pytest.raises(SyntaxError, match="break outside of loop"):
            parse_analog("break")

    def test_continue_outside_loop(self):
        with pytest.raises(SyntaxError, match="continue outside of loop"):
            parse_analog("continue")

    def test_nested_control_flow(self):
        program = "while(true) {\n if (a == b) {x = 0} \n if (x == 0) { break}\n}"
        circuit = parse_analog(program)
        while_statement = circuit.statements[0]
        assert isinstance(while_statement, While)
        assert while_statement.condition == Constant(value=True)
        ifelse = while_statement.body[0]
        assert isinstance(ifelse, IfElse)
        assert isinstance(ifelse.condition, Eq)
        assert isinstance(ifelse.then_branch[0], Declaration)
        ifelse = while_statement.body[1]
        assert isinstance(ifelse, IfElse)
        assert isinstance(ifelse.condition, Eq)
        assert isinstance(ifelse.then_branch[0], Break)


## Serialization ##


class TestAnalogSerialize:
    @pytest.mark.parametrize(
        "program",
        [
            "r = qreg(2)",
            "list = [1, 2, 3]",
            "initialize(r)",
            "evolve(%X, 1.0, r)",
            "measure(r)",
            "x = 1\n if (x > 0) {\n y = 2\n}",
            "x = 1\n if (x > 0) {\n y = 2\n} \n else {\n y = 3\n}",
            "while(true) {\n if (a == b) {x = 0} \n if (x == 0) { break}\n}",
        ],
    )
    def test_analog_serialize(self, program):
        circuit = parse_analog(program)
        assert isinstance(circuit, AnalogCircuit)
        serialized = serialize_analog(circuit)
        deserialized_circuit = parse_analog(serialized)
        assert circuit == deserialized_circuit
