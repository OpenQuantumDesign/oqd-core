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

from oqd_core.frontend.analog.AnalogCircuitAST import parse_analog
from oqd_core.frontend.analog.cfg import AnalogCFGBuilder
from oqd_core.frontend.analog.serialize import serialize_analog
from oqd_core.frontend.analog.type_checker import AnalogTypeChecker, AnalogTypeError
from oqd_core.frontend.analysis.utils import SCCAnalysis
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
        assert decl.value == MathNum(value=1)
    
    def test_float(self):
        circuit = parse_analog("pi = 3.14")
        decl = circuit.statements[0]
        assert isinstance(decl, Declaration)
        assert decl.name == "pi"
        assert decl.value == MathNum(value=3.14)
    
    def test_math_var(self):
        circuit = parse_analog("omega = #omega")
        decl = circuit.statements[0]
        assert decl.value == MathVar(name="#omega")
    
    def test_imag(self):
        circuit = parse_analog("z = 1j")
        decl = circuit.statements[0]
        assert decl.value == MathImag()
    
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
        assert decl.value.values == [MathNum(value=1), MathNum(value=2), MathNum(value=3)]

    def test_list_extract(self):
        circuit = parse_analog("r = qreg(2)\nq0 = r[0]")
        decl = circuit.statements[1]
        assert isinstance(decl.value, Extract)
        assert decl.value.access == Access(name="r")
        assert decl.value.index == 0
    
## Math Expressions ##

class TestAnalogMathExpressions:
    
    def test_addition(self):
        circuit = parse_analog("x = 1 + 2")
        decl = circuit.statements[0]
        assert decl.value == MathAdd(expr1=MathNum(value=1), expr2=MathNum(value=2))

    def test_subtraction(self):
        circuit = parse_analog("x = 5 - 3")
        decl = circuit.statements[0]
        assert decl.value == MathSub(expr1=MathNum(value=5), expr2=MathNum(value=3))

    def test_multiplication(self):
        circuit = parse_analog("x = 2 * 3")
        decl = circuit.statements[0]
        assert decl.value == MathMul(expr1=MathNum(value=2), expr2=MathNum(value=3))

    def test_division(self):
        circuit = parse_analog("x = 6 / 2")
        decl = circuit.statements[0]
        assert decl.value == MathDiv(expr1=MathNum(value=6), expr2=MathNum(value=2))

    def test_power(self):
        circuit = parse_analog("x = 2^3")
        decl = circuit.statements[0]
        assert decl.value == MathPow(expr1=MathNum(value=2), expr2=MathNum(value=3))

    def test_negation(self):
        circuit = parse_analog("x = -1")
        decl = circuit.statements[0]
        assert decl.value == MathMul(expr1=MathNum(value=-1), expr2=MathNum(value=1))
    
    def test_nested_expression(self):
        circuit = parse_analog("x = 2 * 3 + 1")
        decl = circuit.statements[0]
        expected = MathAdd(
            expr1=MathMul(expr1=MathNum(value=2), expr2=MathNum(value=3)),
            expr2=MathNum(value=1),
        )
        assert decl.value == expected
    
    def test_paranthesis_expression(self):
        circuit = parse_analog("x = 2 * (3 + 1)")
        decl = circuit.statements[0]
        expected = MathMul(
            expr1=MathNum(value=2),
            expr2=MathAdd(expr1=MathNum(value=3), expr2=MathNum(value=1)),
        )
        assert decl.value == expected

    @pytest.mark.parametrize(
        "func_name",
        ["sin", "cos", "tan", "exp", "log", "abs", "sinh", "cosh", "tanh",
         "atan", "acos", "asin", "atanh", "asinh", "acosh", "conj",
         "heaviside", "real", "imag"],
    )
    def test_unary_math_function(self, func_name):
        circuit = parse_analog(f"x = {func_name}(1)")
        decl = circuit.statements[0]
        assert isinstance(decl.value, MathFunc)
        assert decl.value.func == func_name
        assert decl.value.expr == MathNum(value=1)

    def test_atan2(self):
        circuit = parse_analog("x = atan2(1, 2)")
        decl = circuit.statements[0]
        assert isinstance(decl.value, MathFunc)
        assert decl.value.func == "atan2"
        assert decl.value.expr == [MathNum(value=1), MathNum(value=2)]
    

## Bool Expressions ##

class TestAnalogBool:
    def test_true(self):
        circuit = parse_analog("x = true")
        assert circuit.statements[0].value == Bool(value=True)

    def test_false(self):
        circuit = parse_analog("x = false")
        assert circuit.statements[0].value == Bool(value=False)
    
    @pytest.mark.parametrize(
        "op, cls",
        [("==", BoolEq), ("!=", BoolNotEq), ("<=", BoolLessThanEq), ("<", BoolLessThan), 
         ("and", BoolAnd), ("&&", BoolAnd), ("or", BoolOr), ("||", BoolOr),
         (">=", BoolGreaterThanEq), (">", BoolGreaterThan)],
    )
    def test_comparison(self, op, cls):
        circuit = parse_analog(f"x = 1 {op} 2")
        decl = circuit.statements[0]
        assert isinstance(decl, Declaration)
        assert isinstance(decl.value, cls)
    
    @pytest.mark.parametrize("op, cls", [("not", BoolNot), ("!", BoolNot),])
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
        circuit = parse_analog("op = %X %+ %Y")
        assert circuit.statements[0].value == OperatorAdd(op1=PauliX(), op2=PauliY())

    def test_operator_sub(self):
        circuit = parse_analog("op = %X %- %Y")
        assert circuit.statements[0].value == OperatorSub(op1=PauliX(), op2=PauliY())

    def test_operator_mul(self):
        circuit = parse_analog("op = %X %* %Y")
        assert circuit.statements[0].value == OperatorMul(op1=PauliX(), op2=PauliY())

    def test_operator_kron(self):
        circuit = parse_analog("op = %X %@ %Y")
        assert circuit.statements[0].value == OperatorKron(op1=PauliX(), op2=PauliY())


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
        assert statement.duration == MathNum(value=1.0)
        assert statement.targets == Access(name="r")
    
    def test_multiple_statements(self, register):
        program = "\n".join([
            register,
            "initialize(r)",
            "evolve(%X, 1.0, r)",
            "measure(r)",
        ])
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
        assert isinstance(ifelse.condition, BoolGreaterThan)
        assert len(ifelse.then_branch) == 1
        assert ifelse.else_branch == []
    
    def test_if_else(self):
        program = "x = 1\n if (x > 0) {\n y = 2\n} \n else {\n y = 3\n}"
        circuit = parse_analog(program)
        ifelse = circuit.statements[1]
        assert isinstance(ifelse, IfElse)
        assert isinstance(ifelse.condition, BoolGreaterThan)
        assert len(ifelse.then_branch) == 1
        assert len(ifelse.else_branch) == 1
    
    def test_while_statement(self):
        program = "n = 3\nwhile (n > 0) {\n    n = n - 1\n}"
        circuit = parse_analog(program)
        while_statement = circuit.statements[1]
        assert isinstance(while_statement, While)
        assert isinstance(while_statement.condition, BoolGreaterThan)
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
        assert isinstance(while_statement.condition, Bool)
        ifelse = while_statement.body[0]
        assert isinstance(ifelse, IfElse)
        assert isinstance(ifelse.condition, BoolEq)
        assert isinstance(ifelse.then_branch[0], Declaration)
        ifelse = while_statement.body[1]
        assert isinstance(ifelse, IfElse)
        assert isinstance(ifelse.condition, BoolEq)
        assert isinstance(ifelse.then_branch[0], Break)
        
    
    
## Serialization ##


class TestAnalogSerialize:
    @pytest.mark.parametrize(
        "program",
        ["r = qreg(2)",
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


## Control Flow Graph ##

class TestAnalogCFG:
    def test_analog_cfg(self):
        program = "r = qreg(3) \n x = 1"
        circuit = parse_analog(program)
        cfg = AnalogCFGBuilder().run(circuit)
        assert cfg is not None
    
    def test_analog_cfg_infinite_loop(self):
        program = "while(true) {y = 2}"
        circuit = parse_analog(program)
        cfg = AnalogCFGBuilder().run(circuit)
        with pytest.raises(TypeError):
            SCCAnalysis(cfg).infinite_loop_check()
        

## Type Checker ##

class TestAnalogTypeChecker:
    @pytest.mark.parametrize(
        "program",
        ["r = qreg(2) \n initialize(r)",
         "r = qreg(2) \n measure(r)",
         "r = qreg(2) \n evolve(%X, 1.0, r)",
         "s = 5 * 4",
         "s = 5 + 2",
         "s = qmode(3) \n initialize(s)",
         "H = %X %* %I",
         "cond = true and false",
         "cond = true and false \n if (cond) {t = 0.2}",
         "cond = true or false \n while (cond) {t = 0.2}",
         "r = qreg(3) \n target = [r[0], r[1], r[2]] \n initialize(target)",
         "if (5 <= 4) {s = true}"
        ],
    )
    def test_analog_type_checker(self, program):
        circuit = parse_analog(program)
        checker = AnalogTypeChecker()
        cfg = AnalogCFGBuilder().run(circuit)
        checker.analyze_dataflow(cfg)
        
    @pytest.mark.parametrize(
        "program",
        ["initialize(r)",
         "measure(r)",
         "evolve(%X, 1.0, r)",
         "s = 5 * true",
         "s = 5 + %I",
         "H = %X * %I",
         "cond = true and 4",
         "cond = 5 \n if (cond) {t = 0.2}",
         "cond = %I \n while (cond) {t = 0.2}",
         "s = 5 \n r = qreg(3) \n target = [r[0], r[1], r[2], s] \n initialize(target)"
        ],
    )
    def test_analog_type_checker_error(self, program):
        circuit = parse_analog(program)
        with pytest.raises(AnalogTypeError):
            checker = AnalogTypeChecker()
            cfg = AnalogCFGBuilder().run(circuit)
            checker.analyze_dataflow(cfg)
        

    