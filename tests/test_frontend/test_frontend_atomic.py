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

from oqd_core.frontend.atomic.AtomicCircuitAST import parse_atomic
from oqd_core.frontend.atomic.serialize import serialize_atomic
from oqd_core.interface.atomic import (
    Access,
    AtomicCircuit,
    AtomicList,
    Beam,
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
    Extract,
    IfElse,
    IonRegister,
    MathAdd,
    MathDiv,
    MathFunc,
    MathImag,
    MathMul,
    MathNum,
    MathPow,
    MathSub,
    MathVar,
    ParallelProtocol,
    Pulse,
    While,
)

########################################################################################

def test_program():
    circuit = parse_atomic("")
    assert isinstance(circuit, AtomicCircuit)
    assert circuit.statements == []


## Declarations ##

class TestAtomicDeclarations:
    
    def test_int(self):
        circuit = parse_atomic("x = 1")
        assert len(circuit.statements) == 1
        decl = circuit.statements[0]
        assert isinstance(decl, Declaration)
        assert decl.name == "x"
        assert decl.value == MathNum(value=1)
    
    def test_float(self):
        circuit = parse_atomic("pi = 3.14")
        decl = circuit.statements[0]
        assert isinstance(decl, Declaration)
        assert decl.name == "pi"
        assert decl.value == MathNum(value=3.14)
    
    def test_math_var(self):
        circuit = parse_atomic("omega = #omega")
        decl = circuit.statements[0]
        assert decl.value == MathVar(name="#omega")
    
    def test_imag(self):
        circuit = parse_atomic("z = 1j")
        decl = circuit.statements[0]
        assert decl.value == MathImag()
    
    def test_ion_register(self):
        circuit = parse_atomic("r = ionreg(2)")
        decl = circuit.statements[0]
        assert decl.value == IonRegister(size=2)
    
    def test_list(self):
        circuit = parse_atomic("list = [1, 2, 3]")
        decl = circuit.statements[0]
        assert isinstance(decl.value, AtomicList)
        assert decl.value.values == [MathNum(value=1), MathNum(value=2), MathNum(value=3)]

    def test_list_extract(self):
        circuit = parse_atomic("r = ionreg(2)\n ion = r[0]")
        decl = circuit.statements[1]
        assert isinstance(decl.value, Extract)
        assert decl.value.access == Access(name="r")
        assert decl.value.index == 0
    

## Math Expressions ##

class TestAtomicMathExpressions:
    
    def test_addition(self):
        circuit = parse_atomic("x = 1 + 2")
        decl = circuit.statements[0]
        assert decl.value == MathAdd(expr1=MathNum(value=1), expr2=MathNum(value=2))

    def test_subtraction(self):
        circuit = parse_atomic("x = 5 - 3")
        decl = circuit.statements[0]
        assert decl.value == MathSub(expr1=MathNum(value=5), expr2=MathNum(value=3))

    def test_multiplication(self):
        circuit = parse_atomic("x = 2 * 3")
        decl = circuit.statements[0]
        assert decl.value == MathMul(expr1=MathNum(value=2), expr2=MathNum(value=3))

    def test_division(self):
        circuit = parse_atomic("x = 6 / 2")
        decl = circuit.statements[0]
        assert decl.value == MathDiv(expr1=MathNum(value=6), expr2=MathNum(value=2))

    def test_power(self):
        circuit = parse_atomic("x = 2^3")
        decl = circuit.statements[0]
        assert decl.value == MathPow(expr1=MathNum(value=2), expr2=MathNum(value=3))

    def test_negation(self):
        circuit = parse_atomic("x = -1")
        decl = circuit.statements[0]
        assert decl.value == MathMul(expr1=MathNum(value=-1), expr2=MathNum(value=1))
    
    def test_nested_expression(self):
        circuit = parse_atomic("x = 2 * 3 + 1")
        decl = circuit.statements[0]
        expected = MathAdd(
            expr1=MathMul(expr1=MathNum(value=2), expr2=MathNum(value=3)),
            expr2=MathNum(value=1),
        )
        assert decl.value == expected
    
    def test_paranthesis_expression(self):
        circuit = parse_atomic("x = 2 * (3 + 1)")
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
        circuit = parse_atomic(f"x = {func_name}(1)")
        decl = circuit.statements[0]
        assert isinstance(decl.value, MathFunc)
        assert decl.value.func == func_name
        assert decl.value.expr == MathNum(value=1)

    def test_atan2(self):
        circuit = parse_atomic("x = atan2(1, 2)")
        decl = circuit.statements[0]
        assert isinstance(decl.value, MathFunc)
        assert decl.value.func == "atan2"
        assert decl.value.expr == [MathNum(value=1), MathNum(value=2)]
    

## Bool Expressions ##

class TestAtomicBool:
    def test_true(self):
        circuit = parse_atomic("x = true")
        assert circuit.statements[0].value == Bool(value=True)

    def test_false(self):
        circuit = parse_atomic("x = false")
        assert circuit.statements[0].value == Bool(value=False)
    
    @pytest.mark.parametrize(
        "op, cls",
        [("==", BoolEq), ("!=", BoolNotEq), ("<=", BoolLessThanEq), ("<", BoolLessThan), 
         ("and", BoolAnd), ("&&", BoolAnd), ("or", BoolOr), ("||", BoolOr),
         (">=", BoolGreaterThanEq), (">", BoolGreaterThan)],
    )
    def test_comparison(self, op, cls):
        circuit = parse_atomic(f"x = 1 {op} 2")
        decl = circuit.statements[0]
        assert isinstance(decl, Declaration)
        assert isinstance(decl.value, cls)
    
    @pytest.mark.parametrize("op, cls", [("not", BoolNot), ("!", BoolNot),])
    def test_not(self, op, cls):
        circuit = parse_atomic(f"a = true \n x = {op} a")
        decl = circuit.statements[1]
        assert isinstance(decl, Declaration)
        assert isinstance(decl.value, cls)


## Statements ##

class TestAtomicStatements:
    @pytest.fixture
    def register(self):
        return "r = ionreg(2)\n"
    
    @pytest.fixture
    def beam(self):
        return "beam_mw = beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])\n"
    
    def test_beam(self, register, beam):
        circuit = parse_atomic(register + "beam(2e6, 0.25, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])\n")
        statement = circuit.statements[1]
        assert isinstance(statement, Beam)
        assert statement.frequency == MathNum(value=2e6)
        assert statement.rabi == MathNum(value=0.25)
        assert statement.phase == MathNum(value=0.0)
        assert statement.polarization == AtomicList(values=[0.0, 1.0, 0.0])
        assert statement.wavevector == AtomicList(values=[0.0, 0.0, 1.0])
        
        
    def test_pulse(self, register, beam):
        circuit = parse_atomic(register + beam + "pulse(beam_mw, 1e-5, r, true)")
        statement = circuit.statements[2]
        assert isinstance(statement, Pulse)
        assert statement.beam == Access(name="beam_mw")
        assert statement.duration == MathNum(value=1e-5)
        assert statement.target == Access(name="r")
        assert statement.measured == Bool(value=True)
        
    def test_pulse_not_measured(self, register, beam):
        circuit = parse_atomic(register + beam + "pulse(beam_mw, 1e-5, r)")
        statement = circuit.statements[2]
        assert isinstance(statement, Pulse)
        assert statement.beam == Access(name="beam_mw")
        assert statement.duration == MathNum(value=1e-5)
        assert statement.target == Access(name="r")
        assert statement.measured == Bool(value=False)
    
    def test_parallel(self, register, beam):
        circuit = parse_atomic(register + beam + "parallel {\n pulse(beam_mw, 5e-6, r[0])\n pulse(beam_mw, 5e-6, r[1])}")
        statement = circuit.statements[2]
        assert isinstance(statement, ParallelProtocol)
        pulse1 = statement.pulses[0]
        assert isinstance(pulse1, Pulse)
        pulse2 = statement.pulses[1]
        assert isinstance(pulse2, Pulse)
    


## Control Flow Statements ##

class TestAtomicControlFlow:
    
    def test_if(self):
        program = "x = 1\n if (x > 0) {\n y = 2\n}"
        circuit = parse_atomic(program)
        ifelse = circuit.statements[1]
        assert isinstance(ifelse, IfElse)
        assert isinstance(ifelse.condition, BoolGreaterThan)
        assert len(ifelse.then_branch) == 1
        assert ifelse.else_branch == []
    
    def test_if_else(self):
        program = "x = 1\n if (x > 0) {\n y = 2\n} \n else {\n y = 3\n}"
        circuit = parse_atomic(program)
        ifelse = circuit.statements[1]
        assert isinstance(ifelse, IfElse)
        assert isinstance(ifelse.condition, BoolGreaterThan)
        assert len(ifelse.then_branch) == 1
        assert len(ifelse.else_branch) == 1
    
    def test_while_statement(self):
        program = "n = 3\nwhile (n > 0) {\n    n = n - 1\n}"
        circuit = parse_atomic(program)
        while_statement = circuit.statements[1]
        assert isinstance(while_statement, While)
        assert isinstance(while_statement.condition, BoolGreaterThan)
        assert len(while_statement.body) == 1

    def test_break_loop(self):
        program = "while (true) {\n break\n}"
        circuit = parse_atomic(program)
        while_statement = circuit.statements[0]
        assert isinstance(while_statement.body[0], Break)

    def test_continue_loop(self):
        program = "while (true) {\n continue\n}"
        circuit = parse_atomic(program)
        while_statement = circuit.statements[0]
        assert isinstance(while_statement.body[0], Continue)
    
    def test_break_outside_loop(self):
        with pytest.raises(SyntaxError, match="break outside of loop"):
            parse_atomic("break")
    
    def test_continue_outside_loop(self):
        with pytest.raises(SyntaxError, match="continue outside of loop"):
            parse_atomic("continue")
    
    def test_nested_control_flow(self):
        program = "while(true) {\n if (a == b) {x = 0} \n if (x == 0) { break}\n}"
        circuit = parse_atomic(program)
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

def test_atomic_serialize():
    program = "while(true) {\n if (a == b) {x = 0} \n if (x == 0) { break}\n}"
    circuit = parse_atomic(program)
    assert isinstance(circuit, AtomicCircuit)
    serialized = serialize_atomic(circuit)
    assert isinstance(serialized, str)
    

