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


import inspect

import ast_comments as ast
from oqd_compiler_infrastructure import ConversionRule, Post

from oqd_core.interface.analog import (
    Access,
    Add,
    AnalogCircuit,
    AnalogList,
    And,
    Annihilation,
    BuiltinCall,
    Complex,
    Constant,
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

########################################################################################

__all__ = ["PyASTtoAnalog", "analog"]

########################################################################################


class PyASTtoAnalog(ConversionRule):
    def map_Module(self, model, operands):
        return AnalogCircuit(statements=operands["body"][0])

    def map_FunctionDef(self, model, operands):
        return operands["body"]

    def map_Assign(self, model, operands):
        if len(model.targets) != 1:
            raise ValueError()
        if model.targets[0].id.startswith("__"):
            raise ValueError()
        return Declaration(name=model.targets[0].id, value=operands["value"])

    def map_Constant(self, model, operands):
        if isinstance(model.value, complex):
            return Constant(value=Complex(real=model.value.real, imag=model.value.imag))
        return Constant(value=model.value)

    def map_Return(self, model, operands):
        return operands["value"]

    def map_Call(self, model, operands):
        args = operands["args"]

        if not isinstance(model.func, ast.Name):
            raise ValueError

        name = model.func.id

        match name:
            case (
                "sin"
                | "cos"
                | "heaviside"
                | "abs"
                | "sin"
                | "cos"
                | "tan"
                | "exp"
                | "log"
                | "sinh"
                | "cosh"
                | "tanh"
                | "atan"
                | "acos"
                | "asin"
                | "atanh"
                | "asinh"
                | "acosh"
                | "heaviside"
                | "conj"
                | "real"
                | "imag"
                | "atan2"
                | "round"
                | "range"
                | "flatten"
                | "print"
                | "len"
            ):
                return BuiltinCall(func=name, args=args)

            case "evolve":
                if len(args) != 3:
                    raise ValueError()
                return Evolve(hamiltonian=args[0], duration=args[1], targets=args[2])

            case "initialize" | "measure":
                if len(args) != 1:
                    raise ValueError()
                return dict(initialize=Initialize, measure=Measure)[name](
                    targets=args[0]
                )

            case "mreg":
                if len(args) != 1:
                    raise ValueError()
                return ModeRegister(size=args[0])

            case "qreg":
                if len(args) not in [1, 2]:
                    raise ValueError()

                if len(args) == 1:
                    return QuantumRegister(size=args[0])

                return QuantumRegister(size=args[0], dim=args[1])

            case "X" | "Y" | "Z" | "I":
                if len(args) not in [0, 2, 3]:
                    raise ValueError()
                if len(args) == 0:
                    return dict(X=PauliX, Y=PauliY, Z=PauliZ, I=PauliI)[name]()
                if len(args) == 2:
                    return dict(X=PauliX, Y=PauliY, Z=PauliZ, I=PauliI)[name](
                        level1=args[0], level2=args[1]
                    )
                return dict(X=PauliX, Y=PauliY, Z=PauliZ, I=PauliI)[name](
                    level1=args[0], level2=args[1], dim=args[2]
                )
            case "A" | "C" | "J":
                return dict(A=Annihilation, C=Creation, J=Identity)()

            case _:
                raise ValueError("Unsupported builtin function call")

    def map_List(self, model, operands):
        return AnalogList(values=operands["elts"])

    def map_Subscript(self, model, operands):
        return Extract(access=operands["value"], index=operands["slice"])

    def map_UnaryOp(self, model, operands):
        operand = operands["operand"]
        match model.op:
            case ast.USub():
                return Neg(expr=operand)
            case ast.UAdd():
                return Pos(expr=operand)
            case ast.Not():
                return Not(expr=operand)

    def map_BinOp(self, model, operands):
        left, right = operands["left"], operands["right"]
        match model.op:
            case ast.Add():
                return Add(exprs=[left, right])
            case ast.Sub():
                return Sub(exprs=[left, right])
            case ast.Mult():
                return Mul(exprs=[left, right])
            case ast.Div():
                return Div(exprs=[left, right])
            case ast.Pow():
                return Pow(exprs=[left, right])
            case ast.MatMult():
                return Kron(exprs=[left, right])
            case ast.BitAnd():
                return And(exprs=[left, right])
            case ast.BitOr():
                return Or(exprs=[left, right])
            case ast.BitXor():
                return Xor(exprs=[left, right])

    def map_BoolOp(self, model, operands):
        args = operands["values"]
        match model.op:
            case ast.And():
                return And(exprs=args)
            case ast.Or():
                return Or(exprs=args)
            case ast.Xor():
                return Xor(exprs=args)

    def _compare_helper(self, op, left, right):
        match op:
            case ast.Eq():
                return Eq(exprs=[left, right])
            case ast.NotEq():
                return Neq(exprs=[left, right])
            case ast.Lt():
                return Lt(exprs=[left, right])
            case ast.LtE():
                return Leq(exprs=[left, right])
            case ast.Gt():
                return Gt(exprs=[left, right])
            case ast.GtE():
                return Geq(exprs=[left, right])

    def map_Compare(self, model, operands):
        left = operands["left"]
        args = operands["comparators"]
        ops = model.ops

        current = left

        while ops:
            current = self._compare_helper(ops.pop(0), current, args.pop(0))

        return current

    def map_While(self, model, operands):
        return While(condition=operands["test"], body=operands["body"])

    def map_Name(self, model, operands):
        if isinstance(model.ctx, ast.Load):
            return (
                RuntimeVar(name="#" + model.id.removeprefix("__"))
                if model.id.startswith("__")
                else Access(name=model.id)
            )

    def map_If(self, model, operands):
        if model.orelse:
            return IfElse(
                condition=operands["test"],
                then_branch=operands["body"],
                else_branch=operands["orelse"],
            )

        return IfElse(condition=operands["test"], then_branch=operands["body"])

    def generic_map(self, model, operands):
        return ()

    def map_Expr(self, model, operands):
        return operands["value"]


def analog(func):

    source = inspect.getsource(func)
    pyast = ast.parse(source)

    circuit = Post(PyASTtoAnalog())(pyast)

    return circuit
