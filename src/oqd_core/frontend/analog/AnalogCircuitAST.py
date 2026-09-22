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

import antlr4

from oqd_core.frontend.analog.AnalogLexer import AnalogLexer
from oqd_core.frontend.analog.AnalogParser import AnalogParser
from oqd_core.frontend.analog.AnalogParserVisitor import AnalogParserVisitor
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

########################################################################################

__all__ = ["AnalogASTBuilder", "parse_analog"]


########################################################################################


class AnalogASTBuilder(AnalogParserVisitor):
    """
    Visitor that converts ANTLR parse tree to Analog interface AST
    """

    def __init__(self):
        self._loop_depth = 0

    def visitProgram(self, ctx: AnalogParser.ProgramContext):
        block = ctx.block()
        statements = self.visit(block) if block else []
        return AnalogCircuit(statements=statements)

    def visitBlock(self, ctx: AnalogParser.BlockContext):
        statements = [self.visit(stmt) for stmt in ctx.statement() if stmt]
        return statements

    def visitStatement(self, ctx: AnalogParser.StatementContext):
        return self.visit(ctx.getChild(0))

    ## Structural Control Flow

    def visitIfelse_stmt(self, ctx: AnalogParser.Ifelse_stmtContext):
        cond = self.visit(ctx.expr())

        then_branch = self.visit(ctx.block(0))
        else_branch = self.visit(ctx.block(1)) if len(ctx.block()) > 1 else []
        return IfElse(condition=cond, then_branch=then_branch, else_branch=else_branch)

    def visitWhile_stmt(self, ctx: AnalogParser.While_stmtContext):
        self._loop_depth += 1
        cond = self.visit(ctx.expr())
        body = self.visit(ctx.block())
        self._loop_depth -= 1
        return While(condition=cond, body=body)

    def visitBreak_stmt(self, ctx):
        if self._loop_depth == 0:
            raise SyntaxError("break outside of loop")
        return Break()

    def visitContinue_stmt(self, ctx):
        if self._loop_depth == 0:
            raise SyntaxError("continue outside of loop")
        return Continue()

    ## Atom ##

    def visitTerminal(self, ctx: AnalogParser.TerminalContext):
        return self.visitChildren(ctx)

    ## Variable ##

    def visitDeclaration(self, ctx: AnalogParser.DeclarationContext):
        name = ctx.ID().getText()
        value = self.visit(ctx.expr())

        return Declaration(name=name, value=value)

    def visitAccess(self, ctx: AnalogParser.DeclarationContext):
        name = ctx.ID().getText()
        return Access(name=name)

    ## List ##

    def visitAnalog_list(self, ctx: AnalogParser.Analog_listContext):
        values = [self.visit(e) for e in ctx.expr()]
        return AnalogList(values=values)

    def visitAnalog_list_extract(self, ctx: AnalogParser.Analog_list_extractContext):
        access = self.visit(ctx.access())
        index = self.visit(ctx.expr())

        return Extract(access=access, index=index)

    ## Quantum Op ##

    def visitOperator_terminal(self, ctx: AnalogParser.Operator_terminalContext):
        return self.visitChildren(ctx)

    def visitPauli_op(self, ctx: AnalogParser.Pauli_opContext):
        args = self.visit(ctx.args()) if ctx.args() else []

        if len(args) not in [0, 2, 3]:
            raise ValueError(f"Pauli operator takes 0 or 2 arguments, got {len(args)}")

        level1 = args[0] if args else 0
        level2 = args[1] if args else 1
        dim = args[2] if args and len(args) > 2 else 2

        if ctx.PAULI_X():
            return PauliX(level1=level1, level2=level2, dim=dim)
        if ctx.PAULI_Y():
            return PauliY(level1=level1, level2=level2, dim=dim)
        if ctx.PAULI_Z():
            return PauliZ(level1=level1, level2=level2, dim=dim)
        if ctx.PAULI_I():
            return PauliI(level1=level1, level2=level2, dim=dim)

    def visitLadder_op(self, ctx: AnalogParser.Ladder_opContext):
        if ctx.ANNIHILATION():
            return Annihilation()
        if ctx.CREATION():
            return Creation()
        if ctx.IDENTITY_OP():
            return Identity()

    ## Boolean ##

    def visitBool_literal(self, ctx: AnalogParser.Bool_literalContext):
        if ctx.TRUE():
            return Constant(value=True)
        if ctx.FALSE():
            return Constant(value=False)

    ## Function ##

    def visitMath_func(self, ctx: AnalogParser.Math_funcContext):
        return ctx.getChild(0).getText()

    def visitQuantum_func(self, ctx: AnalogParser.Quantum_funcContext):
        return ctx.getChild(0).getText()

    def visitList_func(self, ctx: AnalogParser.List_funcContext):
        return ctx.getChild(0).getText()

    def visitFunc_names(self, ctx: AnalogParser.Func_namesContext):
        return self.visitChildren(ctx)

    def visitArgs(self, ctx: AnalogParser.ArgsContext):
        args = [self.visit(args) for args in ctx.expr() if args is not None]
        return args

    def visitFunc(self, ctx: AnalogParser.FuncContext):
        func = self.visit(ctx.func_names())
        args = self.visit(ctx.args()) if ctx.args() else []

        match func:
            case "initialize":
                if len(args) != 1:
                    raise ValueError(f"Initialize takes 1 arguments, got {len(args)}")
                return Initialize(targets=args[0])
            case "evolve":
                if len(args) != 3:
                    raise ValueError(f"Evolve takes 3 arguments, got {len(args)}")
                return Evolve(hamiltonian=args[0], duration=args[1], targets=args[2])
            case "measure":
                if len(args) != 1:
                    raise ValueError(f"Measure takes 1 arguments, got {len(args)}")
                return Measure(targets=args[0])
            case "qreg":
                if len(args) not in [1, 2]:
                    raise ValueError(f"Measure takes 1 arguments, got {len(args)}")

                size = args[0]
                dim = args[1] if len(args) > 1 else 2
                return QuantumRegister(size=size, dim=dim)
            case "qmode":
                if len(args) != 1:
                    raise ValueError(f"Measure takes 1 arguments, got {len(args)}")
                return QuantumRegister(size=args[0])

        return BuiltinCall(func=func, args=args)

    ## Arithmetic ##

    def visitMath_terminal(self, ctx: AnalogParser.Math_terminalContext):
        if ctx.INT() is not None:
            return Constant(value=int(ctx.INT().getText()))
        if ctx.FLOAT() is not None:
            return Constant(value=float(ctx.FLOAT().getText()))
        if ctx.MATH_VAR() is not None:
            return RuntimeVar(name=ctx.MATH_VAR().getText())
        if ctx.access() is not None:
            return self.visit(ctx.access())
        if ctx.pexpr() is not None:
            return self.visit(ctx.pexpr())
        if ctx.complex_() is not None:
            return self.visit(ctx.complex_())

    def visitComplex(self, ctx: AnalogParser.ComplexContext):
        real = float(ctx.REAL_PART().getText()[:-1]) if ctx.REAL_PART() else 0
        imag = float(ctx.IMAG_PART().getText()[:-1]) if ctx.IMAG_PART() else 0

        return Complex(real=real, imag=imag)

    def _get_binop_args(self, expr):
        args = [self.visit(expr.getChild(0))]

        if expr.getChild(2):
            args.append(self.visit(expr.getChild(2)))

        return args

    def visitPexpr(self, ctx: AnalogParser.PexprContext):
        return self.visit(ctx.expr())

    def visitEexpr(self, ctx: AnalogParser.EexprContext):
        args = self._get_binop_args(ctx)

        return Pow(exprs=args) if len(args) > 1 else args[0]

    def visitUexpr(self, ctx: AnalogParser.UexprContext):
        expr = self.visit(ctx.eexpr())

        if ctx.PLUS():
            return Pos(expr=expr)
        if ctx.MINUS():
            return Neg(expr=expr)
        if ctx.NOT():
            return Not(expr=expr)

        return expr

    def visitMexpr(self, ctx: AnalogParser.MexprContext):
        args = self._get_binop_args(ctx)

        if len(args) == 1:
            return args[0]

        if ctx.MULT():
            return Mul(exprs=args)
        if ctx.DIV():
            return Div(exprs=args)
        if ctx.AT():
            return Kron(exprs=args)

    def visitAexpr(self, ctx: AnalogParser.AexprContext):
        args = self._get_binop_args(ctx)

        if len(args) == 1:
            return args[0]

        if ctx.PLUS():
            return Add(exprs=args)
        if ctx.MINUS():
            return Sub(exprs=args)

    def visitCexpr(self, ctx: AnalogParser.CexprContext):
        args = self._get_binop_args(ctx)

        if len(args) == 1:
            return args[0]

        if ctx.LT():
            return Lt(exprs=args)
        if ctx.LEQ():
            return Leq(exprs=args)
        if ctx.GT():
            return Gt(exprs=args)
        if ctx.GEQ():
            return Geq(exprs=args)

    def visitEqexpr(self, ctx: AnalogParser.EqexprContext):
        args = self._get_binop_args(ctx)

        if len(args) == 1:
            return args[0]

        if ctx.EQ():
            return Eq(exprs=args)
        if ctx.NEQ():
            return Neq(exprs=args)

    def visitAndexpr(self, ctx: AnalogParser.AndContext):
        args = self._get_binop_args(ctx)

        if len(args) == 1:
            return args[0]

        return And(exprs=args) if len(args) > 1 else args[0]

    def visitXorexpr(self, ctx: AnalogParser.XorContext):
        args = self._get_binop_args(ctx)

        if len(args) == 1:
            return args[0]

        return Xor(exprs=args) if len(args) > 1 else args[0]

    def visitOrexpr(self, ctx: AnalogParser.OrContext):
        args = self._get_binop_args(ctx)

        if len(args) == 1:
            return args[0]

        return Or(exprs=args) if len(args) > 1 else args[0]

    def visitExpr(self, ctx: AnalogParser.ExprContext):
        return self.visit(ctx.orexpr())

    ## Register and operator terminals ##

    def visitQuantum_register(self, ctx: AnalogParser.Quantum_registerContext):
        return QuantumRegister(size=int(ctx.INT().getText()))

    def visitMode_register(self, ctx: AnalogParser.Mode_registerContext):
        return ModeRegister(size=int(ctx.INT().getText()))


########################################################################################


def parse_analog(source):
    stream = antlr4.InputStream(source)
    lexer = AnalogLexer(stream)
    tokens = antlr4.CommonTokenStream(lexer)
    parse_result = AnalogParser(tokens)
    tree = parse_result.program()

    builder = AnalogASTBuilder()

    circuit = builder.visit(tree)

    return circuit
