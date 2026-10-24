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

from oqd_core.frontend.atomic._generated import (
    AtomicLexer,
    AtomicParser,
    AtomicParserVisitor,
)
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

########################################################################################

__all__ = ["AtomicASTBuilder", "parse_atomic"]

########################################################################################


class AtomicASTBuilder(AtomicParserVisitor):
    """
    Visitor that converts ANTLR parse tree to Atomic interface AST
    """

    def __init__(self):
        self._loop_depth = 0

    def visitProgram(self, ctx: AtomicParser.ProgramContext):
        block = ctx.block()
        statements = self.visit(block) if block else []
        return AtomicCircuit(statements=statements)

    def visitBlock(self, ctx: AtomicParser.BlockContext):
        statements = [self.visit(stmt) for stmt in ctx.statement() if stmt]
        return statements

    def visitStatement(self, ctx: AtomicParser.StatementContext):
        return self.visit(ctx.getChild(0))
    
    ## Structural Control Flow ##

    def visitIfelse_stmt(self, ctx: AtomicParser.Ifelse_stmtContext):
        cond = self.visit(ctx.expr())
        then_branch = self.visit(ctx.block(0))
        else_branch = self.visit(ctx.block(1)) if len(ctx.block()) > 1 else []
        return IfElse(condition=cond, then_branch=then_branch, else_branch=else_branch)
    
    def visitWhile_stmt(self, ctx: AtomicParser.While_stmtContext):
        self._loop_depth += 1
        cond = self.visit(ctx.expr())
        body = self.visit(ctx.block())
        self._loop_depth -= 1
        return While(condition=cond, body=body)

    def visitBreak_stmt(self, ctx: AtomicParser.Break_stmtContext):
        if self._loop_depth == 0:
            raise SyntaxError("break outside of loop")
        return Break()

    def visitContinue_stmt(self, ctx: AtomicParser.Continue_stmtContext):
        if self._loop_depth == 0:
            raise SyntaxError("continue outside of loop")
        return Continue()
    
    def visitParallel_stmt(self, ctx: AtomicParser.Parallel_stmtContext):
        body = self.visit(ctx.block())
        return ParallelProtocol(pulses=body)

    def visitSerial_stmt(self, ctx: AtomicParser.Serial_stmtContext):
        body = self.visit(ctx.block())
        return SerialProtocol(pulses=body)

    ## Atom ##

    def visitTerminal(self, ctx: AtomicParser.TerminalContext):
        return self.visitChildren(ctx)

    ## Variable ##

    def visitDeclaration(self, ctx: AtomicParser.DeclarationContext):
        name = ctx.ID().getText()
        value = self.visit(ctx.expr())

        return Declaration(name=name, value=value)
    
    def visitAccess(self, ctx: AtomicParser.AccessContext):
        return Access(name=ctx.ID().getText())
    
    ## List ##

    def visitAtomic_list(self, ctx: AtomicParser.Atomic_listContext):
        values = [self.visit(e) for e in ctx.expr()]
        return AtomicList(values=values)
    
    def visitIexpr(self, ctx: AtomicParser.IexprContext):
        if ctx.terminal():
            return self.visit(ctx.terminal())

        expr = self.visit(ctx.iexpr())
        index = self.visit(ctx.expr())

        return Extract(value=expr, index=index)
    
    ## Boolean ##

    def visitBool_literal(self, ctx: AtomicParser.Bool_literalContext):
        if ctx.TRUE():
            return Constant(value=True)
        if ctx.FALSE():
            return Constant(value=False)
    

    ## Function ##

    def visitMath_func(self, ctx: AtomicParser.Math_funcContext):
        return ctx.getChild(0).getText()

    def visitQuantum_func(self, ctx: AtomicParser.Quantum_funcContext):
        return ctx.getChild(0).getText()

    def visitList_func(self, ctx: AtomicParser.List_funcContext):
        return ctx.getChild(0).getText()

    def visitMisc_func(self, ctx: AtomicParser.Misc_funcContext):
        return ctx.getChild(0).getText()

    def visitArith_func(self, ctx: AtomicParser.Arith_funcContext):
        return ctx.getChild(0).getText()

    def visitFunc_names(self, ctx: AtomicParser.Func_namesContext):
        return self.visitChildren(ctx)

    def visitArgs(self, ctx: AtomicParser.ArgsContext):
        args = [self.visit(args) for args in ctx.expr()]
        return args

    def visitFunc(self, ctx: AtomicParser.FuncContext):
        func = self.visit(ctx.func_names())
        args = self.visit(ctx.args()) if ctx.args() else []

        match func:
            case "ionreg":
                if len(args) != 1:
                    raise ValueError(f"ionreg takes 1 arguments, got {len(args)}")
                return IonRegister(size=args[0])
            case "beam":
                if len(args) != 5:
                    raise ValueError(f"beam takes 5 arguments, got {len(args)}")
                (frequency, rabi, phase, polarization, wavevector) = args
                return Beam(
                    frequency=frequency,
                    rabi=rabi,
                    phase=phase,
                    polarization=polarization,
                    wavevector=wavevector,
                )
            case "pulse":
                num_args = len(args)
                match num_args:
                    case 3:
                        (beam, duration, target) = args
                        measured = False
                    case 4:
                        (beam, duration, target, measured) = args
                    case _:
                        raise ValueError(
                            f"pulse takes 3 or 4 arguments, got {num_args}"
                        )

                return Pulse(
                    beam=beam,
                    duration=duration,
                    target=target,
                    measured=measured,
                )

        return BuiltinCall(func=func, args=args)

    ## Arithmetic ##

    def visitMath_terminal(self, ctx: AtomicParser.Math_terminalContext):
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

    def visitComplex(self, ctx: AtomicParser.ComplexContext):
        real = float(ctx.REAL_PART().getText()[:-1]) if ctx.REAL_PART() else 0
        imag = float(ctx.IMAG_PART().getText()[:-1]) if ctx.IMAG_PART() else 0

        return Complex(real=real, imag=imag)

    def _get_binop_args(self, expr):
        args = [self.visit(expr.getChild(0))]

        if expr.getChild(2):
            args.append(self.visit(expr.getChild(2)))

        return args

    def visitPexpr(self, ctx: AtomicParser.PexprContext):
        return self.visit(ctx.expr())

    def visitEexpr(self, ctx: AtomicParser.EexprContext):
        args = self._get_binop_args(ctx)

        return Pow(exprs=args) if len(args) > 1 else args[0]

    def visitUexpr(self, ctx: AtomicParser.UexprContext):
        if ctx.eexpr():
            return self.visit(ctx.eexpr())

        expr = self.visit(ctx.uexpr())
        if ctx.PLUS():
            return Pos(expr=expr)
        if ctx.MINUS():
            return Neg(expr=expr)
        if ctx.NOT():
            return Not(expr=expr)

        return expr

    def visitMexpr(self, ctx: AtomicParser.MexprContext):
        if ctx.mexpr() is None:
            return self.visit(ctx.uexpr()[0])

        args = [self.visit(ctx.mexpr())] + [self.visit(expr) for expr in ctx.uexpr()]

        if ctx.MULT():
            return Mul(exprs=args)
        if ctx.DIV():
            return Div(exprs=args)

    def visitAexpr(self, ctx: AtomicParser.AexprContext):
        if ctx.aexpr() is None:
            return self.visit(ctx.mexpr()[0])

        args = [self.visit(ctx.aexpr())] + [self.visit(expr) for expr in ctx.mexpr()]

        if ctx.PLUS():
            return Add(exprs=args)
        if ctx.MINUS():
            return Sub(exprs=args)

    def visitCexpr(self, ctx: AtomicParser.CexprContext):
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

    def visitEqexpr(self, ctx: AtomicParser.EqexprContext):
        args = self._get_binop_args(ctx)

        if len(args) == 1:
            return args[0]

        if ctx.EQ():
            return Eq(exprs=args)
        if ctx.NEQ():
            return Neq(exprs=args)

    def visitAndexpr(self, ctx: AtomicParser.AndContext):
        args = [self.visit(expr) for expr in ctx.eqexpr()]

        if len(args) == 1:
            return args[0]

        return And(exprs=args) if len(args) > 1 else args[0]

    def visitXorexpr(self, ctx: AtomicParser.XorContext):
        args = [self.visit(expr) for expr in ctx.andexpr()]

        if len(args) == 1:
            return args[0]

        return Xor(exprs=args) if len(args) > 1 else args[0]

    def visitOrexpr(self, ctx: AtomicParser.OrContext):
        args = [self.visit(expr) for expr in ctx.xorexpr()]

        if len(args) == 1:
            return args[0]

        return Or(exprs=args) if len(args) > 1 else args[0]

    def visitExpr(self, ctx: AtomicParser.ExprContext):
        return self.visit(ctx.orexpr())


########################################################################################


def parse_atomic(source):
    stream = antlr4.InputStream(source)
    lexer = AtomicLexer(stream)
    tokens = antlr4.CommonTokenStream(lexer)
    parser = AtomicParser(tokens)
    tree = parser.program()
    
    builder = AtomicASTBuilder()

    circuit = builder.visit(tree)

    return circuit
