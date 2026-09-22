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

# _BOOL_OP_MAP = {
#     AnalogLexer.AND: BoolAnd,
#     AnalogLexer.AND2: BoolAnd,
#     AnalogLexer.OR: BoolOr,
#     AnalogLexer.OR2: BoolOr,
#     AnalogLexer.EQ: BoolEq,
#     AnalogLexer.NEQ: BoolNotEq,
#     AnalogLexer.LT: BoolLessThan,
#     AnalogLexer.LTE: BoolLessThanEq,
#     AnalogLexer.GT: BoolGreaterThan,
#     AnalogLexer.GTE: BoolGreaterThanEq,
# }

# _OP_TERMINAL_MAP = {
#     "I": PauliI,
#     "X": PauliX,
#     "Y": PauliY,
#     "Z": PauliZ,
#     "C": Creation,
#     "A": Annihilation,
#     "J": Identity,
# }

# _FUNC_TOKEN_TO_NAME = {
#     AnalogLexer.ABS: "abs",
#     AnalogLexer.SIN: "sin",
#     AnalogLexer.COS: "cos",
#     AnalogLexer.TAN: "tan",
#     AnalogLexer.EXP: "exp",
#     AnalogLexer.LOG: "log",
#     AnalogLexer.SINH: "sinh",
#     AnalogLexer.COSH: "cosh",
#     AnalogLexer.TANH: "tanh",
#     AnalogLexer.ATAN: "atan",
#     AnalogLexer.ACOS: "acos",
#     AnalogLexer.ASIN: "asin",
#     AnalogLexer.ATANH: "atanh",
#     AnalogLexer.ASINH: "asinh",
#     AnalogLexer.ACOSH: "acosh",
#     AnalogLexer.HEAVISIDE: "heaviside",
#     AnalogLexer.CONJ: "conj",
#     AnalogLexer.REAL: "real",
#     AnalogLexer.IMAG_FN: "imag",
#     AnalogLexer.ATAN2: "atan2",
# }


# def _get_token_type(node) -> int:
#     """Extract token type from a terminal or context"""
#     if isinstance(node, TerminalNodeImpl):
#         return node.symbol.type
#     payload = getattr(node, "getPayload", lambda: None)()
#     return getattr(payload, "type", -1) if payload else -1


# def _get_text(node) -> str:
#     """Get text from a parse tree node"""
#     return node.getText() if hasattr(node, "getText") else str(node)


# def _comparator_to_bool_class(cmp_ctx: AnalogParser.ComparatorsContext):
#     op_ctx = (
#         cmp_ctx.bool_eq_op()
#         or cmp_ctx.bool_not_eq_op()
#         or cmp_ctx.bool_lt_op()
#         or cmp_ctx.bool_lte_op()
#         or cmp_ctx.bool_gt_op()
#         or cmp_ctx.bool_gte_op()
#     )
#     if op_ctx is None:
#         raise ValueError("Empty comparators")
#     tt = _get_token_type(op_ctx.getChild(0))
#     cls = _BOOL_OP_MAP.get(tt)
#     if cls is None:
#         raise ValueError(f"Unknown comparator token type: {tt}")
#     return cls


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
        statements = [self.visit(stmt) for stmt in ctx.statement() if stmt is not None]
        return statements

    def visitStatement(self, ctx: AnalogParser.StatementContext):
        return self.visitChildren(ctx)

    ## Structural Control Flow

    def visitIfelse_stmt(self, ctx: AnalogParser.Ifelse_stmtContext):
        cond = self.visit(ctx.expr())
        then_branch = self.visit(ctx.block(0))
        else_branch = self.visit(ctx.block(1))
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
        return Extract(acces=access, index=index)

    ## Quantum Op ##

    def visitOperator_terminal(self, ctx: AnalogParser.Operator_terminalContext):
        return self.visitChildren(ctx)

    def visitPauli_op(self, ctx: AnalogParser.Pauli_opContext):
        args = self.visit(ctx.args())

        if len(args) not in [0, 2]:
            raise ValueError(f"Pauli operator takes 0 or 2 arguments, got {len(args)}")

        level1 = self.visit(args[0])
        level2 = self.visit(args[1])

        if ctx.PAULI_X():
            return (
                PauliX(level1=level1, level2=level2) if level1 and level2 else PauliX()
            )
        if ctx.PAULI_Y():
            return (
                PauliY(level1=level1, level2=level2) if level1 and level2 else PauliY()
            )
        if ctx.PAULI_Z():
            return (
                PauliZ(level1=level1, level2=level2) if level1 and level2 else PauliZ()
            )
        if ctx.PAULI_I():
            return (
                PauliI(level1=level1, level2=level2) if level1 and level2 else PauliI()
            )

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
        args = self.visit(ctx.args())

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
            return Access(name=ctx.ID().getText())
        if ctx.pexpr() is not None:
            return self.visit(ctx.pexpr())
        if ctx.complex_() is not None:
            return self.visit(ctx.complex_())

    def visitComplex(self, ctx: AnalogParser.ComplexContext):
        real = self.visit(ctx.real_part()) if ctx.real_part() else 0
        imag = self.visit(ctx.imag_part()) if ctx.imag_part() else 0

        return Complex(real=real, imag=imag)

    def visitReal_part(self, ctx: AnalogParser.Real_partContext):
        return float(ctx.getChild(0).getText())

    def visitImag_part(self, ctx: AnalogParser.Imag_partContext):
        return float(ctx.getChild(0).getText())

    def visitPexpr(self, ctx: AnalogParser.PexprContext):
        return self.visit(ctx.expr())

    def visitEexpr(self, ctx: AnalogParser.EexprContext):
        left = self.visit(ctx.eexpr()) if ctx.eexpr() else None
        right = self.visit(ctx.terminal())

        return Pow(expr1=left, expr2=right) if left else right

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
        left = self.visit(ctx.mexpr()) if ctx.mexpr() else None
        right = self.visit(ctx.uexpr())

        if not left:
            return right

        if ctx.MULT():
            return Mul(expr1=left, expr2=right)
        if ctx.DIV():
            return Div(expr1=left, expr2=right)
        if ctx.AT():
            return Kron(expr1=left, expr2=right)

    def visitAexpr(self, ctx: AnalogParser.AexprContext):
        left = self.visit(ctx.aexpr()) if ctx.aexpr() else None
        right = self.visit(ctx.mexpr())

        if not left:
            return right

        if ctx.PLUS():
            return Add(expr1=left, expr2=right)
        if ctx.MINUS():
            return Sub(expr1=left, expr2=right)

    def visitCexpr(self, ctx: AnalogParser.CexprContext):
        left = self.visit(ctx.cexpr()) if ctx.cexpr() else None
        right = self.visit(ctx.aexpr())

        if not left:
            return right

        if ctx.LT():
            return Lt(expr1=left, expr2=right)
        if ctx.LEQ():
            return Leq(expr1=left, expr2=right)
        if ctx.GT():
            return Gt(expr1=left, expr2=right)
        if ctx.GEQ():
            return Geq(expr1=left, expr2=right)

    def visitEqexpr(self, ctx: AnalogParser.EqexprContext):
        left = self.visit(ctx.eqexpr()) if ctx.eqexpr() else None
        right = self.visit(ctx.cexpr())

        if not left:
            return right

        if ctx.EQ():
            return Eq(expr1=left, expr2=right)
        if ctx.NEQ():
            return Neq(expr1=left, expr2=right)

    def visitAndexpr(self, ctx: AnalogParser.AndContext):
        left = self.visit(ctx.andexpr()) if ctx.andexpr() else None
        right = self.visit(ctx.eqexpr())

        return And(expr1=left, expr2=right) if left else right

    def visitXorexpr(self, ctx: AnalogParser.XorContext):
        left = self.visit(ctx.xorexpr()) if ctx.xorexpr() else None
        right = self.visit(ctx.andexpr())

        return Xor(expr1=left, expr2=right) if left else right

    def visitOrexpr(self, ctx: AnalogParser.OrContext):
        left = self.visit(ctx.orexpr()) if ctx.orexpr() else None
        right = self.visit(ctx.xorexpr())

        return Or(expr1=left, expr2=right) if left else right

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
