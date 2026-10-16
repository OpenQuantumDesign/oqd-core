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
from antlr4.tree.Tree import TerminalNodeImpl

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
    SerialProtocol,
    While,
)

from .AtomicLexer import AtomicLexer
from .AtomicParser import AtomicParser
from .AtomicParserVisitor import AtomicParserVisitor

########################################################################################

__all__ = ["AtomicASTBuilder"]

########################################################################################

_BOOL_OP_MAP = {
    AtomicLexer.AND: BoolAnd,
    AtomicLexer.AND2: BoolAnd,
    AtomicLexer.OR: BoolOr,
    AtomicLexer.OR2: BoolOr,
    AtomicLexer.EQ: BoolEq,
    AtomicLexer.NEQ: BoolNotEq,
    AtomicLexer.LT: BoolLessThan,
    AtomicLexer.LTE: BoolLessThanEq,
    AtomicLexer.GT: BoolGreaterThan,
    AtomicLexer.GTE: BoolGreaterThanEq,
}

_FUNC_TOKEN_TO_NAME = {
    AtomicLexer.ABS: "abs",
    AtomicLexer.SIN: "sin",
    AtomicLexer.COS: "cos",
    AtomicLexer.TAN: "tan",
    AtomicLexer.EXP: "exp",
    AtomicLexer.LOG: "log",
    AtomicLexer.SINH: "sinh",
    AtomicLexer.COSH: "cosh",
    AtomicLexer.TANH: "tanh",
    AtomicLexer.ATAN: "atan",
    AtomicLexer.ACOS: "acos",
    AtomicLexer.ASIN: "asin",
    AtomicLexer.ATANH: "atanh",
    AtomicLexer.ASINH: "asinh",
    AtomicLexer.ACOSH: "acosh",
    AtomicLexer.HEAVISIDE: "heaviside",
    AtomicLexer.CONJ: "conj",
    AtomicLexer.REAL: "real",
    AtomicLexer.IMAG_FN: "imag",
    AtomicLexer.ATAN2: "atan2",
}


def _get_token_type(node) -> int:
    """Extract token type from a terminal or context"""
    if isinstance(node, TerminalNodeImpl):
        return node.symbol.type
    payload = getattr(node, "getPayload", lambda: None)()
    return getattr(payload, "type", -1) if payload else -1


def _get_text(node) -> str:
    """Get text from a parse tree node"""
    return node.getText() if hasattr(node, "getText") else str(node)


def _comparator_to_bool_class(cmp_ctx: AtomicParser.ComparatorsContext):
    op_ctx = (
        cmp_ctx.bool_eq_op()
        or cmp_ctx.bool_not_eq_op()
        or cmp_ctx.bool_lt_op()
        or cmp_ctx.bool_lte_op()
        or cmp_ctx.bool_gt_op()
        or cmp_ctx.bool_gte_op()
    )
    if op_ctx is None:
        raise ValueError("Empty comparators")
    tt = _get_token_type(op_ctx.getChild(0))
    cls = _BOOL_OP_MAP.get(tt)
    if cls is None:
        raise ValueError(f"Unknown comparator token type: {tt}")
    return cls


########################################################################################


class AtomicASTBuilder(AtomicParserVisitor):
    def __init__(self):
        self._loop_depth = 0

    def visitProgram(self, ctx: AtomicParser.ProgramContext):
        block = ctx.block()
        statements = []
        if block:
            statements = self.visit(block)

        return AtomicCircuit(statements=statements)

    def visitBlock(self, ctx: AtomicParser.BlockContext):
        statements = []
        for stmt_ctx in ctx.statement():
            stmt = self.visit(stmt_ctx)
            if stmt is not None:
                statements.append(stmt)
        return statements

    def visitStatement(self, ctx: AtomicParser.StatementContext):
        child = ctx.getChild(0)
        return self.visit(child)

    def visitDeclaration(self, ctx: AtomicParser.DeclarationContext):
        name = ctx.ID().getText()
        val = self.visit(ctx.expr())
        decl = Declaration(name=name, value=val)
        return decl

    ## Statements ##

    def visitTargets(self, ctx: AtomicParser.TargetsContext):
        return self.visit(ctx.expr())

    def visitParallel_stmt(self, ctx: AtomicParser.Parallel_stmtContext):
        body = self.visit(ctx.block())
        return ParallelProtocol(pulses=body)

    def visitSerial_stmt(self, ctx: AtomicParser.Serial_stmtContext):
        body = self.visit(ctx.block())
        return SerialProtocol(pulses=body)

    def visitWhile_stmt(self, ctx: AtomicParser.While_stmtContext):
        self._loop_depth += 1
        try:
            cond = self.visit(ctx.cond())
            body = self.visit(ctx.block())
            return While(condition=cond, body=body)
        finally:
            self._loop_depth -= 1

    def visitIfelse_stmt(self, ctx: AtomicParser.Ifelse_stmtContext):
        cond = self.visit(ctx.cond())
        blocks = list(ctx.block())
        then_branch = self.visit(blocks[0]) if blocks else []
        else_branch = self.visit(blocks[1]) if len(blocks) > 1 else []
        return IfElse(condition=cond, then_branch=then_branch, else_branch=else_branch)

    def visitBreak_stmt(self, ctx):
        if self._loop_depth == 0:
            raise SyntaxError("break outside of loop")
        return Break()

    def visitContinue_stmt(self, ctx):
        if self._loop_depth == 0:
            raise SyntaxError("continue outside of loop")
        return Continue()

    ## Expressions

    def visitExpr(self, ctx: AtomicParser.ExprContext):
        if ctx.bool_and_op() or ctx.bool_or_op():
            left = self.visit(ctx.expr(0))
            right = self.visit(ctx.expr(1))
            if ctx.bool_and_op():
                return BoolAnd(expr1=left, expr2=right)
            return BoolOr(expr1=left, expr2=right)

        if ctx.bool_not_op():
            return BoolNot(expr=self.visit(ctx.expr(0)))
        if ctx.LBRACKET():
            return self.visit(ctx.expr(0))
        if ctx.atomic_list() is not None:
            return self.visit(ctx.atomic_list())

        comps = ctx.comparators()
        aexprs = ctx.aexpr()
        if comps:
            op_cls = _comparator_to_bool_class(comps[0])
            left = self.visit(aexprs[0])
            right = self.visit(aexprs[1])
            return op_cls(expr1=left, expr2=right)

        if ctx.terminal() is not None:
            return self.visit(ctx.terminal())
        if aexprs and len(aexprs) == 1:
            return self.visit(aexprs[0])

        raise ValueError("Undefined value")

    def visitAtomic_list_extract(self, ctx: AtomicParser.Atomic_list_extractContext):
        access = self.visit(ctx.access())
        index = int(ctx.INT().getText())
        return Extract(access=access, index=index)

    def visitAtomic_list(self, ctx: AtomicParser.Atomic_listContext):
        values = [self.visit(e) for e in ctx.expr()]
        return AtomicList(values=values)

    def visitTerminal(self, ctx: AtomicParser.TerminalContext):
        return self.visitChildren(ctx)

    def visitAccess(self, ctx: AtomicParser.AccessContext):
        return Access(name=ctx.ID().getText())

    def visitIon_register(self, ctx: AtomicParser.Ion_registerContext):
        return IonRegister(size=int(ctx.INT().getText()))

    ## Math Terminals ##

    def visitMath_terminal(self, ctx: AtomicParser.Math_terminalContext):
        if ctx.INT() is not None:
            return MathNum(value=int(ctx.INT().getText()))
        if ctx.FLOAT() is not None:
            return MathNum(value=float(ctx.FLOAT().getText()))
        if ctx.MATH_VAR() is not None:
            return MathVar(name=ctx.MATH_VAR().getText())
        if ctx.IMAG() is not None:
            return MathImag()
        if ctx.access() is not None:
            return self.visit(ctx.access())
        if ctx.pexpr() is not None:
            return self.visit(ctx.pexpr())
        if ctx.fexpr() is not None:
            return self.visit(ctx.fexpr())
        if ctx.atomic_list() is not None:
            return self.visit(ctx.atomic_list())
        if ctx.atomic_list_extract() is not None:
            return self.visit(ctx.atomic_list_extract())
        raise ValueError("Empty math_terminal")

    ## Arithmetic Expressions ##

    def visitAexpr(self, ctx: AtomicParser.AexprContext):
        if ctx.getChildCount() == 1:
            return self.visit(ctx.mexpr())
        left = self.visit(ctx.aexpr())
        right = self.visit(ctx.mexpr())
        op_token = None
        for i in range(ctx.getChildCount()):
            c = ctx.getChild(i)
            if isinstance(c, TerminalNodeImpl):
                tt = c.symbol.type
                if tt in (AtomicLexer.PLUS, AtomicLexer.MINUS):
                    op_token = tt
                    break
        if op_token == AtomicLexer.PLUS:
            return MathAdd(expr1=left, expr2=right)
        if op_token == AtomicLexer.MINUS:
            return MathSub(expr1=left, expr2=right)
        return self.visitChildren(ctx)

    def visitMexpr(self, ctx: AtomicParser.MexprContext):
        if ctx.getChildCount() == 1:
            return self.visit(ctx.uexpr())
        left = self.visit(ctx.mexpr())
        right = self.visit(ctx.uexpr())
        op_token = None
        for i in range(ctx.getChildCount()):
            c = ctx.getChild(i)
            if isinstance(c, TerminalNodeImpl):
                tt = c.symbol.type
                if tt in (AtomicLexer.MULT, AtomicLexer.DIV):
                    op_token = tt
                    break
        if op_token == AtomicLexer.MULT:
            return MathMul(expr1=left, expr2=right)
        if op_token == AtomicLexer.DIV:
            return MathDiv(expr1=left, expr2=right)
        return self.visitChildren(ctx)

    def visitUexpr(self, ctx: AtomicParser.UexprContext):
        if ctx.getChildCount() == 1:
            return self.visit(ctx.eexpr())
        sign = None
        for i in range(ctx.getChildCount()):
            c = ctx.getChild(i)
            if isinstance(c, TerminalNodeImpl) and c.symbol.type in (
                AtomicLexer.PLUS,
                AtomicLexer.MINUS,
            ):
                sign = c.symbol.type
                break
        val = self.visit(ctx.eexpr())
        if sign == AtomicLexer.MINUS:
            return MathMul(expr1=MathNum(value=-1), expr2=val)
        return val

    def visitEexpr(self, ctx: AtomicParser.EexprContext):
        if ctx.getChildCount() == 1:
            return self.visit(ctx.terminal())
        base = self.visit(ctx.terminal())
        exp = self.visit(ctx.uexpr())
        return MathPow(expr1=base, expr2=exp)

    def visitPexpr(self, ctx: AtomicParser.PexprContext):
        return self.visit(ctx.aexpr())

    def visitFexpr(self, ctx: AtomicParser.FexprContext):
        fn_ctx = ctx.func_names()
        tt = _get_token_type(fn_ctx.getChild(0))
        args = [self.visit(ax) for ax in ctx.aexpr()]
        if tt == AtomicLexer.BEAM:
            if len(args) != 5:
                raise ValueError(f"beam expects 5 arguments, got {len(args)}")
            return Beam(
                frequency=args[0],
                rabi=args[1],
                phase=args[2],
                polarization=args[3],
                wavevector=args[4],
            )
        if tt == AtomicLexer.PULSE:
            if len(args) == 4:
                return Pulse(
                    beam=args[0], duration=args[1], target=args[2], measured=args[3]
                )
            elif len(args) == 3:
                return Pulse(
                    beam=args[0],
                    duration=args[1],
                    target=args[2],
                    measured=Bool(value=False),
                )
            raise ValueError(f"pulse expects 3/4 arguments, got {len(args)}")
        name = _FUNC_TOKEN_TO_NAME.get(tt)
        if name is None:
            raise ValueError(f"Unknown function token type: {tt}")
        if name == "atan2":
            return MathFunc(func=name, expr=args)

        return MathFunc(func=name, expr=args[0])

    ## Bool Expressions ##

    def visitCond(self, ctx: AtomicParser.CondContext):
        return self.visit(ctx.expr())

    def visitBool_literal(self, ctx: AtomicParser.Bool_literalContext):
        token = ctx.getChild(0).getText()
        if token == "true":
            return Bool(value=True)
        return Bool(value=False)


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
