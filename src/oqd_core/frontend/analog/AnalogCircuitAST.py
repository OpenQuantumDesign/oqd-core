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
from antlr4.tree.Tree import TerminalNodeImpl
from oqd_core.frontend.analog.AnalogLexer import AnalogLexer
from oqd_core.frontend.analog.AnalogParser import AnalogParser
from oqd_core.frontend.analog.AnalogParserVisitor import AnalogParserVisitor
from oqd_core.interface.analog import (
    AnalogCircuit,
    Extract,
    Declaration,
    Evolve,
    Initialize,
    Measure,
    Access,
    AnalogList,
    QuantumRegister,
    ModeRegister,
    IfElse,
    While,
    Break,
    Continue,
    MathNum,
    MathVar,
    MathImag,
    MathAdd,
    MathSub,
    MathMul,
    MathDiv,
    MathPow,
    MathFunc,
    BoolAnd,
    BoolOr,
    BoolNot,
    BoolEq,
    BoolNotEq,
    BoolLessThan,
    BoolLessThanEq,
    BoolGreaterThan,
    BoolGreaterThanEq,
    Bool,
)
from oqd_core.interface.analog.expr import (
    PauliI,
    PauliX,
    PauliY,
    PauliZ,
    Creation,
    Annihilation,
    Identity,
    OperatorAdd,
    OperatorSub,
    OperatorMul,
    OperatorKron,
)

########################################################################################

_BOOL_OP_MAP = {
    AnalogLexer.AND: BoolAnd,
    AnalogLexer.AND2: BoolAnd,
    AnalogLexer.OR: BoolOr,
    AnalogLexer.OR2: BoolOr,
    AnalogLexer.EQ: BoolEq,
    AnalogLexer.NEQ: BoolNotEq,
    AnalogLexer.LT: BoolLessThan,
    AnalogLexer.LTE: BoolLessThanEq,
    AnalogLexer.GT: BoolGreaterThan,
    AnalogLexer.GTE: BoolGreaterThanEq,
}

_OP_TERMINAL_MAP = {
    'I': PauliI, 
    'X': PauliX, 
    'Y': PauliY, 
    'Z': PauliZ,
    'C': Creation, 
    'A': Annihilation, 
    'J': Identity,
}

_FUNC_TOKEN_TO_NAME = {
    AnalogLexer.ABS: "abs",
    AnalogLexer.SIN: "sin",
    AnalogLexer.COS: "cos",
    AnalogLexer.TAN: "tan",
    AnalogLexer.EXP: "exp",
    AnalogLexer.LOG: "log",
    AnalogLexer.SINH: "sinh",
    AnalogLexer.COSH: "cosh",
    AnalogLexer.TANH: "tanh",
    AnalogLexer.ATAN: "atan",
    AnalogLexer.ACOS: "acos",
    AnalogLexer.ASIN: "asin",
    AnalogLexer.ATANH: "atanh",
    AnalogLexer.ASINH: "asinh",
    AnalogLexer.ACOSH: "acosh",
    AnalogLexer.HEAVISIDE: "heaviside",
    AnalogLexer.CONJ: "conj",
    AnalogLexer.REAL: "real",
    AnalogLexer.IMAG_FN: "imag",
    AnalogLexer.ATAN2: "atan2",
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

def _comparator_to_bool_class(cmp_ctx: AnalogParser.ComparatorsContext):
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

class _AnalogASTBuilder(AnalogParserVisitor):
    """
    Visitor that converts ANTLR parse tree to Analog interface AST
    """
    def __init__(self):
        self._loop_depth = 0
    
    def visitProgram(self, ctx: AnalogParser.ProgramContext):
        block = ctx.block()
        statements = []
        if block:
            statements = self.visit(block)
        
        return AnalogCircuit(statements=statements)
    
    def visitBlock(self, ctx: AnalogParser.BlockContext):
        statements = []
        for stmt_ctx in ctx.statement():
            stmt = self.visit(stmt_ctx)
            if stmt is not None:
                statements.append(stmt)
        return statements
    
    def visitStatement(self, ctx: AnalogParser.StatementContext):
        # if ctx.expr() is not None:
        #     return self.visit(ctx.expr())
        child = ctx.getChild(0)
        return self.visit(child)
    
    def visitDeclaration(self, ctx: AnalogParser.DeclarationContext):
        name = ctx.ID().getText()
        val = self.visit(ctx.expr())
        decl = Declaration(name=name, value=val)
        return decl
    
    ## Statements ##
    
    def visitTargets(self, ctx: AnalogParser.TargetsContext):
        return self.visit(ctx.expr())
    
    def visitWhile_stmt(self, ctx: AnalogParser.While_stmtContext):
        self._loop_depth += 1
        try:
            cond = self.visit(ctx.cond())
            body = self.visit(ctx.block())
            return While(condition=cond, body=body)
        finally:
            self._loop_depth -= 1
    
    def visitIfelse_stmt(self, ctx: AnalogParser.Ifelse_stmtContext):
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
    
    ## Expressions ##
    
    def visitExpr(self, ctx: AnalogParser.ExprContext):
        
        if ctx.bool_and_op() or ctx.bool_or_op():
            left = self.visit(ctx.expr()[0])
            right = self.visit(ctx.expr()[1])
            if ctx.bool_and_op():
                return BoolAnd(expr1=left, expr2=right)
            return BoolOr(expr1=left, expr2=right)
        
        if ctx.bool_not_op():
            return BoolNot(expr=self.visit(ctx.expr(0)))
        if ctx.LBRACKET():
            return self.visit(ctx.expr(0))
        if ctx.analog_list_extract() is not None:
            return self.visit(ctx.analog_list_extract())
        if ctx.analog_list() is not None:
            return self.visit(ctx.analog_list())
        
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
        
        raise ValueError('Undefined value')
    
    def visitAnalog_list(self, ctx: AnalogParser.Analog_listContext):
        values = [self.visit(e) for e in ctx.expr()]
        return AnalogList(values=values)
    
    def visitAnalog_list_extract(self, ctx: AnalogParser.Analog_list_extractContext):
        return Extract(
            access=self.visit(ctx.access()),
            index=int(ctx.INT().getText()),
        )
    
    def visitTerminal(self, ctx: AnalogParser.TerminalContext):
        return self.visitChildren(ctx)
    
    def visitAccess(self, ctx: AnalogParser.AccessContext):
        return Access(name=ctx.ID().getText())
    
    ## Register and operator terminals ##
    
    def visitQuantum_register(self, ctx: AnalogParser.Quantum_registerContext):
        return QuantumRegister(size=int(ctx.INT().getText()))
    
    def visitMode_register(self, ctx: AnalogParser.Mode_registerContext):
        return ModeRegister(size=int(ctx.INT().getText()))
    
    def visitOperator_terminal(self, ctx: AnalogParser.Operator_terminalContext):
        child = ctx.getChild(0)
        text = _get_text(child)
        if text[0] != '%':
            raise ValueError("Operator terminals must begin with '%'")
        
        op = _OP_TERMINAL_MAP.get(text[1])
        if op is None:
            raise ValueError(f"Unknown operator terminal: {text}")
        return op()
        
    ## Math Terminals ##
    
    def visitMath_terminal(self, ctx: AnalogParser.Math_terminalContext):
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
        raise ValueError("Empty math_terminal")
    
    ## Arithmetic Expressions ##
    
    
    def visitAexpr(self, ctx: AnalogParser.AexprContext):
        if ctx.getChildCount() == 1:
            return self.visit(ctx.mexpr())
        left = self.visit(ctx.aexpr())
        right = self.visit(ctx.mexpr())
        op_token = None
        for i in range(ctx.getChildCount()):
            c = ctx.getChild(i)
            if isinstance(c, TerminalNodeImpl):
                tt = c.symbol.type
                if tt in (AnalogLexer.PLUS, AnalogLexer.MINUS, AnalogLexer.OP_ADD, AnalogLexer.OP_MINUS):
                    op_token = tt
                    break
        if op_token == AnalogLexer.OP_ADD:
            return OperatorAdd(op1=left, op2=right)
        if op_token == AnalogLexer.OP_MINUS:
            return OperatorSub(op1=left, op2=right)
        if op_token == AnalogLexer.PLUS:
            return MathAdd(expr1=left, expr2=right)
        if op_token == AnalogLexer.MINUS:
            return MathSub(expr1=left, expr2=right)
        return self.visitChildren(ctx)
    
    def visitMexpr(self, ctx: AnalogParser.MexprContext):
        if ctx.getChildCount() == 1:
            return self.visit(ctx.uexpr())
        left = self.visit(ctx.mexpr())
        right = self.visit(ctx.uexpr())
        op_token = None
        for i in range(ctx.getChildCount()):
            c = ctx.getChild(i)
            if isinstance(c, TerminalNodeImpl):
                tt = c.symbol.type
                if tt in (AnalogLexer.MULT, AnalogLexer.DIV, AnalogLexer.OP_MUL, AnalogLexer.AT):
                    op_token = tt
                    break
        if op_token == AnalogLexer.AT:
            return OperatorKron(op1=left, op2=right)
        if op_token == AnalogLexer.OP_MUL:
            return OperatorMul(op1=left, op2=right)
        if op_token == AnalogLexer.MULT:
            return MathMul(expr1=left, expr2=right)
        if op_token == AnalogLexer.DIV:
            return MathDiv(expr1=left, expr2=right)
        return self.visitChildren(ctx)
    
    def visitUexpr(self, ctx: AnalogParser.UexprContext):
        if ctx.getChildCount() == 1:
            return self.visit(ctx.eexpr())
        sign = None
        for i in range(ctx.getChildCount()):
            c = ctx.getChild(i)
            if isinstance(c, TerminalNodeImpl) and c.symbol.type in (AnalogLexer.PLUS, AnalogLexer.MINUS):
                sign = c.symbol.type
                break
        val = self.visit(ctx.eexpr())
        if sign == AnalogLexer.MINUS:
            return MathMul(expr1=MathNum(value=-1), expr2=val)
        return val
    
    def visitEexpr(self, ctx: AnalogParser.EexprContext):
        if ctx.getChildCount() == 1:
            return self.visit(ctx.terminal())
        base = self.visit(ctx.terminal())
        exp = self.visit(ctx.uexpr())
        return MathPow(expr1=base, expr2=exp)
    
    def visitPexpr(self, ctx: AnalogParser.PexprContext):
        return self.visit(ctx.aexpr())
    
    def visitFexpr(self, ctx: AnalogParser.FexprContext):
        fn_ctx = ctx.func_names()
        tt = _get_token_type(fn_ctx.getChild(0))
        args = [self.visit(ax) for ax in ctx.aexpr()]
        if tt == AnalogLexer.EVOLVE:
            if len(args) != 3:
                raise ValueError(f"evolve expects 3 arguments, got {len(args)}")
            return Evolve(hamiltonian=args[1], duration=args[2], targets=args[0])
        if tt == AnalogLexer.MEASURE:
            if len(args) != 1:
                raise ValueError(f"measure expects 1 argument, got {len(args)}")
            return Measure(targets=args[0])
        if tt == AnalogLexer.INITIALIZE:
            if len(args) != 1:
                raise ValueError(f"initialize expects 1 argument, got {len(args)}")
            return Initialize(targets=args[0])
        name = _FUNC_TOKEN_TO_NAME.get(tt)
        if name is None:
            raise ValueError(f"Unknown function token type: {tt}")
        if name == "atan2":
            return MathFunc(func=name, expr=args)
        
        return MathFunc(func=name, expr=args[0])
    
    ## Bool Expressions ##

    def visitCond(self, ctx: AnalogParser.CondContext):
        return self.visit(ctx.expr())
    
    def visitBool_literal(self, ctx: AnalogParser.Bool_literalContext):
        token = ctx.getChild(0).getText()
        if token == 'true':
            return Bool(value=True)
        return Bool(value=False)
    