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
    Declaration,
    Evolve,
    Initialize,
    Measure,
    Access,
    Extract,
    MyList,
    QuantumRegister,
    ModeRegister,
    IfElse,
    While,
    Break,
    Continue,
)
from oqd_core.interface.analog.operator import (
    Operator,
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
    OperatorScalarMul,
)
from oqd_core.interface.analog.math import (
    MathNum,
    MathVar,
    MathImag,
    MathAdd,
    MathSub,
    MathMul,
    MathDiv,
    MathPow,
    MathFunc,
)
from oqd_core.interface.analog.bool import (
    BoolAnd,
    BoolOr,
    BoolNot,
    BoolEq,
    BoolNotEq,
    BoolLessThan,
    BoolLessThanEq,
    BoolGreaterThan,
    BoolGreaterThanEq,
    BoolFalse,
    BoolTrue,
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
        
        return AnalogCircuit(sequence=statements)
    
    def visitBlock(self, ctx: AnalogParser.BlockContext):
        statements = []
        for stmt_ctx in ctx.statement():
            stmt = self.visit(stmt_ctx)
            if stmt is not None:
                statements.append(stmt)
        return statements
    
    def visitStatement(self, ctx: AnalogParser.StatementContext):
        child = ctx.getChild(0)
        return self.visit(child)
    
    def visitDeclaration(self, ctx: AnalogParser.DeclarationContext):
        name = ctx.ID().getText()
        val = self.visit(ctx.expr())
        decl = Declaration(name=name, value=val)
        return decl
    
    ## Statements ##
    
    def visitEvolve_stmt(self, ctx: AnalogParser.Evolve_stmtContext):
        targets = self.visit(ctx.targets())
        hamiltonian = self.visit(ctx.expr(0))
        duration = self.visit(ctx.expr(1))
        return Evolve(hamiltonian=hamiltonian, duration=duration, targets=targets)

    def visitMeasure_stmt(self, ctx: AnalogParser.Measure_stmtContext):
        self.visit(ctx.targets().expr())
        return Measure()
    
    def visitInit_stmt(self, ctx: AnalogParser.Init_stmtContext):
        self.visit(ctx.targets().expr())
        return Initialize()
    
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
                return BoolAnd(left=left, right=right)
            return BoolOr(left=left, right=right)
        
        if ctx.bool_not_op():
            return BoolNot(expr=self.visit(ctx.expr(0)))
        if ctx.LBRACKET():
            return self.visit(ctx.expr(0))
        if ctx.extract() is not None:
            return self.visit(ctx.extract())
        if ctx.my_list() is not None:
            return self.visit(ctx.my_list())
        
        comps = ctx.comparators()
        aexprs = ctx.aexpr()
        if comps:
            op_cls = _comparator_to_bool_class(comps[0])
            left = self.visit(aexprs[0])
            right = self.visit(aexprs[1])
            return op_cls(left=left, right=right)
        
        if ctx.atom() is not None:
            return self.visit(ctx.atom())
        if aexprs and len(aexprs) == 1:
            return self.visit(aexprs[0])
        
        raise ValueError('Undefined value')
    
    def visitExtract(self, ctx: AnalogParser.ExtractContext):
        access = self.visit(ctx.access())
        index = int(ctx.INT().getText())
        return Extract(access=access, index=index)
    
    def visitMy_list(self, ctx: AnalogParser.My_listContext):
        values = [self.visit(e) for e in ctx.expr()]
        return MyList(values=values)
    
    def visitAtom(self, ctx: AnalogParser.AtomContext):
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
        for i in range(ctx.getChildCount()):
            child = ctx.getChild(i)
            if isinstance(child, TerminalNodeImpl):
                tt = child.symbol.type
                text = child.getText()
                if tt == AnalogLexer.INT:
                    return MathNum(value=int(text))
                if tt == AnalogLexer.FLOAT:
                    return MathNum(value=float(text))
                if tt == AnalogLexer.MATH_VAR:
                    return MathVar(name=text)
                if tt == AnalogLexer.IMAG:
                    return MathImag()
                if tt == AnalogLexer.ID:
                    return Access(name=text)
            else:
                return self.visit(child)
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
            if isinstance(left, Operator) and isinstance(right, Operator):
                return OperatorAdd(op1=left, op2=right)
            return MathAdd(expr1=left, expr2=right)
        if op_token == AnalogLexer.MINUS:
            if isinstance(left, Operator) and isinstance(right, Operator):
                return OperatorSub(op1=left, op2=right)
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
            if isinstance(left, Operator) and not isinstance(right, Operator):
                return OperatorScalarMul(op=left, expr=right)
            if isinstance(right, Operator) and not isinstance(left, Operator):
                return OperatorScalarMul(op=right, expr=left)
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
            if isinstance(val, Operator):
                return OperatorScalarMul(op=val, expr=MathNum(value=-1))
            return MathMul(expr1=MathNum(value=-1), expr2=val)
        return val
    
    def visitEexpr(self, ctx: AnalogParser.EexprContext):
        if ctx.getChildCount() == 1:
            return self.visit(ctx.atom())
        base = self.visit(ctx.atom())
        exp = self.visit(ctx.uexpr())
        return MathPow(expr1=base, expr2=exp)
    
    def visitPexpr(self, ctx: AnalogParser.PexprContext):
        return self.visit(ctx.aexpr())
    
    def visitFexpr(self, ctx: AnalogParser.FexprContext):
        func_name = _get_text(ctx.math_func_name()).lower()
        arg = self.visit(ctx.pexpr())
        return MathFunc(func=func_name, expr=arg)
    
    ## Bool Expressions ##

    def visitCond(self, ctx: AnalogParser.CondContext):
        return self.visit(ctx.expr())
    
    def visitBool_literal(self, ctx: AnalogParser.Bool_literalContext):
        token = ctx.getChild(0).getText()
        if token == 'true':
            return BoolTrue()
        return BoolFalse()
    