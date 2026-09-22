# Generated from AnalogParser.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .AnalogParser import AnalogParser
else:
    from AnalogParser import AnalogParser

# This class defines a complete generic visitor for a parse tree produced by AnalogParser.

class AnalogParserVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by AnalogParser#program.
    def visitProgram(self, ctx:AnalogParser.ProgramContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#block.
    def visitBlock(self, ctx:AnalogParser.BlockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#statement.
    def visitStatement(self, ctx:AnalogParser.StatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#ifelse_stmt.
    def visitIfelse_stmt(self, ctx:AnalogParser.Ifelse_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#while_stmt.
    def visitWhile_stmt(self, ctx:AnalogParser.While_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#break_stmt.
    def visitBreak_stmt(self, ctx:AnalogParser.Break_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#continue_stmt.
    def visitContinue_stmt(self, ctx:AnalogParser.Continue_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#terminal.
    def visitTerminal(self, ctx:AnalogParser.TerminalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#declaration.
    def visitDeclaration(self, ctx:AnalogParser.DeclarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#access.
    def visitAccess(self, ctx:AnalogParser.AccessContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#analog_list.
    def visitAnalog_list(self, ctx:AnalogParser.Analog_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#analog_list_extract.
    def visitAnalog_list_extract(self, ctx:AnalogParser.Analog_list_extractContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#pauli_op.
    def visitPauli_op(self, ctx:AnalogParser.Pauli_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#ladder_op.
    def visitLadder_op(self, ctx:AnalogParser.Ladder_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#operator_terminal.
    def visitOperator_terminal(self, ctx:AnalogParser.Operator_terminalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#bool_literal.
    def visitBool_literal(self, ctx:AnalogParser.Bool_literalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#not.
    def visitNot(self, ctx:AnalogParser.NotContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#and.
    def visitAnd(self, ctx:AnalogParser.AndContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#or.
    def visitOr(self, ctx:AnalogParser.OrContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#xor.
    def visitXor(self, ctx:AnalogParser.XorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#math_func.
    def visitMath_func(self, ctx:AnalogParser.Math_funcContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#quantum_func.
    def visitQuantum_func(self, ctx:AnalogParser.Quantum_funcContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#list_func.
    def visitList_func(self, ctx:AnalogParser.List_funcContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#func_names.
    def visitFunc_names(self, ctx:AnalogParser.Func_namesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#args.
    def visitArgs(self, ctx:AnalogParser.ArgsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#func.
    def visitFunc(self, ctx:AnalogParser.FuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#complex.
    def visitComplex(self, ctx:AnalogParser.ComplexContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#math_terminal.
    def visitMath_terminal(self, ctx:AnalogParser.Math_terminalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#pexpr.
    def visitPexpr(self, ctx:AnalogParser.PexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#eexpr.
    def visitEexpr(self, ctx:AnalogParser.EexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#uexpr.
    def visitUexpr(self, ctx:AnalogParser.UexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#mexpr.
    def visitMexpr(self, ctx:AnalogParser.MexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#aexpr.
    def visitAexpr(self, ctx:AnalogParser.AexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#cexpr.
    def visitCexpr(self, ctx:AnalogParser.CexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#eqexpr.
    def visitEqexpr(self, ctx:AnalogParser.EqexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#andexpr.
    def visitAndexpr(self, ctx:AnalogParser.AndexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#xorexpr.
    def visitXorexpr(self, ctx:AnalogParser.XorexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#orexpr.
    def visitOrexpr(self, ctx:AnalogParser.OrexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#expr.
    def visitExpr(self, ctx:AnalogParser.ExprContext):
        return self.visitChildren(ctx)



del AnalogParser