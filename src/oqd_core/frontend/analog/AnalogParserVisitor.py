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


    # Visit a parse tree produced by AnalogParser#statement.
    def visitStatement(self, ctx:AnalogParser.StatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#declaration.
    def visitDeclaration(self, ctx:AnalogParser.DeclarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#atomic_type.
    def visitAtomic_type(self, ctx:AnalogParser.Atomic_typeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#quantum_register.
    def visitQuantum_register(self, ctx:AnalogParser.Quantum_registerContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#mode_register.
    def visitMode_register(self, ctx:AnalogParser.Mode_registerContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#my_list.
    def visitMy_list(self, ctx:AnalogParser.My_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#access.
    def visitAccess(self, ctx:AnalogParser.AccessContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#extract.
    def visitExtract(self, ctx:AnalogParser.ExtractContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#evolve_stmt.
    def visitEvolve_stmt(self, ctx:AnalogParser.Evolve_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#measure_stmt.
    def visitMeasure_stmt(self, ctx:AnalogParser.Measure_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#init_stmt.
    def visitInit_stmt(self, ctx:AnalogParser.Init_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#targets.
    def visitTargets(self, ctx:AnalogParser.TargetsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#if_else_stmt.
    def visitIf_else_stmt(self, ctx:AnalogParser.If_else_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#while_stmt.
    def visitWhile_stmt(self, ctx:AnalogParser.While_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#bool_and_op.
    def visitBool_and_op(self, ctx:AnalogParser.Bool_and_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#bool_or_op.
    def visitBool_or_op(self, ctx:AnalogParser.Bool_or_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#bool_not_op.
    def visitBool_not_op(self, ctx:AnalogParser.Bool_not_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#bool_expr.
    def visitBool_expr(self, ctx:AnalogParser.Bool_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#pauli_op.
    def visitPauli_op(self, ctx:AnalogParser.Pauli_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#ladder_op.
    def visitLadder_op(self, ctx:AnalogParser.Ladder_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#operator_expr.
    def visitOperator_expr(self, ctx:AnalogParser.Operator_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#operator_terminal.
    def visitOperator_terminal(self, ctx:AnalogParser.Operator_terminalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#math_expr.
    def visitMath_expr(self, ctx:AnalogParser.Math_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#math_terminal.
    def visitMath_terminal(self, ctx:AnalogParser.Math_terminalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#math_func_name.
    def visitMath_func_name(self, ctx:AnalogParser.Math_func_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AnalogParser#math_func.
    def visitMath_func(self, ctx:AnalogParser.Math_funcContext):
        return self.visitChildren(ctx)



del AnalogParser