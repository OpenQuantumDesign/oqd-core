# Generated from AnalogParser.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .AnalogParser import AnalogParser
else:
    from AnalogParser import AnalogParser

# This class defines a complete listener for a parse tree produced by AnalogParser.
class AnalogParserListener(ParseTreeListener):

    # Enter a parse tree produced by AnalogParser#program.
    def enterProgram(self, ctx:AnalogParser.ProgramContext):
        pass

    # Exit a parse tree produced by AnalogParser#program.
    def exitProgram(self, ctx:AnalogParser.ProgramContext):
        pass


    # Enter a parse tree produced by AnalogParser#statement.
    def enterStatement(self, ctx:AnalogParser.StatementContext):
        pass

    # Exit a parse tree produced by AnalogParser#statement.
    def exitStatement(self, ctx:AnalogParser.StatementContext):
        pass


    # Enter a parse tree produced by AnalogParser#declaration.
    def enterDeclaration(self, ctx:AnalogParser.DeclarationContext):
        pass

    # Exit a parse tree produced by AnalogParser#declaration.
    def exitDeclaration(self, ctx:AnalogParser.DeclarationContext):
        pass


    # Enter a parse tree produced by AnalogParser#atomic_type.
    def enterAtomic_type(self, ctx:AnalogParser.Atomic_typeContext):
        pass

    # Exit a parse tree produced by AnalogParser#atomic_type.
    def exitAtomic_type(self, ctx:AnalogParser.Atomic_typeContext):
        pass


    # Enter a parse tree produced by AnalogParser#quantum_register.
    def enterQuantum_register(self, ctx:AnalogParser.Quantum_registerContext):
        pass

    # Exit a parse tree produced by AnalogParser#quantum_register.
    def exitQuantum_register(self, ctx:AnalogParser.Quantum_registerContext):
        pass


    # Enter a parse tree produced by AnalogParser#mode_register.
    def enterMode_register(self, ctx:AnalogParser.Mode_registerContext):
        pass

    # Exit a parse tree produced by AnalogParser#mode_register.
    def exitMode_register(self, ctx:AnalogParser.Mode_registerContext):
        pass


    # Enter a parse tree produced by AnalogParser#my_list.
    def enterMy_list(self, ctx:AnalogParser.My_listContext):
        pass

    # Exit a parse tree produced by AnalogParser#my_list.
    def exitMy_list(self, ctx:AnalogParser.My_listContext):
        pass


    # Enter a parse tree produced by AnalogParser#access.
    def enterAccess(self, ctx:AnalogParser.AccessContext):
        pass

    # Exit a parse tree produced by AnalogParser#access.
    def exitAccess(self, ctx:AnalogParser.AccessContext):
        pass


    # Enter a parse tree produced by AnalogParser#extract.
    def enterExtract(self, ctx:AnalogParser.ExtractContext):
        pass

    # Exit a parse tree produced by AnalogParser#extract.
    def exitExtract(self, ctx:AnalogParser.ExtractContext):
        pass


    # Enter a parse tree produced by AnalogParser#evolve_stmt.
    def enterEvolve_stmt(self, ctx:AnalogParser.Evolve_stmtContext):
        pass

    # Exit a parse tree produced by AnalogParser#evolve_stmt.
    def exitEvolve_stmt(self, ctx:AnalogParser.Evolve_stmtContext):
        pass


    # Enter a parse tree produced by AnalogParser#measure_stmt.
    def enterMeasure_stmt(self, ctx:AnalogParser.Measure_stmtContext):
        pass

    # Exit a parse tree produced by AnalogParser#measure_stmt.
    def exitMeasure_stmt(self, ctx:AnalogParser.Measure_stmtContext):
        pass


    # Enter a parse tree produced by AnalogParser#init_stmt.
    def enterInit_stmt(self, ctx:AnalogParser.Init_stmtContext):
        pass

    # Exit a parse tree produced by AnalogParser#init_stmt.
    def exitInit_stmt(self, ctx:AnalogParser.Init_stmtContext):
        pass


    # Enter a parse tree produced by AnalogParser#targets.
    def enterTargets(self, ctx:AnalogParser.TargetsContext):
        pass

    # Exit a parse tree produced by AnalogParser#targets.
    def exitTargets(self, ctx:AnalogParser.TargetsContext):
        pass


    # Enter a parse tree produced by AnalogParser#if_else_stmt.
    def enterIf_else_stmt(self, ctx:AnalogParser.If_else_stmtContext):
        pass

    # Exit a parse tree produced by AnalogParser#if_else_stmt.
    def exitIf_else_stmt(self, ctx:AnalogParser.If_else_stmtContext):
        pass


    # Enter a parse tree produced by AnalogParser#while_stmt.
    def enterWhile_stmt(self, ctx:AnalogParser.While_stmtContext):
        pass

    # Exit a parse tree produced by AnalogParser#while_stmt.
    def exitWhile_stmt(self, ctx:AnalogParser.While_stmtContext):
        pass


    # Enter a parse tree produced by AnalogParser#bool_and_op.
    def enterBool_and_op(self, ctx:AnalogParser.Bool_and_opContext):
        pass

    # Exit a parse tree produced by AnalogParser#bool_and_op.
    def exitBool_and_op(self, ctx:AnalogParser.Bool_and_opContext):
        pass


    # Enter a parse tree produced by AnalogParser#bool_or_op.
    def enterBool_or_op(self, ctx:AnalogParser.Bool_or_opContext):
        pass

    # Exit a parse tree produced by AnalogParser#bool_or_op.
    def exitBool_or_op(self, ctx:AnalogParser.Bool_or_opContext):
        pass


    # Enter a parse tree produced by AnalogParser#bool_not_op.
    def enterBool_not_op(self, ctx:AnalogParser.Bool_not_opContext):
        pass

    # Exit a parse tree produced by AnalogParser#bool_not_op.
    def exitBool_not_op(self, ctx:AnalogParser.Bool_not_opContext):
        pass


    # Enter a parse tree produced by AnalogParser#bool_expr.
    def enterBool_expr(self, ctx:AnalogParser.Bool_exprContext):
        pass

    # Exit a parse tree produced by AnalogParser#bool_expr.
    def exitBool_expr(self, ctx:AnalogParser.Bool_exprContext):
        pass


    # Enter a parse tree produced by AnalogParser#pauli_op.
    def enterPauli_op(self, ctx:AnalogParser.Pauli_opContext):
        pass

    # Exit a parse tree produced by AnalogParser#pauli_op.
    def exitPauli_op(self, ctx:AnalogParser.Pauli_opContext):
        pass


    # Enter a parse tree produced by AnalogParser#ladder_op.
    def enterLadder_op(self, ctx:AnalogParser.Ladder_opContext):
        pass

    # Exit a parse tree produced by AnalogParser#ladder_op.
    def exitLadder_op(self, ctx:AnalogParser.Ladder_opContext):
        pass


    # Enter a parse tree produced by AnalogParser#operator_expr.
    def enterOperator_expr(self, ctx:AnalogParser.Operator_exprContext):
        pass

    # Exit a parse tree produced by AnalogParser#operator_expr.
    def exitOperator_expr(self, ctx:AnalogParser.Operator_exprContext):
        pass


    # Enter a parse tree produced by AnalogParser#operator_terminal.
    def enterOperator_terminal(self, ctx:AnalogParser.Operator_terminalContext):
        pass

    # Exit a parse tree produced by AnalogParser#operator_terminal.
    def exitOperator_terminal(self, ctx:AnalogParser.Operator_terminalContext):
        pass


    # Enter a parse tree produced by AnalogParser#math_expr.
    def enterMath_expr(self, ctx:AnalogParser.Math_exprContext):
        pass

    # Exit a parse tree produced by AnalogParser#math_expr.
    def exitMath_expr(self, ctx:AnalogParser.Math_exprContext):
        pass


    # Enter a parse tree produced by AnalogParser#math_terminal.
    def enterMath_terminal(self, ctx:AnalogParser.Math_terminalContext):
        pass

    # Exit a parse tree produced by AnalogParser#math_terminal.
    def exitMath_terminal(self, ctx:AnalogParser.Math_terminalContext):
        pass


    # Enter a parse tree produced by AnalogParser#math_func_name.
    def enterMath_func_name(self, ctx:AnalogParser.Math_func_nameContext):
        pass

    # Exit a parse tree produced by AnalogParser#math_func_name.
    def exitMath_func_name(self, ctx:AnalogParser.Math_func_nameContext):
        pass


    # Enter a parse tree produced by AnalogParser#math_func.
    def enterMath_func(self, ctx:AnalogParser.Math_funcContext):
        pass

    # Exit a parse tree produced by AnalogParser#math_func.
    def exitMath_func(self, ctx:AnalogParser.Math_funcContext):
        pass



del AnalogParser