# Generated from AtomicParser.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .AtomicParser import AtomicParser
else:
    from AtomicParser import AtomicParser

# This class defines a complete listener for a parse tree produced by AtomicParser.
class AtomicParserListener(ParseTreeListener):

    # Enter a parse tree produced by AtomicParser#program.
    def enterProgram(self, ctx:AtomicParser.ProgramContext):
        pass

    # Exit a parse tree produced by AtomicParser#program.
    def exitProgram(self, ctx:AtomicParser.ProgramContext):
        pass


    # Enter a parse tree produced by AtomicParser#statement.
    def enterStatement(self, ctx:AtomicParser.StatementContext):
        pass

    # Exit a parse tree produced by AtomicParser#statement.
    def exitStatement(self, ctx:AtomicParser.StatementContext):
        pass


    # Enter a parse tree produced by AtomicParser#block.
    def enterBlock(self, ctx:AtomicParser.BlockContext):
        pass

    # Exit a parse tree produced by AtomicParser#block.
    def exitBlock(self, ctx:AtomicParser.BlockContext):
        pass


    # Enter a parse tree produced by AtomicParser#atom.
    def enterAtom(self, ctx:AtomicParser.AtomContext):
        pass

    # Exit a parse tree produced by AtomicParser#atom.
    def exitAtom(self, ctx:AtomicParser.AtomContext):
        pass


    # Enter a parse tree produced by AtomicParser#expr.
    def enterExpr(self, ctx:AtomicParser.ExprContext):
        pass

    # Exit a parse tree produced by AtomicParser#expr.
    def exitExpr(self, ctx:AtomicParser.ExprContext):
        pass


    # Enter a parse tree produced by AtomicParser#cond.
    def enterCond(self, ctx:AtomicParser.CondContext):
        pass

    # Exit a parse tree produced by AtomicParser#cond.
    def exitCond(self, ctx:AtomicParser.CondContext):
        pass


    # Enter a parse tree produced by AtomicParser#atomic_list.
    def enterAtomic_list(self, ctx:AtomicParser.Atomic_listContext):
        pass

    # Exit a parse tree produced by AtomicParser#atomic_list.
    def exitAtomic_list(self, ctx:AtomicParser.Atomic_listContext):
        pass


    # Enter a parse tree produced by AtomicParser#declaration.
    def enterDeclaration(self, ctx:AtomicParser.DeclarationContext):
        pass

    # Exit a parse tree produced by AtomicParser#declaration.
    def exitDeclaration(self, ctx:AtomicParser.DeclarationContext):
        pass


    # Enter a parse tree produced by AtomicParser#access.
    def enterAccess(self, ctx:AtomicParser.AccessContext):
        pass

    # Exit a parse tree produced by AtomicParser#access.
    def exitAccess(self, ctx:AtomicParser.AccessContext):
        pass


    # Enter a parse tree produced by AtomicParser#atomic_list_extract.
    def enterAtomic_list_extract(self, ctx:AtomicParser.Atomic_list_extractContext):
        pass

    # Exit a parse tree produced by AtomicParser#atomic_list_extract.
    def exitAtomic_list_extract(self, ctx:AtomicParser.Atomic_list_extractContext):
        pass


    # Enter a parse tree produced by AtomicParser#break_stmt.
    def enterBreak_stmt(self, ctx:AtomicParser.Break_stmtContext):
        pass

    # Exit a parse tree produced by AtomicParser#break_stmt.
    def exitBreak_stmt(self, ctx:AtomicParser.Break_stmtContext):
        pass


    # Enter a parse tree produced by AtomicParser#continue_stmt.
    def enterContinue_stmt(self, ctx:AtomicParser.Continue_stmtContext):
        pass

    # Exit a parse tree produced by AtomicParser#continue_stmt.
    def exitContinue_stmt(self, ctx:AtomicParser.Continue_stmtContext):
        pass


    # Enter a parse tree produced by AtomicParser#while_stmt.
    def enterWhile_stmt(self, ctx:AtomicParser.While_stmtContext):
        pass

    # Exit a parse tree produced by AtomicParser#while_stmt.
    def exitWhile_stmt(self, ctx:AtomicParser.While_stmtContext):
        pass


    # Enter a parse tree produced by AtomicParser#ifelse_stmt.
    def enterIfelse_stmt(self, ctx:AtomicParser.Ifelse_stmtContext):
        pass

    # Exit a parse tree produced by AtomicParser#ifelse_stmt.
    def exitIfelse_stmt(self, ctx:AtomicParser.Ifelse_stmtContext):
        pass


    # Enter a parse tree produced by AtomicParser#ion_register.
    def enterIon_register(self, ctx:AtomicParser.Ion_registerContext):
        pass

    # Exit a parse tree produced by AtomicParser#ion_register.
    def exitIon_register(self, ctx:AtomicParser.Ion_registerContext):
        pass


    # Enter a parse tree produced by AtomicParser#beam_expr.
    def enterBeam_expr(self, ctx:AtomicParser.Beam_exprContext):
        pass

    # Exit a parse tree produced by AtomicParser#beam_expr.
    def exitBeam_expr(self, ctx:AtomicParser.Beam_exprContext):
        pass


    # Enter a parse tree produced by AtomicParser#vec3.
    def enterVec3(self, ctx:AtomicParser.Vec3Context):
        pass

    # Exit a parse tree produced by AtomicParser#vec3.
    def exitVec3(self, ctx:AtomicParser.Vec3Context):
        pass


    # Enter a parse tree produced by AtomicParser#parallel_stmt.
    def enterParallel_stmt(self, ctx:AtomicParser.Parallel_stmtContext):
        pass

    # Exit a parse tree produced by AtomicParser#parallel_stmt.
    def exitParallel_stmt(self, ctx:AtomicParser.Parallel_stmtContext):
        pass


    # Enter a parse tree produced by AtomicParser#pulse_stmt.
    def enterPulse_stmt(self, ctx:AtomicParser.Pulse_stmtContext):
        pass

    # Exit a parse tree produced by AtomicParser#pulse_stmt.
    def exitPulse_stmt(self, ctx:AtomicParser.Pulse_stmtContext):
        pass


    # Enter a parse tree produced by AtomicParser#measured.
    def enterMeasured(self, ctx:AtomicParser.MeasuredContext):
        pass

    # Exit a parse tree produced by AtomicParser#measured.
    def exitMeasured(self, ctx:AtomicParser.MeasuredContext):
        pass


    # Enter a parse tree produced by AtomicParser#targets.
    def enterTargets(self, ctx:AtomicParser.TargetsContext):
        pass

    # Exit a parse tree produced by AtomicParser#targets.
    def exitTargets(self, ctx:AtomicParser.TargetsContext):
        pass


    # Enter a parse tree produced by AtomicParser#bool_and_op.
    def enterBool_and_op(self, ctx:AtomicParser.Bool_and_opContext):
        pass

    # Exit a parse tree produced by AtomicParser#bool_and_op.
    def exitBool_and_op(self, ctx:AtomicParser.Bool_and_opContext):
        pass


    # Enter a parse tree produced by AtomicParser#bool_or_op.
    def enterBool_or_op(self, ctx:AtomicParser.Bool_or_opContext):
        pass

    # Exit a parse tree produced by AtomicParser#bool_or_op.
    def exitBool_or_op(self, ctx:AtomicParser.Bool_or_opContext):
        pass


    # Enter a parse tree produced by AtomicParser#bool_not_op.
    def enterBool_not_op(self, ctx:AtomicParser.Bool_not_opContext):
        pass

    # Exit a parse tree produced by AtomicParser#bool_not_op.
    def exitBool_not_op(self, ctx:AtomicParser.Bool_not_opContext):
        pass


    # Enter a parse tree produced by AtomicParser#bool_eq_op.
    def enterBool_eq_op(self, ctx:AtomicParser.Bool_eq_opContext):
        pass

    # Exit a parse tree produced by AtomicParser#bool_eq_op.
    def exitBool_eq_op(self, ctx:AtomicParser.Bool_eq_opContext):
        pass


    # Enter a parse tree produced by AtomicParser#bool_not_eq_op.
    def enterBool_not_eq_op(self, ctx:AtomicParser.Bool_not_eq_opContext):
        pass

    # Exit a parse tree produced by AtomicParser#bool_not_eq_op.
    def exitBool_not_eq_op(self, ctx:AtomicParser.Bool_not_eq_opContext):
        pass


    # Enter a parse tree produced by AtomicParser#bool_lt_op.
    def enterBool_lt_op(self, ctx:AtomicParser.Bool_lt_opContext):
        pass

    # Exit a parse tree produced by AtomicParser#bool_lt_op.
    def exitBool_lt_op(self, ctx:AtomicParser.Bool_lt_opContext):
        pass


    # Enter a parse tree produced by AtomicParser#bool_lte_op.
    def enterBool_lte_op(self, ctx:AtomicParser.Bool_lte_opContext):
        pass

    # Exit a parse tree produced by AtomicParser#bool_lte_op.
    def exitBool_lte_op(self, ctx:AtomicParser.Bool_lte_opContext):
        pass


    # Enter a parse tree produced by AtomicParser#bool_gt_op.
    def enterBool_gt_op(self, ctx:AtomicParser.Bool_gt_opContext):
        pass

    # Exit a parse tree produced by AtomicParser#bool_gt_op.
    def exitBool_gt_op(self, ctx:AtomicParser.Bool_gt_opContext):
        pass


    # Enter a parse tree produced by AtomicParser#bool_gte_op.
    def enterBool_gte_op(self, ctx:AtomicParser.Bool_gte_opContext):
        pass

    # Exit a parse tree produced by AtomicParser#bool_gte_op.
    def exitBool_gte_op(self, ctx:AtomicParser.Bool_gte_opContext):
        pass


    # Enter a parse tree produced by AtomicParser#bool_literal.
    def enterBool_literal(self, ctx:AtomicParser.Bool_literalContext):
        pass

    # Exit a parse tree produced by AtomicParser#bool_literal.
    def exitBool_literal(self, ctx:AtomicParser.Bool_literalContext):
        pass


    # Enter a parse tree produced by AtomicParser#comparators.
    def enterComparators(self, ctx:AtomicParser.ComparatorsContext):
        pass

    # Exit a parse tree produced by AtomicParser#comparators.
    def exitComparators(self, ctx:AtomicParser.ComparatorsContext):
        pass


    # Enter a parse tree produced by AtomicParser#math_terminal.
    def enterMath_terminal(self, ctx:AtomicParser.Math_terminalContext):
        pass

    # Exit a parse tree produced by AtomicParser#math_terminal.
    def exitMath_terminal(self, ctx:AtomicParser.Math_terminalContext):
        pass


    # Enter a parse tree produced by AtomicParser#math_func_name.
    def enterMath_func_name(self, ctx:AtomicParser.Math_func_nameContext):
        pass

    # Exit a parse tree produced by AtomicParser#math_func_name.
    def exitMath_func_name(self, ctx:AtomicParser.Math_func_nameContext):
        pass


    # Enter a parse tree produced by AtomicParser#pexpr.
    def enterPexpr(self, ctx:AtomicParser.PexprContext):
        pass

    # Exit a parse tree produced by AtomicParser#pexpr.
    def exitPexpr(self, ctx:AtomicParser.PexprContext):
        pass


    # Enter a parse tree produced by AtomicParser#fexpr.
    def enterFexpr(self, ctx:AtomicParser.FexprContext):
        pass

    # Exit a parse tree produced by AtomicParser#fexpr.
    def exitFexpr(self, ctx:AtomicParser.FexprContext):
        pass


    # Enter a parse tree produced by AtomicParser#aexpr.
    def enterAexpr(self, ctx:AtomicParser.AexprContext):
        pass

    # Exit a parse tree produced by AtomicParser#aexpr.
    def exitAexpr(self, ctx:AtomicParser.AexprContext):
        pass


    # Enter a parse tree produced by AtomicParser#mexpr.
    def enterMexpr(self, ctx:AtomicParser.MexprContext):
        pass

    # Exit a parse tree produced by AtomicParser#mexpr.
    def exitMexpr(self, ctx:AtomicParser.MexprContext):
        pass


    # Enter a parse tree produced by AtomicParser#uexpr.
    def enterUexpr(self, ctx:AtomicParser.UexprContext):
        pass

    # Exit a parse tree produced by AtomicParser#uexpr.
    def exitUexpr(self, ctx:AtomicParser.UexprContext):
        pass


    # Enter a parse tree produced by AtomicParser#eexpr.
    def enterEexpr(self, ctx:AtomicParser.EexprContext):
        pass

    # Exit a parse tree produced by AtomicParser#eexpr.
    def exitEexpr(self, ctx:AtomicParser.EexprContext):
        pass



del AtomicParser