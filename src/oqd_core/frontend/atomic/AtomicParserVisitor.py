# Generated from AtomicParser.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .AtomicParser import AtomicParser
else:
    from AtomicParser import AtomicParser

# This class defines a complete generic visitor for a parse tree produced by AtomicParser.

class AtomicParserVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by AtomicParser#program.
    def visitProgram(self, ctx:AtomicParser.ProgramContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#statement.
    def visitStatement(self, ctx:AtomicParser.StatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#block.
    def visitBlock(self, ctx:AtomicParser.BlockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#atom.
    def visitAtom(self, ctx:AtomicParser.AtomContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#expr.
    def visitExpr(self, ctx:AtomicParser.ExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#cond.
    def visitCond(self, ctx:AtomicParser.CondContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#my_list.
    def visitMy_list(self, ctx:AtomicParser.My_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#declaration.
    def visitDeclaration(self, ctx:AtomicParser.DeclarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#access.
    def visitAccess(self, ctx:AtomicParser.AccessContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#extract.
    def visitExtract(self, ctx:AtomicParser.ExtractContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#break_stmt.
    def visitBreak_stmt(self, ctx:AtomicParser.Break_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#continue_stmt.
    def visitContinue_stmt(self, ctx:AtomicParser.Continue_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#while_stmt.
    def visitWhile_stmt(self, ctx:AtomicParser.While_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#ifelse_stmt.
    def visitIfelse_stmt(self, ctx:AtomicParser.Ifelse_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#ion_register.
    def visitIon_register(self, ctx:AtomicParser.Ion_registerContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#beam_expr.
    def visitBeam_expr(self, ctx:AtomicParser.Beam_exprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#vec3.
    def visitVec3(self, ctx:AtomicParser.Vec3Context):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#parallel_stmt.
    def visitParallel_stmt(self, ctx:AtomicParser.Parallel_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#pulse_stmt.
    def visitPulse_stmt(self, ctx:AtomicParser.Pulse_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#measured.
    def visitMeasured(self, ctx:AtomicParser.MeasuredContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#targets.
    def visitTargets(self, ctx:AtomicParser.TargetsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#bool_and_op.
    def visitBool_and_op(self, ctx:AtomicParser.Bool_and_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#bool_or_op.
    def visitBool_or_op(self, ctx:AtomicParser.Bool_or_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#bool_not_op.
    def visitBool_not_op(self, ctx:AtomicParser.Bool_not_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#bool_eq_op.
    def visitBool_eq_op(self, ctx:AtomicParser.Bool_eq_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#bool_not_eq_op.
    def visitBool_not_eq_op(self, ctx:AtomicParser.Bool_not_eq_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#bool_lt_op.
    def visitBool_lt_op(self, ctx:AtomicParser.Bool_lt_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#bool_lte_op.
    def visitBool_lte_op(self, ctx:AtomicParser.Bool_lte_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#bool_gt_op.
    def visitBool_gt_op(self, ctx:AtomicParser.Bool_gt_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#bool_gte_op.
    def visitBool_gte_op(self, ctx:AtomicParser.Bool_gte_opContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#bool_literal.
    def visitBool_literal(self, ctx:AtomicParser.Bool_literalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#comparators.
    def visitComparators(self, ctx:AtomicParser.ComparatorsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#math_terminal.
    def visitMath_terminal(self, ctx:AtomicParser.Math_terminalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#math_func_name.
    def visitMath_func_name(self, ctx:AtomicParser.Math_func_nameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#pexpr.
    def visitPexpr(self, ctx:AtomicParser.PexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#fexpr.
    def visitFexpr(self, ctx:AtomicParser.FexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#aexpr.
    def visitAexpr(self, ctx:AtomicParser.AexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#mexpr.
    def visitMexpr(self, ctx:AtomicParser.MexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#uexpr.
    def visitUexpr(self, ctx:AtomicParser.UexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#eexpr.
    def visitEexpr(self, ctx:AtomicParser.EexprContext):
        return self.visitChildren(ctx)



del AtomicParser