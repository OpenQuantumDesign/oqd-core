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


    # Visit a parse tree produced by AtomicParser#parallel_stmt.
    def visitParallel_stmt(self, ctx:AtomicParser.Parallel_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#serial_stmt.
    def visitSerial_stmt(self, ctx:AtomicParser.Serial_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#ifelse_stmt.
    def visitIfelse_stmt(self, ctx:AtomicParser.Ifelse_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#while_stmt.
    def visitWhile_stmt(self, ctx:AtomicParser.While_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#break_stmt.
    def visitBreak_stmt(self, ctx:AtomicParser.Break_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#continue_stmt.
    def visitContinue_stmt(self, ctx:AtomicParser.Continue_stmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#terminal.
    def visitTerminal(self, ctx:AtomicParser.TerminalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#declaration.
    def visitDeclaration(self, ctx:AtomicParser.DeclarationContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#access.
    def visitAccess(self, ctx:AtomicParser.AccessContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#bool_literal.
    def visitBool_literal(self, ctx:AtomicParser.Bool_literalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#arith_func.
    def visitArith_func(self, ctx:AtomicParser.Arith_funcContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#math_func.
    def visitMath_func(self, ctx:AtomicParser.Math_funcContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#quantum_func.
    def visitQuantum_func(self, ctx:AtomicParser.Quantum_funcContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#list_func.
    def visitList_func(self, ctx:AtomicParser.List_funcContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#misc_func.
    def visitMisc_func(self, ctx:AtomicParser.Misc_funcContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#func_names.
    def visitFunc_names(self, ctx:AtomicParser.Func_namesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#args.
    def visitArgs(self, ctx:AtomicParser.ArgsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#func.
    def visitFunc(self, ctx:AtomicParser.FuncContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#atomic_list.
    def visitAtomic_list(self, ctx:AtomicParser.Atomic_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#iexpr.
    def visitIexpr(self, ctx:AtomicParser.IexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#complex.
    def visitComplex(self, ctx:AtomicParser.ComplexContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#math_terminal.
    def visitMath_terminal(self, ctx:AtomicParser.Math_terminalContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#pexpr.
    def visitPexpr(self, ctx:AtomicParser.PexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#eexpr.
    def visitEexpr(self, ctx:AtomicParser.EexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#uexpr.
    def visitUexpr(self, ctx:AtomicParser.UexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#mexpr.
    def visitMexpr(self, ctx:AtomicParser.MexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#aexpr.
    def visitAexpr(self, ctx:AtomicParser.AexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#cexpr.
    def visitCexpr(self, ctx:AtomicParser.CexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#eqexpr.
    def visitEqexpr(self, ctx:AtomicParser.EqexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#andexpr.
    def visitAndexpr(self, ctx:AtomicParser.AndexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#xorexpr.
    def visitXorexpr(self, ctx:AtomicParser.XorexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#orexpr.
    def visitOrexpr(self, ctx:AtomicParser.OrexprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by AtomicParser#expr.
    def visitExpr(self, ctx:AtomicParser.ExprContext):
        return self.visitChildren(ctx)



del AtomicParser