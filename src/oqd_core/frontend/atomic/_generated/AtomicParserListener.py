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


    # Enter a parse tree produced by AtomicParser#parallel_stmt.
    def enterParallel_stmt(self, ctx:AtomicParser.Parallel_stmtContext):
        pass

    # Exit a parse tree produced by AtomicParser#parallel_stmt.
    def exitParallel_stmt(self, ctx:AtomicParser.Parallel_stmtContext):
        pass


    # Enter a parse tree produced by AtomicParser#serial_stmt.
    def enterSerial_stmt(self, ctx:AtomicParser.Serial_stmtContext):
        pass

    # Exit a parse tree produced by AtomicParser#serial_stmt.
    def exitSerial_stmt(self, ctx:AtomicParser.Serial_stmtContext):
        pass


    # Enter a parse tree produced by AtomicParser#ifelse_stmt.
    def enterIfelse_stmt(self, ctx:AtomicParser.Ifelse_stmtContext):
        pass

    # Exit a parse tree produced by AtomicParser#ifelse_stmt.
    def exitIfelse_stmt(self, ctx:AtomicParser.Ifelse_stmtContext):
        pass


    # Enter a parse tree produced by AtomicParser#while_stmt.
    def enterWhile_stmt(self, ctx:AtomicParser.While_stmtContext):
        pass

    # Exit a parse tree produced by AtomicParser#while_stmt.
    def exitWhile_stmt(self, ctx:AtomicParser.While_stmtContext):
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


    # Enter a parse tree produced by AtomicParser#terminal.
    def enterTerminal(self, ctx:AtomicParser.TerminalContext):
        pass

    # Exit a parse tree produced by AtomicParser#terminal.
    def exitTerminal(self, ctx:AtomicParser.TerminalContext):
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


    # Enter a parse tree produced by AtomicParser#bool_literal.
    def enterBool_literal(self, ctx:AtomicParser.Bool_literalContext):
        pass

    # Exit a parse tree produced by AtomicParser#bool_literal.
    def exitBool_literal(self, ctx:AtomicParser.Bool_literalContext):
        pass


    # Enter a parse tree produced by AtomicParser#arith_func.
    def enterArith_func(self, ctx:AtomicParser.Arith_funcContext):
        pass

    # Exit a parse tree produced by AtomicParser#arith_func.
    def exitArith_func(self, ctx:AtomicParser.Arith_funcContext):
        pass


    # Enter a parse tree produced by AtomicParser#math_func.
    def enterMath_func(self, ctx:AtomicParser.Math_funcContext):
        pass

    # Exit a parse tree produced by AtomicParser#math_func.
    def exitMath_func(self, ctx:AtomicParser.Math_funcContext):
        pass


    # Enter a parse tree produced by AtomicParser#quantum_func.
    def enterQuantum_func(self, ctx:AtomicParser.Quantum_funcContext):
        pass

    # Exit a parse tree produced by AtomicParser#quantum_func.
    def exitQuantum_func(self, ctx:AtomicParser.Quantum_funcContext):
        pass


    # Enter a parse tree produced by AtomicParser#list_func.
    def enterList_func(self, ctx:AtomicParser.List_funcContext):
        pass

    # Exit a parse tree produced by AtomicParser#list_func.
    def exitList_func(self, ctx:AtomicParser.List_funcContext):
        pass


    # Enter a parse tree produced by AtomicParser#misc_func.
    def enterMisc_func(self, ctx:AtomicParser.Misc_funcContext):
        pass

    # Exit a parse tree produced by AtomicParser#misc_func.
    def exitMisc_func(self, ctx:AtomicParser.Misc_funcContext):
        pass


    # Enter a parse tree produced by AtomicParser#func_names.
    def enterFunc_names(self, ctx:AtomicParser.Func_namesContext):
        pass

    # Exit a parse tree produced by AtomicParser#func_names.
    def exitFunc_names(self, ctx:AtomicParser.Func_namesContext):
        pass


    # Enter a parse tree produced by AtomicParser#args.
    def enterArgs(self, ctx:AtomicParser.ArgsContext):
        pass

    # Exit a parse tree produced by AtomicParser#args.
    def exitArgs(self, ctx:AtomicParser.ArgsContext):
        pass


    # Enter a parse tree produced by AtomicParser#func.
    def enterFunc(self, ctx:AtomicParser.FuncContext):
        pass

    # Exit a parse tree produced by AtomicParser#func.
    def exitFunc(self, ctx:AtomicParser.FuncContext):
        pass


    # Enter a parse tree produced by AtomicParser#atomic_list.
    def enterAtomic_list(self, ctx:AtomicParser.Atomic_listContext):
        pass

    # Exit a parse tree produced by AtomicParser#atomic_list.
    def exitAtomic_list(self, ctx:AtomicParser.Atomic_listContext):
        pass


    # Enter a parse tree produced by AtomicParser#iexpr.
    def enterIexpr(self, ctx:AtomicParser.IexprContext):
        pass

    # Exit a parse tree produced by AtomicParser#iexpr.
    def exitIexpr(self, ctx:AtomicParser.IexprContext):
        pass


    # Enter a parse tree produced by AtomicParser#complex.
    def enterComplex(self, ctx:AtomicParser.ComplexContext):
        pass

    # Exit a parse tree produced by AtomicParser#complex.
    def exitComplex(self, ctx:AtomicParser.ComplexContext):
        pass


    # Enter a parse tree produced by AtomicParser#math_terminal.
    def enterMath_terminal(self, ctx:AtomicParser.Math_terminalContext):
        pass

    # Exit a parse tree produced by AtomicParser#math_terminal.
    def exitMath_terminal(self, ctx:AtomicParser.Math_terminalContext):
        pass


    # Enter a parse tree produced by AtomicParser#pexpr.
    def enterPexpr(self, ctx:AtomicParser.PexprContext):
        pass

    # Exit a parse tree produced by AtomicParser#pexpr.
    def exitPexpr(self, ctx:AtomicParser.PexprContext):
        pass


    # Enter a parse tree produced by AtomicParser#eexpr.
    def enterEexpr(self, ctx:AtomicParser.EexprContext):
        pass

    # Exit a parse tree produced by AtomicParser#eexpr.
    def exitEexpr(self, ctx:AtomicParser.EexprContext):
        pass


    # Enter a parse tree produced by AtomicParser#uexpr.
    def enterUexpr(self, ctx:AtomicParser.UexprContext):
        pass

    # Exit a parse tree produced by AtomicParser#uexpr.
    def exitUexpr(self, ctx:AtomicParser.UexprContext):
        pass


    # Enter a parse tree produced by AtomicParser#mexpr.
    def enterMexpr(self, ctx:AtomicParser.MexprContext):
        pass

    # Exit a parse tree produced by AtomicParser#mexpr.
    def exitMexpr(self, ctx:AtomicParser.MexprContext):
        pass


    # Enter a parse tree produced by AtomicParser#aexpr.
    def enterAexpr(self, ctx:AtomicParser.AexprContext):
        pass

    # Exit a parse tree produced by AtomicParser#aexpr.
    def exitAexpr(self, ctx:AtomicParser.AexprContext):
        pass


    # Enter a parse tree produced by AtomicParser#cexpr.
    def enterCexpr(self, ctx:AtomicParser.CexprContext):
        pass

    # Exit a parse tree produced by AtomicParser#cexpr.
    def exitCexpr(self, ctx:AtomicParser.CexprContext):
        pass


    # Enter a parse tree produced by AtomicParser#eqexpr.
    def enterEqexpr(self, ctx:AtomicParser.EqexprContext):
        pass

    # Exit a parse tree produced by AtomicParser#eqexpr.
    def exitEqexpr(self, ctx:AtomicParser.EqexprContext):
        pass


    # Enter a parse tree produced by AtomicParser#andexpr.
    def enterAndexpr(self, ctx:AtomicParser.AndexprContext):
        pass

    # Exit a parse tree produced by AtomicParser#andexpr.
    def exitAndexpr(self, ctx:AtomicParser.AndexprContext):
        pass


    # Enter a parse tree produced by AtomicParser#xorexpr.
    def enterXorexpr(self, ctx:AtomicParser.XorexprContext):
        pass

    # Exit a parse tree produced by AtomicParser#xorexpr.
    def exitXorexpr(self, ctx:AtomicParser.XorexprContext):
        pass


    # Enter a parse tree produced by AtomicParser#orexpr.
    def enterOrexpr(self, ctx:AtomicParser.OrexprContext):
        pass

    # Exit a parse tree produced by AtomicParser#orexpr.
    def exitOrexpr(self, ctx:AtomicParser.OrexprContext):
        pass


    # Enter a parse tree produced by AtomicParser#expr.
    def enterExpr(self, ctx:AtomicParser.ExprContext):
        pass

    # Exit a parse tree produced by AtomicParser#expr.
    def exitExpr(self, ctx:AtomicParser.ExprContext):
        pass



del AtomicParser