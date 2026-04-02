# Generated from AtomicParser.g4 by ANTLR 4.13.2
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,69,303,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,20,
        7,20,2,21,7,21,2,22,7,22,2,23,7,23,2,24,7,24,2,25,7,25,2,26,7,26,
        2,27,7,27,2,28,7,28,2,29,7,29,2,30,7,30,2,31,7,31,2,32,7,32,2,33,
        7,33,2,34,7,34,2,35,7,35,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        3,1,83,8,1,1,2,1,2,1,2,1,2,5,2,89,8,2,10,2,12,2,92,9,2,1,2,3,2,95,
        8,2,1,3,1,3,1,3,3,3,100,8,3,1,4,1,4,1,4,1,4,1,4,4,4,107,8,4,11,4,
        12,4,108,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,3,4,122,8,4,
        1,4,1,4,1,4,3,4,127,8,4,1,4,1,4,5,4,131,8,4,10,4,12,4,134,9,4,1,
        5,1,5,1,6,1,6,3,6,140,8,6,1,6,1,6,5,6,144,8,6,10,6,12,6,147,9,6,
        1,6,1,6,1,7,1,7,1,7,1,7,1,8,1,8,1,9,1,9,1,9,1,9,1,9,1,10,1,10,1,
        11,1,11,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,13,1,13,1,13,1,
        13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,3,
        13,190,8,13,1,13,1,13,1,13,1,13,1,13,3,13,197,8,13,1,14,1,14,1,14,
        1,14,1,14,1,15,1,15,1,15,1,15,1,15,1,16,1,16,1,17,1,17,1,18,1,18,
        1,19,1,19,1,20,1,20,1,21,1,21,1,22,1,22,1,23,1,23,1,24,1,24,1,25,
        1,25,1,26,1,26,1,27,1,27,1,27,1,27,1,27,1,27,3,27,237,8,27,1,28,
        1,28,1,28,1,28,1,28,1,28,1,28,1,28,3,28,247,8,28,1,29,1,29,1,30,
        1,30,1,30,1,30,1,31,1,31,1,31,1,31,1,31,5,31,260,8,31,10,31,12,31,
        263,9,31,3,31,265,8,31,1,31,1,31,1,32,1,32,1,32,1,32,1,32,1,32,5,
        32,275,8,32,10,32,12,32,278,9,32,1,33,1,33,1,33,1,33,1,33,1,33,5,
        33,286,8,33,10,33,12,33,289,9,33,1,34,1,34,1,34,3,34,294,8,34,1,
        35,1,35,1,35,1,35,1,35,3,35,301,8,35,1,35,0,3,8,64,66,36,0,2,4,6,
        8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,
        52,54,56,58,60,62,64,66,68,70,0,7,1,0,16,17,1,0,18,19,1,0,20,21,
        1,0,22,23,2,0,14,15,49,68,1,0,35,36,1,0,33,34,308,0,72,1,0,0,0,2,
        82,1,0,0,0,4,90,1,0,0,0,6,99,1,0,0,0,8,121,1,0,0,0,10,135,1,0,0,
        0,12,137,1,0,0,0,14,150,1,0,0,0,16,154,1,0,0,0,18,156,1,0,0,0,20,
        161,1,0,0,0,22,163,1,0,0,0,24,165,1,0,0,0,26,196,1,0,0,0,28,198,
        1,0,0,0,30,203,1,0,0,0,32,208,1,0,0,0,34,210,1,0,0,0,36,212,1,0,
        0,0,38,214,1,0,0,0,40,216,1,0,0,0,42,218,1,0,0,0,44,220,1,0,0,0,
        46,222,1,0,0,0,48,224,1,0,0,0,50,226,1,0,0,0,52,228,1,0,0,0,54,236,
        1,0,0,0,56,246,1,0,0,0,58,248,1,0,0,0,60,250,1,0,0,0,62,254,1,0,
        0,0,64,268,1,0,0,0,66,279,1,0,0,0,68,293,1,0,0,0,70,300,1,0,0,0,
        72,73,3,4,2,0,73,74,5,0,0,1,74,1,1,0,0,0,75,83,3,14,7,0,76,83,3,
        30,15,0,77,83,3,24,12,0,78,83,3,26,13,0,79,83,3,20,10,0,80,83,3,
        22,11,0,81,83,3,8,4,0,82,75,1,0,0,0,82,76,1,0,0,0,82,77,1,0,0,0,
        82,78,1,0,0,0,82,79,1,0,0,0,82,80,1,0,0,0,82,81,1,0,0,0,83,3,1,0,
        0,0,84,85,3,2,1,0,85,86,5,2,0,0,86,89,1,0,0,0,87,89,5,2,0,0,88,84,
        1,0,0,0,88,87,1,0,0,0,89,92,1,0,0,0,90,88,1,0,0,0,90,91,1,0,0,0,
        91,94,1,0,0,0,92,90,1,0,0,0,93,95,3,2,1,0,94,93,1,0,0,0,94,95,1,
        0,0,0,95,5,1,0,0,0,96,100,3,28,14,0,97,100,3,56,28,0,98,100,3,52,
        26,0,99,96,1,0,0,0,99,97,1,0,0,0,99,98,1,0,0,0,100,7,1,0,0,0,101,
        102,6,4,-1,0,102,106,3,64,32,0,103,104,3,54,27,0,104,105,3,64,32,
        0,105,107,1,0,0,0,106,103,1,0,0,0,107,108,1,0,0,0,108,106,1,0,0,
        0,108,109,1,0,0,0,109,122,1,0,0,0,110,111,3,38,19,0,111,112,3,8,
        4,6,112,122,1,0,0,0,113,114,5,27,0,0,114,115,3,8,4,0,115,116,5,28,
        0,0,116,122,1,0,0,0,117,122,3,18,9,0,118,122,3,12,6,0,119,122,3,
        6,3,0,120,122,3,64,32,0,121,101,1,0,0,0,121,110,1,0,0,0,121,113,
        1,0,0,0,121,117,1,0,0,0,121,118,1,0,0,0,121,119,1,0,0,0,121,120,
        1,0,0,0,122,132,1,0,0,0,123,126,10,7,0,0,124,127,3,34,17,0,125,127,
        3,36,18,0,126,124,1,0,0,0,126,125,1,0,0,0,127,128,1,0,0,0,128,129,
        3,8,4,8,129,131,1,0,0,0,130,123,1,0,0,0,131,134,1,0,0,0,132,130,
        1,0,0,0,132,133,1,0,0,0,133,9,1,0,0,0,134,132,1,0,0,0,135,136,3,
        8,4,0,136,11,1,0,0,0,137,139,5,29,0,0,138,140,3,8,4,0,139,138,1,
        0,0,0,139,140,1,0,0,0,140,145,1,0,0,0,141,142,5,26,0,0,142,144,3,
        8,4,0,143,141,1,0,0,0,144,147,1,0,0,0,145,143,1,0,0,0,145,146,1,
        0,0,0,146,148,1,0,0,0,147,145,1,0,0,0,148,149,5,30,0,0,149,13,1,
        0,0,0,150,151,5,69,0,0,151,152,5,38,0,0,152,153,3,8,4,0,153,15,1,
        0,0,0,154,155,5,69,0,0,155,17,1,0,0,0,156,157,3,16,8,0,157,158,5,
        29,0,0,158,159,5,45,0,0,159,160,5,30,0,0,160,19,1,0,0,0,161,162,
        5,11,0,0,162,21,1,0,0,0,163,164,5,12,0,0,164,23,1,0,0,0,165,166,
        5,8,0,0,166,167,5,27,0,0,167,168,3,10,5,0,168,169,5,28,0,0,169,170,
        5,31,0,0,170,171,3,4,2,0,171,172,5,32,0,0,172,25,1,0,0,0,173,174,
        5,6,0,0,174,175,5,27,0,0,175,176,3,10,5,0,176,177,5,28,0,0,177,178,
        5,31,0,0,178,179,3,4,2,0,179,180,5,32,0,0,180,197,1,0,0,0,181,182,
        5,6,0,0,182,183,5,27,0,0,183,184,3,10,5,0,184,185,5,28,0,0,185,186,
        5,31,0,0,186,187,3,4,2,0,187,189,5,32,0,0,188,190,5,2,0,0,189,188,
        1,0,0,0,189,190,1,0,0,0,190,191,1,0,0,0,191,192,5,7,0,0,192,193,
        5,31,0,0,193,194,3,4,2,0,194,195,5,32,0,0,195,197,1,0,0,0,196,173,
        1,0,0,0,196,181,1,0,0,0,197,27,1,0,0,0,198,199,5,13,0,0,199,200,
        5,27,0,0,200,201,5,45,0,0,201,202,5,28,0,0,202,29,1,0,0,0,203,204,
        5,5,0,0,204,205,5,31,0,0,205,206,3,4,2,0,206,207,5,32,0,0,207,31,
        1,0,0,0,208,209,3,8,4,0,209,33,1,0,0,0,210,211,7,0,0,0,211,35,1,
        0,0,0,212,213,7,1,0,0,213,37,1,0,0,0,214,215,7,2,0,0,215,39,1,0,
        0,0,216,217,5,39,0,0,217,41,1,0,0,0,218,219,5,40,0,0,219,43,1,0,
        0,0,220,221,5,41,0,0,221,45,1,0,0,0,222,223,5,42,0,0,223,47,1,0,
        0,0,224,225,5,43,0,0,225,49,1,0,0,0,226,227,5,44,0,0,227,51,1,0,
        0,0,228,229,7,3,0,0,229,53,1,0,0,0,230,237,3,40,20,0,231,237,3,42,
        21,0,232,237,3,44,22,0,233,237,3,46,23,0,234,237,3,48,24,0,235,237,
        3,50,25,0,236,230,1,0,0,0,236,231,1,0,0,0,236,232,1,0,0,0,236,233,
        1,0,0,0,236,234,1,0,0,0,236,235,1,0,0,0,237,55,1,0,0,0,238,247,5,
        45,0,0,239,247,5,46,0,0,240,247,5,47,0,0,241,247,5,48,0,0,242,247,
        3,16,8,0,243,247,3,60,30,0,244,247,3,62,31,0,245,247,3,12,6,0,246,
        238,1,0,0,0,246,239,1,0,0,0,246,240,1,0,0,0,246,241,1,0,0,0,246,
        242,1,0,0,0,246,243,1,0,0,0,246,244,1,0,0,0,246,245,1,0,0,0,247,
        57,1,0,0,0,248,249,7,4,0,0,249,59,1,0,0,0,250,251,5,27,0,0,251,252,
        3,64,32,0,252,253,5,28,0,0,253,61,1,0,0,0,254,255,3,58,29,0,255,
        264,5,27,0,0,256,261,3,64,32,0,257,258,5,26,0,0,258,260,3,64,32,
        0,259,257,1,0,0,0,260,263,1,0,0,0,261,259,1,0,0,0,261,262,1,0,0,
        0,262,265,1,0,0,0,263,261,1,0,0,0,264,256,1,0,0,0,264,265,1,0,0,
        0,265,266,1,0,0,0,266,267,5,28,0,0,267,63,1,0,0,0,268,269,6,32,-1,
        0,269,270,3,66,33,0,270,276,1,0,0,0,271,272,10,1,0,0,272,273,7,5,
        0,0,273,275,3,66,33,0,274,271,1,0,0,0,275,278,1,0,0,0,276,274,1,
        0,0,0,276,277,1,0,0,0,277,65,1,0,0,0,278,276,1,0,0,0,279,280,6,33,
        -1,0,280,281,3,68,34,0,281,287,1,0,0,0,282,283,10,1,0,0,283,284,
        7,6,0,0,284,286,3,68,34,0,285,282,1,0,0,0,286,289,1,0,0,0,287,285,
        1,0,0,0,287,288,1,0,0,0,288,67,1,0,0,0,289,287,1,0,0,0,290,294,3,
        70,35,0,291,292,7,5,0,0,292,294,3,70,35,0,293,290,1,0,0,0,293,291,
        1,0,0,0,294,69,1,0,0,0,295,301,3,6,3,0,296,297,3,6,3,0,297,298,5,
        37,0,0,298,299,3,68,34,0,299,301,1,0,0,0,300,295,1,0,0,0,300,296,
        1,0,0,0,301,71,1,0,0,0,21,82,88,90,94,99,108,121,126,132,139,145,
        189,196,236,246,261,264,276,287,293,300
    ]

class AtomicParser ( Parser ):

    grammarFileName = "AtomicParser.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                     "<INVALID>", "'parallel'", "'if'", "'else'", "'while'", 
                     "'with'", "'for'", "'break'", "'continue'", "'ionreg'", 
                     "'beam'", "'pulse'", "'and'", "'&&'", "'or'", "'||'", 
                     "'not'", "'!'", "'true'", "'false'", "':'", "';'", 
                     "','", "'('", "')'", "'['", "']'", "'{'", "'}'", "'*'", 
                     "'/'", "'+'", "'-'", "'^'", "'='", "'=='", "'!='", 
                     "'<'", "'<='", "'>'", "'>='", "<INVALID>", "<INVALID>", 
                     "<INVALID>", "'1j'", "'abs'", "'sin'", "'cos'", "'tan'", 
                     "'exp'", "'log'", "'sinh'", "'cosh'", "'tanh'", "'atan'", 
                     "'acos'", "'asin'", "'atanh'", "'asinh'", "'acosh'", 
                     "'heaviside'", "'conj'", "'real'", "'imag'", "'atan2'" ]

    symbolicNames = [ "<INVALID>", "WHITESPACE", "EOL", "NEWLINE", "COMMENT", 
                      "PARALLEL", "IF", "ELSE", "WHILE", "WITH", "FOR", 
                      "BREAK", "CONTINUE", "IONREGISTER", "BEAM", "PULSE", 
                      "AND", "AND2", "OR", "OR2", "NOT", "NOT2", "TRUE", 
                      "FALSE", "COLON", "SEMICOLON", "COMMA", "LBRACKET", 
                      "RBRACKET", "SQUARELBRACKET", "SQUARERBRACKET", "LBRACE", 
                      "RBRACE", "MULT", "DIV", "PLUS", "MINUS", "POWER", 
                      "ASSIGN", "EQ", "NEQ", "LT", "LTE", "GT", "GTE", "INT", 
                      "FLOAT", "MATH_VAR", "IMAG", "ABS", "SIN", "COS", 
                      "TAN", "EXP", "LOG", "SINH", "COSH", "TANH", "ATAN", 
                      "ACOS", "ASIN", "ATANH", "ASINH", "ACOSH", "HEAVISIDE", 
                      "CONJ", "REAL", "IMAG_FN", "ATAN2", "ID" ]

    RULE_program = 0
    RULE_statement = 1
    RULE_block = 2
    RULE_terminal = 3
    RULE_expr = 4
    RULE_cond = 5
    RULE_atomic_list = 6
    RULE_declaration = 7
    RULE_access = 8
    RULE_atomic_list_extract = 9
    RULE_break_stmt = 10
    RULE_continue_stmt = 11
    RULE_while_stmt = 12
    RULE_ifelse_stmt = 13
    RULE_ion_register = 14
    RULE_parallel_stmt = 15
    RULE_targets = 16
    RULE_bool_and_op = 17
    RULE_bool_or_op = 18
    RULE_bool_not_op = 19
    RULE_bool_eq_op = 20
    RULE_bool_not_eq_op = 21
    RULE_bool_lt_op = 22
    RULE_bool_lte_op = 23
    RULE_bool_gt_op = 24
    RULE_bool_gte_op = 25
    RULE_bool_literal = 26
    RULE_comparators = 27
    RULE_math_terminal = 28
    RULE_func_names = 29
    RULE_pexpr = 30
    RULE_fexpr = 31
    RULE_aexpr = 32
    RULE_mexpr = 33
    RULE_uexpr = 34
    RULE_eexpr = 35

    ruleNames =  [ "program", "statement", "block", "terminal", "expr", 
                   "cond", "atomic_list", "declaration", "access", "atomic_list_extract", 
                   "break_stmt", "continue_stmt", "while_stmt", "ifelse_stmt", 
                   "ion_register", "parallel_stmt", "targets", "bool_and_op", 
                   "bool_or_op", "bool_not_op", "bool_eq_op", "bool_not_eq_op", 
                   "bool_lt_op", "bool_lte_op", "bool_gt_op", "bool_gte_op", 
                   "bool_literal", "comparators", "math_terminal", "func_names", 
                   "pexpr", "fexpr", "aexpr", "mexpr", "uexpr", "eexpr" ]

    EOF = Token.EOF
    WHITESPACE=1
    EOL=2
    NEWLINE=3
    COMMENT=4
    PARALLEL=5
    IF=6
    ELSE=7
    WHILE=8
    WITH=9
    FOR=10
    BREAK=11
    CONTINUE=12
    IONREGISTER=13
    BEAM=14
    PULSE=15
    AND=16
    AND2=17
    OR=18
    OR2=19
    NOT=20
    NOT2=21
    TRUE=22
    FALSE=23
    COLON=24
    SEMICOLON=25
    COMMA=26
    LBRACKET=27
    RBRACKET=28
    SQUARELBRACKET=29
    SQUARERBRACKET=30
    LBRACE=31
    RBRACE=32
    MULT=33
    DIV=34
    PLUS=35
    MINUS=36
    POWER=37
    ASSIGN=38
    EQ=39
    NEQ=40
    LT=41
    LTE=42
    GT=43
    GTE=44
    INT=45
    FLOAT=46
    MATH_VAR=47
    IMAG=48
    ABS=49
    SIN=50
    COS=51
    TAN=52
    EXP=53
    LOG=54
    SINH=55
    COSH=56
    TANH=57
    ATAN=58
    ACOS=59
    ASIN=60
    ATANH=61
    ASINH=62
    ACOSH=63
    HEAVISIDE=64
    CONJ=65
    REAL=66
    IMAG_FN=67
    ATAN2=68
    ID=69

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.13.2")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class ProgramContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def block(self):
            return self.getTypedRuleContext(AtomicParser.BlockContext,0)


        def EOF(self):
            return self.getToken(AtomicParser.EOF, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_program

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterProgram" ):
                listener.enterProgram(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitProgram" ):
                listener.exitProgram(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitProgram" ):
                return visitor.visitProgram(self)
            else:
                return visitor.visitChildren(self)




    def program(self):

        localctx = AtomicParser.ProgramContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_program)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 72
            self.block()
            self.state = 73
            self.match(AtomicParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class StatementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def declaration(self):
            return self.getTypedRuleContext(AtomicParser.DeclarationContext,0)


        def parallel_stmt(self):
            return self.getTypedRuleContext(AtomicParser.Parallel_stmtContext,0)


        def while_stmt(self):
            return self.getTypedRuleContext(AtomicParser.While_stmtContext,0)


        def ifelse_stmt(self):
            return self.getTypedRuleContext(AtomicParser.Ifelse_stmtContext,0)


        def break_stmt(self):
            return self.getTypedRuleContext(AtomicParser.Break_stmtContext,0)


        def continue_stmt(self):
            return self.getTypedRuleContext(AtomicParser.Continue_stmtContext,0)


        def expr(self):
            return self.getTypedRuleContext(AtomicParser.ExprContext,0)


        def getRuleIndex(self):
            return AtomicParser.RULE_statement

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterStatement" ):
                listener.enterStatement(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitStatement" ):
                listener.exitStatement(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitStatement" ):
                return visitor.visitStatement(self)
            else:
                return visitor.visitChildren(self)




    def statement(self):

        localctx = AtomicParser.StatementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_statement)
        try:
            self.state = 82
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,0,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 75
                self.declaration()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 76
                self.parallel_stmt()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 77
                self.while_stmt()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 78
                self.ifelse_stmt()
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 79
                self.break_stmt()
                pass

            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 80
                self.continue_stmt()
                pass

            elif la_ == 7:
                self.enterOuterAlt(localctx, 7)
                self.state = 81
                self.expr(0)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class BlockContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def statement(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AtomicParser.StatementContext)
            else:
                return self.getTypedRuleContext(AtomicParser.StatementContext,i)


        def EOL(self, i:int=None):
            if i is None:
                return self.getTokens(AtomicParser.EOL)
            else:
                return self.getToken(AtomicParser.EOL, i)

        def getRuleIndex(self):
            return AtomicParser.RULE_block

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBlock" ):
                listener.enterBlock(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBlock" ):
                listener.exitBlock(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBlock" ):
                return visitor.visitBlock(self)
            else:
                return visitor.visitChildren(self)




    def block(self):

        localctx = AtomicParser.BlockContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_block)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 90
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,2,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    self.state = 88
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [5, 6, 8, 11, 12, 13, 14, 15, 20, 21, 22, 23, 27, 29, 35, 36, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69]:
                        self.state = 84
                        self.statement()
                        self.state = 85
                        self.match(AtomicParser.EOL)
                        pass
                    elif token in [2]:
                        self.state = 87
                        self.match(AtomicParser.EOL)
                        pass
                    else:
                        raise NoViableAltException(self)
             
                self.state = 92
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,2,self._ctx)

            self.state = 94
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if (((_la) & ~0x3f) == 0 and ((1 << _la) & -35080605992608) != 0) or ((((_la - 64)) & ~0x3f) == 0 and ((1 << (_la - 64)) & 63) != 0):
                self.state = 93
                self.statement()


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TerminalContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ion_register(self):
            return self.getTypedRuleContext(AtomicParser.Ion_registerContext,0)


        def math_terminal(self):
            return self.getTypedRuleContext(AtomicParser.Math_terminalContext,0)


        def bool_literal(self):
            return self.getTypedRuleContext(AtomicParser.Bool_literalContext,0)


        def getRuleIndex(self):
            return AtomicParser.RULE_terminal

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTerminal" ):
                listener.enterTerminal(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTerminal" ):
                listener.exitTerminal(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTerminal" ):
                return visitor.visitTerminal(self)
            else:
                return visitor.visitChildren(self)




    def terminal(self):

        localctx = AtomicParser.TerminalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_terminal)
        try:
            self.state = 99
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [13]:
                self.enterOuterAlt(localctx, 1)
                self.state = 96
                self.ion_register()
                pass
            elif token in [14, 15, 27, 29, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69]:
                self.enterOuterAlt(localctx, 2)
                self.state = 97
                self.math_terminal()
                pass
            elif token in [22, 23]:
                self.enterOuterAlt(localctx, 3)
                self.state = 98
                self.bool_literal()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ExprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def aexpr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AtomicParser.AexprContext)
            else:
                return self.getTypedRuleContext(AtomicParser.AexprContext,i)


        def comparators(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AtomicParser.ComparatorsContext)
            else:
                return self.getTypedRuleContext(AtomicParser.ComparatorsContext,i)


        def bool_not_op(self):
            return self.getTypedRuleContext(AtomicParser.Bool_not_opContext,0)


        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AtomicParser.ExprContext)
            else:
                return self.getTypedRuleContext(AtomicParser.ExprContext,i)


        def LBRACKET(self):
            return self.getToken(AtomicParser.LBRACKET, 0)

        def RBRACKET(self):
            return self.getToken(AtomicParser.RBRACKET, 0)

        def atomic_list_extract(self):
            return self.getTypedRuleContext(AtomicParser.Atomic_list_extractContext,0)


        def atomic_list(self):
            return self.getTypedRuleContext(AtomicParser.Atomic_listContext,0)


        def terminal(self):
            return self.getTypedRuleContext(AtomicParser.TerminalContext,0)


        def bool_and_op(self):
            return self.getTypedRuleContext(AtomicParser.Bool_and_opContext,0)


        def bool_or_op(self):
            return self.getTypedRuleContext(AtomicParser.Bool_or_opContext,0)


        def getRuleIndex(self):
            return AtomicParser.RULE_expr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterExpr" ):
                listener.enterExpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitExpr" ):
                listener.exitExpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitExpr" ):
                return visitor.visitExpr(self)
            else:
                return visitor.visitChildren(self)



    def expr(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = AtomicParser.ExprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 8
        self.enterRecursionRule(localctx, 8, self.RULE_expr, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 121
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,6,self._ctx)
            if la_ == 1:
                self.state = 102
                self.aexpr(0)
                self.state = 106 
                self._errHandler.sync(self)
                _alt = 1
                while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 103
                        self.comparators()
                        self.state = 104
                        self.aexpr(0)

                    else:
                        raise NoViableAltException(self)
                    self.state = 108 
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input,5,self._ctx)

                pass

            elif la_ == 2:
                self.state = 110
                self.bool_not_op()
                self.state = 111
                self.expr(6)
                pass

            elif la_ == 3:
                self.state = 113
                self.match(AtomicParser.LBRACKET)
                self.state = 114
                self.expr(0)
                self.state = 115
                self.match(AtomicParser.RBRACKET)
                pass

            elif la_ == 4:
                self.state = 117
                self.atomic_list_extract()
                pass

            elif la_ == 5:
                self.state = 118
                self.atomic_list()
                pass

            elif la_ == 6:
                self.state = 119
                self.terminal()
                pass

            elif la_ == 7:
                self.state = 120
                self.aexpr(0)
                pass


            self._ctx.stop = self._input.LT(-1)
            self.state = 132
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,8,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = AtomicParser.ExprContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                    self.state = 123
                    if not self.precpred(self._ctx, 7):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 7)")
                    self.state = 126
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [16, 17]:
                        self.state = 124
                        self.bool_and_op()
                        pass
                    elif token in [18, 19]:
                        self.state = 125
                        self.bool_or_op()
                        pass
                    else:
                        raise NoViableAltException(self)

                    self.state = 128
                    self.expr(8) 
                self.state = 134
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,8,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class CondContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self):
            return self.getTypedRuleContext(AtomicParser.ExprContext,0)


        def getRuleIndex(self):
            return AtomicParser.RULE_cond

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterCond" ):
                listener.enterCond(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitCond" ):
                listener.exitCond(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitCond" ):
                return visitor.visitCond(self)
            else:
                return visitor.visitChildren(self)




    def cond(self):

        localctx = AtomicParser.CondContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_cond)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 135
            self.expr(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Atomic_listContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def SQUARELBRACKET(self):
            return self.getToken(AtomicParser.SQUARELBRACKET, 0)

        def SQUARERBRACKET(self):
            return self.getToken(AtomicParser.SQUARERBRACKET, 0)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AtomicParser.ExprContext)
            else:
                return self.getTypedRuleContext(AtomicParser.ExprContext,i)


        def COMMA(self, i:int=None):
            if i is None:
                return self.getTokens(AtomicParser.COMMA)
            else:
                return self.getToken(AtomicParser.COMMA, i)

        def getRuleIndex(self):
            return AtomicParser.RULE_atomic_list

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAtomic_list" ):
                listener.enterAtomic_list(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAtomic_list" ):
                listener.exitAtomic_list(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAtomic_list" ):
                return visitor.visitAtomic_list(self)
            else:
                return visitor.visitChildren(self)




    def atomic_list(self):

        localctx = AtomicParser.Atomic_listContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_atomic_list)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 137
            self.match(AtomicParser.SQUARELBRACKET)
            self.state = 139
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if ((((_la - 13)) & ~0x3f) == 0 and ((1 << (_la - 13)) & 144115183793555335) != 0):
                self.state = 138
                self.expr(0)


            self.state = 145
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==26:
                self.state = 141
                self.match(AtomicParser.COMMA)
                self.state = 142
                self.expr(0)
                self.state = 147
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 148
            self.match(AtomicParser.SQUARERBRACKET)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class DeclarationContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(AtomicParser.ID, 0)

        def ASSIGN(self):
            return self.getToken(AtomicParser.ASSIGN, 0)

        def expr(self):
            return self.getTypedRuleContext(AtomicParser.ExprContext,0)


        def getRuleIndex(self):
            return AtomicParser.RULE_declaration

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterDeclaration" ):
                listener.enterDeclaration(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitDeclaration" ):
                listener.exitDeclaration(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitDeclaration" ):
                return visitor.visitDeclaration(self)
            else:
                return visitor.visitChildren(self)




    def declaration(self):

        localctx = AtomicParser.DeclarationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_declaration)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 150
            self.match(AtomicParser.ID)
            self.state = 151
            self.match(AtomicParser.ASSIGN)
            self.state = 152
            self.expr(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AccessContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(AtomicParser.ID, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_access

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAccess" ):
                listener.enterAccess(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAccess" ):
                listener.exitAccess(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAccess" ):
                return visitor.visitAccess(self)
            else:
                return visitor.visitChildren(self)




    def access(self):

        localctx = AtomicParser.AccessContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_access)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 154
            self.match(AtomicParser.ID)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Atomic_list_extractContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def access(self):
            return self.getTypedRuleContext(AtomicParser.AccessContext,0)


        def SQUARELBRACKET(self):
            return self.getToken(AtomicParser.SQUARELBRACKET, 0)

        def INT(self):
            return self.getToken(AtomicParser.INT, 0)

        def SQUARERBRACKET(self):
            return self.getToken(AtomicParser.SQUARERBRACKET, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_atomic_list_extract

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAtomic_list_extract" ):
                listener.enterAtomic_list_extract(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAtomic_list_extract" ):
                listener.exitAtomic_list_extract(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAtomic_list_extract" ):
                return visitor.visitAtomic_list_extract(self)
            else:
                return visitor.visitChildren(self)




    def atomic_list_extract(self):

        localctx = AtomicParser.Atomic_list_extractContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_atomic_list_extract)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 156
            self.access()
            self.state = 157
            self.match(AtomicParser.SQUARELBRACKET)
            self.state = 158
            self.match(AtomicParser.INT)
            self.state = 159
            self.match(AtomicParser.SQUARERBRACKET)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Break_stmtContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def BREAK(self):
            return self.getToken(AtomicParser.BREAK, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_break_stmt

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBreak_stmt" ):
                listener.enterBreak_stmt(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBreak_stmt" ):
                listener.exitBreak_stmt(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBreak_stmt" ):
                return visitor.visitBreak_stmt(self)
            else:
                return visitor.visitChildren(self)




    def break_stmt(self):

        localctx = AtomicParser.Break_stmtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_break_stmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 161
            self.match(AtomicParser.BREAK)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Continue_stmtContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def CONTINUE(self):
            return self.getToken(AtomicParser.CONTINUE, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_continue_stmt

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterContinue_stmt" ):
                listener.enterContinue_stmt(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitContinue_stmt" ):
                listener.exitContinue_stmt(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitContinue_stmt" ):
                return visitor.visitContinue_stmt(self)
            else:
                return visitor.visitChildren(self)




    def continue_stmt(self):

        localctx = AtomicParser.Continue_stmtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_continue_stmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 163
            self.match(AtomicParser.CONTINUE)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class While_stmtContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def WHILE(self):
            return self.getToken(AtomicParser.WHILE, 0)

        def LBRACKET(self):
            return self.getToken(AtomicParser.LBRACKET, 0)

        def cond(self):
            return self.getTypedRuleContext(AtomicParser.CondContext,0)


        def RBRACKET(self):
            return self.getToken(AtomicParser.RBRACKET, 0)

        def LBRACE(self):
            return self.getToken(AtomicParser.LBRACE, 0)

        def block(self):
            return self.getTypedRuleContext(AtomicParser.BlockContext,0)


        def RBRACE(self):
            return self.getToken(AtomicParser.RBRACE, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_while_stmt

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterWhile_stmt" ):
                listener.enterWhile_stmt(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitWhile_stmt" ):
                listener.exitWhile_stmt(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitWhile_stmt" ):
                return visitor.visitWhile_stmt(self)
            else:
                return visitor.visitChildren(self)




    def while_stmt(self):

        localctx = AtomicParser.While_stmtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_while_stmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 165
            self.match(AtomicParser.WHILE)
            self.state = 166
            self.match(AtomicParser.LBRACKET)
            self.state = 167
            self.cond()
            self.state = 168
            self.match(AtomicParser.RBRACKET)
            self.state = 169
            self.match(AtomicParser.LBRACE)
            self.state = 170
            self.block()
            self.state = 171
            self.match(AtomicParser.RBRACE)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Ifelse_stmtContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def IF(self):
            return self.getToken(AtomicParser.IF, 0)

        def LBRACKET(self):
            return self.getToken(AtomicParser.LBRACKET, 0)

        def cond(self):
            return self.getTypedRuleContext(AtomicParser.CondContext,0)


        def RBRACKET(self):
            return self.getToken(AtomicParser.RBRACKET, 0)

        def LBRACE(self, i:int=None):
            if i is None:
                return self.getTokens(AtomicParser.LBRACE)
            else:
                return self.getToken(AtomicParser.LBRACE, i)

        def block(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AtomicParser.BlockContext)
            else:
                return self.getTypedRuleContext(AtomicParser.BlockContext,i)


        def RBRACE(self, i:int=None):
            if i is None:
                return self.getTokens(AtomicParser.RBRACE)
            else:
                return self.getToken(AtomicParser.RBRACE, i)

        def ELSE(self):
            return self.getToken(AtomicParser.ELSE, 0)

        def EOL(self):
            return self.getToken(AtomicParser.EOL, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_ifelse_stmt

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterIfelse_stmt" ):
                listener.enterIfelse_stmt(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitIfelse_stmt" ):
                listener.exitIfelse_stmt(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitIfelse_stmt" ):
                return visitor.visitIfelse_stmt(self)
            else:
                return visitor.visitChildren(self)




    def ifelse_stmt(self):

        localctx = AtomicParser.Ifelse_stmtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_ifelse_stmt)
        self._la = 0 # Token type
        try:
            self.state = 196
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,12,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 173
                self.match(AtomicParser.IF)
                self.state = 174
                self.match(AtomicParser.LBRACKET)
                self.state = 175
                self.cond()
                self.state = 176
                self.match(AtomicParser.RBRACKET)
                self.state = 177
                self.match(AtomicParser.LBRACE)
                self.state = 178
                self.block()
                self.state = 179
                self.match(AtomicParser.RBRACE)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 181
                self.match(AtomicParser.IF)
                self.state = 182
                self.match(AtomicParser.LBRACKET)
                self.state = 183
                self.cond()
                self.state = 184
                self.match(AtomicParser.RBRACKET)
                self.state = 185
                self.match(AtomicParser.LBRACE)
                self.state = 186
                self.block()
                self.state = 187
                self.match(AtomicParser.RBRACE)
                self.state = 189
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==2:
                    self.state = 188
                    self.match(AtomicParser.EOL)


                self.state = 191
                self.match(AtomicParser.ELSE)
                self.state = 192
                self.match(AtomicParser.LBRACE)
                self.state = 193
                self.block()
                self.state = 194
                self.match(AtomicParser.RBRACE)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Ion_registerContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def IONREGISTER(self):
            return self.getToken(AtomicParser.IONREGISTER, 0)

        def LBRACKET(self):
            return self.getToken(AtomicParser.LBRACKET, 0)

        def INT(self):
            return self.getToken(AtomicParser.INT, 0)

        def RBRACKET(self):
            return self.getToken(AtomicParser.RBRACKET, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_ion_register

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterIon_register" ):
                listener.enterIon_register(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitIon_register" ):
                listener.exitIon_register(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitIon_register" ):
                return visitor.visitIon_register(self)
            else:
                return visitor.visitChildren(self)




    def ion_register(self):

        localctx = AtomicParser.Ion_registerContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_ion_register)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 198
            self.match(AtomicParser.IONREGISTER)
            self.state = 199
            self.match(AtomicParser.LBRACKET)
            self.state = 200
            self.match(AtomicParser.INT)
            self.state = 201
            self.match(AtomicParser.RBRACKET)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Parallel_stmtContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def PARALLEL(self):
            return self.getToken(AtomicParser.PARALLEL, 0)

        def LBRACE(self):
            return self.getToken(AtomicParser.LBRACE, 0)

        def block(self):
            return self.getTypedRuleContext(AtomicParser.BlockContext,0)


        def RBRACE(self):
            return self.getToken(AtomicParser.RBRACE, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_parallel_stmt

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterParallel_stmt" ):
                listener.enterParallel_stmt(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitParallel_stmt" ):
                listener.exitParallel_stmt(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitParallel_stmt" ):
                return visitor.visitParallel_stmt(self)
            else:
                return visitor.visitChildren(self)




    def parallel_stmt(self):

        localctx = AtomicParser.Parallel_stmtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_parallel_stmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 203
            self.match(AtomicParser.PARALLEL)
            self.state = 204
            self.match(AtomicParser.LBRACE)
            self.state = 205
            self.block()
            self.state = 206
            self.match(AtomicParser.RBRACE)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class TargetsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self):
            return self.getTypedRuleContext(AtomicParser.ExprContext,0)


        def getRuleIndex(self):
            return AtomicParser.RULE_targets

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterTargets" ):
                listener.enterTargets(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitTargets" ):
                listener.exitTargets(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitTargets" ):
                return visitor.visitTargets(self)
            else:
                return visitor.visitChildren(self)




    def targets(self):

        localctx = AtomicParser.TargetsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 32, self.RULE_targets)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 208
            self.expr(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Bool_and_opContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def AND(self):
            return self.getToken(AtomicParser.AND, 0)

        def AND2(self):
            return self.getToken(AtomicParser.AND2, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_bool_and_op

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBool_and_op" ):
                listener.enterBool_and_op(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBool_and_op" ):
                listener.exitBool_and_op(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBool_and_op" ):
                return visitor.visitBool_and_op(self)
            else:
                return visitor.visitChildren(self)




    def bool_and_op(self):

        localctx = AtomicParser.Bool_and_opContext(self, self._ctx, self.state)
        self.enterRule(localctx, 34, self.RULE_bool_and_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 210
            _la = self._input.LA(1)
            if not(_la==16 or _la==17):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Bool_or_opContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def OR(self):
            return self.getToken(AtomicParser.OR, 0)

        def OR2(self):
            return self.getToken(AtomicParser.OR2, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_bool_or_op

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBool_or_op" ):
                listener.enterBool_or_op(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBool_or_op" ):
                listener.exitBool_or_op(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBool_or_op" ):
                return visitor.visitBool_or_op(self)
            else:
                return visitor.visitChildren(self)




    def bool_or_op(self):

        localctx = AtomicParser.Bool_or_opContext(self, self._ctx, self.state)
        self.enterRule(localctx, 36, self.RULE_bool_or_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 212
            _la = self._input.LA(1)
            if not(_la==18 or _la==19):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Bool_not_opContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NOT(self):
            return self.getToken(AtomicParser.NOT, 0)

        def NOT2(self):
            return self.getToken(AtomicParser.NOT2, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_bool_not_op

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBool_not_op" ):
                listener.enterBool_not_op(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBool_not_op" ):
                listener.exitBool_not_op(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBool_not_op" ):
                return visitor.visitBool_not_op(self)
            else:
                return visitor.visitChildren(self)




    def bool_not_op(self):

        localctx = AtomicParser.Bool_not_opContext(self, self._ctx, self.state)
        self.enterRule(localctx, 38, self.RULE_bool_not_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 214
            _la = self._input.LA(1)
            if not(_la==20 or _la==21):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Bool_eq_opContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EQ(self):
            return self.getToken(AtomicParser.EQ, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_bool_eq_op

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBool_eq_op" ):
                listener.enterBool_eq_op(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBool_eq_op" ):
                listener.exitBool_eq_op(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBool_eq_op" ):
                return visitor.visitBool_eq_op(self)
            else:
                return visitor.visitChildren(self)




    def bool_eq_op(self):

        localctx = AtomicParser.Bool_eq_opContext(self, self._ctx, self.state)
        self.enterRule(localctx, 40, self.RULE_bool_eq_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 216
            self.match(AtomicParser.EQ)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Bool_not_eq_opContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def NEQ(self):
            return self.getToken(AtomicParser.NEQ, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_bool_not_eq_op

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBool_not_eq_op" ):
                listener.enterBool_not_eq_op(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBool_not_eq_op" ):
                listener.exitBool_not_eq_op(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBool_not_eq_op" ):
                return visitor.visitBool_not_eq_op(self)
            else:
                return visitor.visitChildren(self)




    def bool_not_eq_op(self):

        localctx = AtomicParser.Bool_not_eq_opContext(self, self._ctx, self.state)
        self.enterRule(localctx, 42, self.RULE_bool_not_eq_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 218
            self.match(AtomicParser.NEQ)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Bool_lt_opContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def LT(self):
            return self.getToken(AtomicParser.LT, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_bool_lt_op

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBool_lt_op" ):
                listener.enterBool_lt_op(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBool_lt_op" ):
                listener.exitBool_lt_op(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBool_lt_op" ):
                return visitor.visitBool_lt_op(self)
            else:
                return visitor.visitChildren(self)




    def bool_lt_op(self):

        localctx = AtomicParser.Bool_lt_opContext(self, self._ctx, self.state)
        self.enterRule(localctx, 44, self.RULE_bool_lt_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 220
            self.match(AtomicParser.LT)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Bool_lte_opContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def LTE(self):
            return self.getToken(AtomicParser.LTE, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_bool_lte_op

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBool_lte_op" ):
                listener.enterBool_lte_op(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBool_lte_op" ):
                listener.exitBool_lte_op(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBool_lte_op" ):
                return visitor.visitBool_lte_op(self)
            else:
                return visitor.visitChildren(self)




    def bool_lte_op(self):

        localctx = AtomicParser.Bool_lte_opContext(self, self._ctx, self.state)
        self.enterRule(localctx, 46, self.RULE_bool_lte_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 222
            self.match(AtomicParser.LTE)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Bool_gt_opContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def GT(self):
            return self.getToken(AtomicParser.GT, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_bool_gt_op

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBool_gt_op" ):
                listener.enterBool_gt_op(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBool_gt_op" ):
                listener.exitBool_gt_op(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBool_gt_op" ):
                return visitor.visitBool_gt_op(self)
            else:
                return visitor.visitChildren(self)




    def bool_gt_op(self):

        localctx = AtomicParser.Bool_gt_opContext(self, self._ctx, self.state)
        self.enterRule(localctx, 48, self.RULE_bool_gt_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 224
            self.match(AtomicParser.GT)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Bool_gte_opContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def GTE(self):
            return self.getToken(AtomicParser.GTE, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_bool_gte_op

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBool_gte_op" ):
                listener.enterBool_gte_op(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBool_gte_op" ):
                listener.exitBool_gte_op(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBool_gte_op" ):
                return visitor.visitBool_gte_op(self)
            else:
                return visitor.visitChildren(self)




    def bool_gte_op(self):

        localctx = AtomicParser.Bool_gte_opContext(self, self._ctx, self.state)
        self.enterRule(localctx, 50, self.RULE_bool_gte_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 226
            self.match(AtomicParser.GTE)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Bool_literalContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def TRUE(self):
            return self.getToken(AtomicParser.TRUE, 0)

        def FALSE(self):
            return self.getToken(AtomicParser.FALSE, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_bool_literal

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBool_literal" ):
                listener.enterBool_literal(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBool_literal" ):
                listener.exitBool_literal(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBool_literal" ):
                return visitor.visitBool_literal(self)
            else:
                return visitor.visitChildren(self)




    def bool_literal(self):

        localctx = AtomicParser.Bool_literalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 52, self.RULE_bool_literal)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 228
            _la = self._input.LA(1)
            if not(_la==22 or _la==23):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ComparatorsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def bool_eq_op(self):
            return self.getTypedRuleContext(AtomicParser.Bool_eq_opContext,0)


        def bool_not_eq_op(self):
            return self.getTypedRuleContext(AtomicParser.Bool_not_eq_opContext,0)


        def bool_lt_op(self):
            return self.getTypedRuleContext(AtomicParser.Bool_lt_opContext,0)


        def bool_lte_op(self):
            return self.getTypedRuleContext(AtomicParser.Bool_lte_opContext,0)


        def bool_gt_op(self):
            return self.getTypedRuleContext(AtomicParser.Bool_gt_opContext,0)


        def bool_gte_op(self):
            return self.getTypedRuleContext(AtomicParser.Bool_gte_opContext,0)


        def getRuleIndex(self):
            return AtomicParser.RULE_comparators

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterComparators" ):
                listener.enterComparators(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitComparators" ):
                listener.exitComparators(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitComparators" ):
                return visitor.visitComparators(self)
            else:
                return visitor.visitChildren(self)




    def comparators(self):

        localctx = AtomicParser.ComparatorsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 54, self.RULE_comparators)
        try:
            self.state = 236
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [39]:
                self.enterOuterAlt(localctx, 1)
                self.state = 230
                self.bool_eq_op()
                pass
            elif token in [40]:
                self.enterOuterAlt(localctx, 2)
                self.state = 231
                self.bool_not_eq_op()
                pass
            elif token in [41]:
                self.enterOuterAlt(localctx, 3)
                self.state = 232
                self.bool_lt_op()
                pass
            elif token in [42]:
                self.enterOuterAlt(localctx, 4)
                self.state = 233
                self.bool_lte_op()
                pass
            elif token in [43]:
                self.enterOuterAlt(localctx, 5)
                self.state = 234
                self.bool_gt_op()
                pass
            elif token in [44]:
                self.enterOuterAlt(localctx, 6)
                self.state = 235
                self.bool_gte_op()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Math_terminalContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def INT(self):
            return self.getToken(AtomicParser.INT, 0)

        def FLOAT(self):
            return self.getToken(AtomicParser.FLOAT, 0)

        def MATH_VAR(self):
            return self.getToken(AtomicParser.MATH_VAR, 0)

        def IMAG(self):
            return self.getToken(AtomicParser.IMAG, 0)

        def access(self):
            return self.getTypedRuleContext(AtomicParser.AccessContext,0)


        def pexpr(self):
            return self.getTypedRuleContext(AtomicParser.PexprContext,0)


        def fexpr(self):
            return self.getTypedRuleContext(AtomicParser.FexprContext,0)


        def atomic_list(self):
            return self.getTypedRuleContext(AtomicParser.Atomic_listContext,0)


        def getRuleIndex(self):
            return AtomicParser.RULE_math_terminal

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMath_terminal" ):
                listener.enterMath_terminal(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMath_terminal" ):
                listener.exitMath_terminal(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMath_terminal" ):
                return visitor.visitMath_terminal(self)
            else:
                return visitor.visitChildren(self)




    def math_terminal(self):

        localctx = AtomicParser.Math_terminalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 56, self.RULE_math_terminal)
        try:
            self.state = 246
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [45]:
                self.enterOuterAlt(localctx, 1)
                self.state = 238
                self.match(AtomicParser.INT)
                pass
            elif token in [46]:
                self.enterOuterAlt(localctx, 2)
                self.state = 239
                self.match(AtomicParser.FLOAT)
                pass
            elif token in [47]:
                self.enterOuterAlt(localctx, 3)
                self.state = 240
                self.match(AtomicParser.MATH_VAR)
                pass
            elif token in [48]:
                self.enterOuterAlt(localctx, 4)
                self.state = 241
                self.match(AtomicParser.IMAG)
                pass
            elif token in [69]:
                self.enterOuterAlt(localctx, 5)
                self.state = 242
                self.access()
                pass
            elif token in [27]:
                self.enterOuterAlt(localctx, 6)
                self.state = 243
                self.pexpr()
                pass
            elif token in [14, 15, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68]:
                self.enterOuterAlt(localctx, 7)
                self.state = 244
                self.fexpr()
                pass
            elif token in [29]:
                self.enterOuterAlt(localctx, 8)
                self.state = 245
                self.atomic_list()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Func_namesContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ABS(self):
            return self.getToken(AtomicParser.ABS, 0)

        def SIN(self):
            return self.getToken(AtomicParser.SIN, 0)

        def COS(self):
            return self.getToken(AtomicParser.COS, 0)

        def TAN(self):
            return self.getToken(AtomicParser.TAN, 0)

        def EXP(self):
            return self.getToken(AtomicParser.EXP, 0)

        def LOG(self):
            return self.getToken(AtomicParser.LOG, 0)

        def SINH(self):
            return self.getToken(AtomicParser.SINH, 0)

        def COSH(self):
            return self.getToken(AtomicParser.COSH, 0)

        def TANH(self):
            return self.getToken(AtomicParser.TANH, 0)

        def ATAN(self):
            return self.getToken(AtomicParser.ATAN, 0)

        def ACOS(self):
            return self.getToken(AtomicParser.ACOS, 0)

        def ASIN(self):
            return self.getToken(AtomicParser.ASIN, 0)

        def ATANH(self):
            return self.getToken(AtomicParser.ATANH, 0)

        def ASINH(self):
            return self.getToken(AtomicParser.ASINH, 0)

        def ACOSH(self):
            return self.getToken(AtomicParser.ACOSH, 0)

        def ATAN2(self):
            return self.getToken(AtomicParser.ATAN2, 0)

        def HEAVISIDE(self):
            return self.getToken(AtomicParser.HEAVISIDE, 0)

        def CONJ(self):
            return self.getToken(AtomicParser.CONJ, 0)

        def REAL(self):
            return self.getToken(AtomicParser.REAL, 0)

        def IMAG_FN(self):
            return self.getToken(AtomicParser.IMAG_FN, 0)

        def BEAM(self):
            return self.getToken(AtomicParser.BEAM, 0)

        def PULSE(self):
            return self.getToken(AtomicParser.PULSE, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_func_names

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFunc_names" ):
                listener.enterFunc_names(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFunc_names" ):
                listener.exitFunc_names(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFunc_names" ):
                return visitor.visitFunc_names(self)
            else:
                return visitor.visitChildren(self)




    def func_names(self):

        localctx = AtomicParser.Func_namesContext(self, self._ctx, self.state)
        self.enterRule(localctx, 58, self.RULE_func_names)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 248
            _la = self._input.LA(1)
            if not(((((_la - 14)) & ~0x3f) == 0 and ((1 << (_la - 14)) & 36028762659225603) != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PexprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def LBRACKET(self):
            return self.getToken(AtomicParser.LBRACKET, 0)

        def aexpr(self):
            return self.getTypedRuleContext(AtomicParser.AexprContext,0)


        def RBRACKET(self):
            return self.getToken(AtomicParser.RBRACKET, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_pexpr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPexpr" ):
                listener.enterPexpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPexpr" ):
                listener.exitPexpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPexpr" ):
                return visitor.visitPexpr(self)
            else:
                return visitor.visitChildren(self)




    def pexpr(self):

        localctx = AtomicParser.PexprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 60, self.RULE_pexpr)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 250
            self.match(AtomicParser.LBRACKET)
            self.state = 251
            self.aexpr(0)
            self.state = 252
            self.match(AtomicParser.RBRACKET)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class FexprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def func_names(self):
            return self.getTypedRuleContext(AtomicParser.Func_namesContext,0)


        def LBRACKET(self):
            return self.getToken(AtomicParser.LBRACKET, 0)

        def RBRACKET(self):
            return self.getToken(AtomicParser.RBRACKET, 0)

        def aexpr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AtomicParser.AexprContext)
            else:
                return self.getTypedRuleContext(AtomicParser.AexprContext,i)


        def COMMA(self, i:int=None):
            if i is None:
                return self.getTokens(AtomicParser.COMMA)
            else:
                return self.getToken(AtomicParser.COMMA, i)

        def getRuleIndex(self):
            return AtomicParser.RULE_fexpr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFexpr" ):
                listener.enterFexpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFexpr" ):
                listener.exitFexpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFexpr" ):
                return visitor.visitFexpr(self)
            else:
                return visitor.visitChildren(self)




    def fexpr(self):

        localctx = AtomicParser.FexprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 62, self.RULE_fexpr)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 254
            self.func_names()
            self.state = 255
            self.match(AtomicParser.LBRACKET)
            self.state = 264
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if ((((_la - 13)) & ~0x3f) == 0 and ((1 << (_la - 13)) & 144115183793554951) != 0):
                self.state = 256
                self.aexpr(0)
                self.state = 261
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la==26:
                    self.state = 257
                    self.match(AtomicParser.COMMA)
                    self.state = 258
                    self.aexpr(0)
                    self.state = 263
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)



            self.state = 266
            self.match(AtomicParser.RBRACKET)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AexprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def mexpr(self):
            return self.getTypedRuleContext(AtomicParser.MexprContext,0)


        def aexpr(self):
            return self.getTypedRuleContext(AtomicParser.AexprContext,0)


        def PLUS(self):
            return self.getToken(AtomicParser.PLUS, 0)

        def MINUS(self):
            return self.getToken(AtomicParser.MINUS, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_aexpr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAexpr" ):
                listener.enterAexpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAexpr" ):
                listener.exitAexpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAexpr" ):
                return visitor.visitAexpr(self)
            else:
                return visitor.visitChildren(self)



    def aexpr(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = AtomicParser.AexprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 64
        self.enterRecursionRule(localctx, 64, self.RULE_aexpr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 269
            self.mexpr(0)
            self._ctx.stop = self._input.LT(-1)
            self.state = 276
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,17,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = AtomicParser.AexprContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_aexpr)
                    self.state = 271
                    if not self.precpred(self._ctx, 1):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 1)")
                    self.state = 272
                    _la = self._input.LA(1)
                    if not(_la==35 or _la==36):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 273
                    self.mexpr(0) 
                self.state = 278
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,17,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class MexprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def uexpr(self):
            return self.getTypedRuleContext(AtomicParser.UexprContext,0)


        def mexpr(self):
            return self.getTypedRuleContext(AtomicParser.MexprContext,0)


        def MULT(self):
            return self.getToken(AtomicParser.MULT, 0)

        def DIV(self):
            return self.getToken(AtomicParser.DIV, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_mexpr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMexpr" ):
                listener.enterMexpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMexpr" ):
                listener.exitMexpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMexpr" ):
                return visitor.visitMexpr(self)
            else:
                return visitor.visitChildren(self)



    def mexpr(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = AtomicParser.MexprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 66
        self.enterRecursionRule(localctx, 66, self.RULE_mexpr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 280
            self.uexpr()
            self._ctx.stop = self._input.LT(-1)
            self.state = 287
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,18,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = AtomicParser.MexprContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_mexpr)
                    self.state = 282
                    if not self.precpred(self._ctx, 1):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 1)")
                    self.state = 283
                    _la = self._input.LA(1)
                    if not(_la==33 or _la==34):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 284
                    self.uexpr() 
                self.state = 289
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,18,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class UexprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def eexpr(self):
            return self.getTypedRuleContext(AtomicParser.EexprContext,0)


        def PLUS(self):
            return self.getToken(AtomicParser.PLUS, 0)

        def MINUS(self):
            return self.getToken(AtomicParser.MINUS, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_uexpr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterUexpr" ):
                listener.enterUexpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitUexpr" ):
                listener.exitUexpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitUexpr" ):
                return visitor.visitUexpr(self)
            else:
                return visitor.visitChildren(self)




    def uexpr(self):

        localctx = AtomicParser.UexprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 68, self.RULE_uexpr)
        self._la = 0 # Token type
        try:
            self.state = 293
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [13, 14, 15, 22, 23, 27, 29, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69]:
                self.enterOuterAlt(localctx, 1)
                self.state = 290
                self.eexpr()
                pass
            elif token in [35, 36]:
                self.enterOuterAlt(localctx, 2)
                self.state = 291
                _la = self._input.LA(1)
                if not(_la==35 or _la==36):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 292
                self.eexpr()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class EexprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def terminal(self):
            return self.getTypedRuleContext(AtomicParser.TerminalContext,0)


        def POWER(self):
            return self.getToken(AtomicParser.POWER, 0)

        def uexpr(self):
            return self.getTypedRuleContext(AtomicParser.UexprContext,0)


        def getRuleIndex(self):
            return AtomicParser.RULE_eexpr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterEexpr" ):
                listener.enterEexpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitEexpr" ):
                listener.exitEexpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitEexpr" ):
                return visitor.visitEexpr(self)
            else:
                return visitor.visitChildren(self)




    def eexpr(self):

        localctx = AtomicParser.EexprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 70, self.RULE_eexpr)
        try:
            self.state = 300
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,20,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 295
                self.terminal()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 296
                self.terminal()
                self.state = 297
                self.match(AtomicParser.POWER)
                self.state = 298
                self.uexpr()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx



    def sempred(self, localctx:RuleContext, ruleIndex:int, predIndex:int):
        if self._predicates == None:
            self._predicates = dict()
        self._predicates[4] = self.expr_sempred
        self._predicates[32] = self.aexpr_sempred
        self._predicates[33] = self.mexpr_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception("No predicate with index:" + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def expr_sempred(self, localctx:ExprContext, predIndex:int):
            if predIndex == 0:
                return self.precpred(self._ctx, 7)
         

    def aexpr_sempred(self, localctx:AexprContext, predIndex:int):
            if predIndex == 1:
                return self.precpred(self._ctx, 1)
         

    def mexpr_sempred(self, localctx:MexprContext, predIndex:int):
            if predIndex == 2:
                return self.precpred(self._ctx, 1)
         




