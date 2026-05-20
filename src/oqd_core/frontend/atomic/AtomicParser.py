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
        4,1,70,312,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,20,
        7,20,2,21,7,21,2,22,7,22,2,23,7,23,2,24,7,24,2,25,7,25,2,26,7,26,
        2,27,7,27,2,28,7,28,2,29,7,29,2,30,7,30,2,31,7,31,2,32,7,32,2,33,
        7,33,2,34,7,34,2,35,7,35,2,36,7,36,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,3,1,86,8,1,1,2,1,2,1,2,1,2,5,2,92,8,2,10,2,12,2,95,
        9,2,1,2,3,2,98,8,2,1,3,1,3,1,3,3,3,103,8,3,1,4,1,4,1,4,1,4,1,4,4,
        4,110,8,4,11,4,12,4,111,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,
        1,4,3,4,125,8,4,1,4,1,4,1,4,3,4,130,8,4,1,4,1,4,5,4,134,8,4,10,4,
        12,4,137,9,4,1,5,1,5,1,6,1,6,3,6,143,8,6,1,6,1,6,5,6,147,8,6,10,
        6,12,6,150,9,6,1,6,1,6,1,7,1,7,1,7,1,7,1,8,1,8,1,9,1,9,1,9,1,9,1,
        9,1,10,1,10,1,11,1,11,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,
        13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,
        13,1,13,1,13,3,13,193,8,13,1,13,1,13,1,13,1,13,1,13,3,13,200,8,13,
        1,14,1,14,1,14,1,14,1,14,1,15,1,15,1,15,1,15,1,15,1,16,1,16,1,16,
        1,16,1,16,1,17,1,17,1,18,1,18,1,19,1,19,1,20,1,20,1,21,1,21,1,22,
        1,22,1,23,1,23,1,24,1,24,1,25,1,25,1,26,1,26,1,27,1,27,1,28,1,28,
        1,28,1,28,1,28,1,28,3,28,245,8,28,1,29,1,29,1,29,1,29,1,29,1,29,
        1,29,1,29,1,29,3,29,256,8,29,1,30,1,30,1,31,1,31,1,31,1,31,1,32,
        1,32,1,32,1,32,1,32,5,32,269,8,32,10,32,12,32,272,9,32,3,32,274,
        8,32,1,32,1,32,1,33,1,33,1,33,1,33,1,33,1,33,5,33,284,8,33,10,33,
        12,33,287,9,33,1,34,1,34,1,34,1,34,1,34,1,34,5,34,295,8,34,10,34,
        12,34,298,9,34,1,35,1,35,1,35,3,35,303,8,35,1,36,1,36,1,36,1,36,
        1,36,3,36,310,8,36,1,36,0,3,8,66,68,37,0,2,4,6,8,10,12,14,16,18,
        20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,
        64,66,68,70,72,0,7,1,0,17,18,1,0,19,20,1,0,21,22,1,0,23,24,2,0,15,
        16,50,69,1,0,36,37,1,0,34,35,318,0,74,1,0,0,0,2,85,1,0,0,0,4,93,
        1,0,0,0,6,102,1,0,0,0,8,124,1,0,0,0,10,138,1,0,0,0,12,140,1,0,0,
        0,14,153,1,0,0,0,16,157,1,0,0,0,18,159,1,0,0,0,20,164,1,0,0,0,22,
        166,1,0,0,0,24,168,1,0,0,0,26,199,1,0,0,0,28,201,1,0,0,0,30,206,
        1,0,0,0,32,211,1,0,0,0,34,216,1,0,0,0,36,218,1,0,0,0,38,220,1,0,
        0,0,40,222,1,0,0,0,42,224,1,0,0,0,44,226,1,0,0,0,46,228,1,0,0,0,
        48,230,1,0,0,0,50,232,1,0,0,0,52,234,1,0,0,0,54,236,1,0,0,0,56,244,
        1,0,0,0,58,255,1,0,0,0,60,257,1,0,0,0,62,259,1,0,0,0,64,263,1,0,
        0,0,66,277,1,0,0,0,68,288,1,0,0,0,70,302,1,0,0,0,72,309,1,0,0,0,
        74,75,3,4,2,0,75,76,5,0,0,1,76,1,1,0,0,0,77,86,3,14,7,0,78,86,3,
        30,15,0,79,86,3,32,16,0,80,86,3,24,12,0,81,86,3,26,13,0,82,86,3,
        20,10,0,83,86,3,22,11,0,84,86,3,8,4,0,85,77,1,0,0,0,85,78,1,0,0,
        0,85,79,1,0,0,0,85,80,1,0,0,0,85,81,1,0,0,0,85,82,1,0,0,0,85,83,
        1,0,0,0,85,84,1,0,0,0,86,3,1,0,0,0,87,88,3,2,1,0,88,89,5,2,0,0,89,
        92,1,0,0,0,90,92,5,2,0,0,91,87,1,0,0,0,91,90,1,0,0,0,92,95,1,0,0,
        0,93,91,1,0,0,0,93,94,1,0,0,0,94,97,1,0,0,0,95,93,1,0,0,0,96,98,
        3,2,1,0,97,96,1,0,0,0,97,98,1,0,0,0,98,5,1,0,0,0,99,103,3,28,14,
        0,100,103,3,58,29,0,101,103,3,54,27,0,102,99,1,0,0,0,102,100,1,0,
        0,0,102,101,1,0,0,0,103,7,1,0,0,0,104,105,6,4,-1,0,105,109,3,66,
        33,0,106,107,3,56,28,0,107,108,3,66,33,0,108,110,1,0,0,0,109,106,
        1,0,0,0,110,111,1,0,0,0,111,109,1,0,0,0,111,112,1,0,0,0,112,125,
        1,0,0,0,113,114,3,40,20,0,114,115,3,8,4,6,115,125,1,0,0,0,116,117,
        5,28,0,0,117,118,3,8,4,0,118,119,5,29,0,0,119,125,1,0,0,0,120,125,
        3,18,9,0,121,125,3,12,6,0,122,125,3,6,3,0,123,125,3,66,33,0,124,
        104,1,0,0,0,124,113,1,0,0,0,124,116,1,0,0,0,124,120,1,0,0,0,124,
        121,1,0,0,0,124,122,1,0,0,0,124,123,1,0,0,0,125,135,1,0,0,0,126,
        129,10,7,0,0,127,130,3,36,18,0,128,130,3,38,19,0,129,127,1,0,0,0,
        129,128,1,0,0,0,130,131,1,0,0,0,131,132,3,8,4,8,132,134,1,0,0,0,
        133,126,1,0,0,0,134,137,1,0,0,0,135,133,1,0,0,0,135,136,1,0,0,0,
        136,9,1,0,0,0,137,135,1,0,0,0,138,139,3,8,4,0,139,11,1,0,0,0,140,
        142,5,30,0,0,141,143,3,8,4,0,142,141,1,0,0,0,142,143,1,0,0,0,143,
        148,1,0,0,0,144,145,5,27,0,0,145,147,3,8,4,0,146,144,1,0,0,0,147,
        150,1,0,0,0,148,146,1,0,0,0,148,149,1,0,0,0,149,151,1,0,0,0,150,
        148,1,0,0,0,151,152,5,31,0,0,152,13,1,0,0,0,153,154,5,70,0,0,154,
        155,5,39,0,0,155,156,3,8,4,0,156,15,1,0,0,0,157,158,5,70,0,0,158,
        17,1,0,0,0,159,160,3,16,8,0,160,161,5,30,0,0,161,162,5,46,0,0,162,
        163,5,31,0,0,163,19,1,0,0,0,164,165,5,12,0,0,165,21,1,0,0,0,166,
        167,5,13,0,0,167,23,1,0,0,0,168,169,5,9,0,0,169,170,5,28,0,0,170,
        171,3,10,5,0,171,172,5,29,0,0,172,173,5,32,0,0,173,174,3,4,2,0,174,
        175,5,33,0,0,175,25,1,0,0,0,176,177,5,7,0,0,177,178,5,28,0,0,178,
        179,3,10,5,0,179,180,5,29,0,0,180,181,5,32,0,0,181,182,3,4,2,0,182,
        183,5,33,0,0,183,200,1,0,0,0,184,185,5,7,0,0,185,186,5,28,0,0,186,
        187,3,10,5,0,187,188,5,29,0,0,188,189,5,32,0,0,189,190,3,4,2,0,190,
        192,5,33,0,0,191,193,5,2,0,0,192,191,1,0,0,0,192,193,1,0,0,0,193,
        194,1,0,0,0,194,195,5,8,0,0,195,196,5,32,0,0,196,197,3,4,2,0,197,
        198,5,33,0,0,198,200,1,0,0,0,199,176,1,0,0,0,199,184,1,0,0,0,200,
        27,1,0,0,0,201,202,5,14,0,0,202,203,5,28,0,0,203,204,5,46,0,0,204,
        205,5,29,0,0,205,29,1,0,0,0,206,207,5,5,0,0,207,208,5,32,0,0,208,
        209,3,4,2,0,209,210,5,33,0,0,210,31,1,0,0,0,211,212,5,6,0,0,212,
        213,5,32,0,0,213,214,3,4,2,0,214,215,5,33,0,0,215,33,1,0,0,0,216,
        217,3,8,4,0,217,35,1,0,0,0,218,219,7,0,0,0,219,37,1,0,0,0,220,221,
        7,1,0,0,221,39,1,0,0,0,222,223,7,2,0,0,223,41,1,0,0,0,224,225,5,
        40,0,0,225,43,1,0,0,0,226,227,5,41,0,0,227,45,1,0,0,0,228,229,5,
        42,0,0,229,47,1,0,0,0,230,231,5,43,0,0,231,49,1,0,0,0,232,233,5,
        44,0,0,233,51,1,0,0,0,234,235,5,45,0,0,235,53,1,0,0,0,236,237,7,
        3,0,0,237,55,1,0,0,0,238,245,3,42,21,0,239,245,3,44,22,0,240,245,
        3,46,23,0,241,245,3,48,24,0,242,245,3,50,25,0,243,245,3,52,26,0,
        244,238,1,0,0,0,244,239,1,0,0,0,244,240,1,0,0,0,244,241,1,0,0,0,
        244,242,1,0,0,0,244,243,1,0,0,0,245,57,1,0,0,0,246,256,5,46,0,0,
        247,256,5,47,0,0,248,256,5,48,0,0,249,256,5,49,0,0,250,256,3,16,
        8,0,251,256,3,62,31,0,252,256,3,64,32,0,253,256,3,12,6,0,254,256,
        3,18,9,0,255,246,1,0,0,0,255,247,1,0,0,0,255,248,1,0,0,0,255,249,
        1,0,0,0,255,250,1,0,0,0,255,251,1,0,0,0,255,252,1,0,0,0,255,253,
        1,0,0,0,255,254,1,0,0,0,256,59,1,0,0,0,257,258,7,4,0,0,258,61,1,
        0,0,0,259,260,5,28,0,0,260,261,3,66,33,0,261,262,5,29,0,0,262,63,
        1,0,0,0,263,264,3,60,30,0,264,273,5,28,0,0,265,270,3,66,33,0,266,
        267,5,27,0,0,267,269,3,66,33,0,268,266,1,0,0,0,269,272,1,0,0,0,270,
        268,1,0,0,0,270,271,1,0,0,0,271,274,1,0,0,0,272,270,1,0,0,0,273,
        265,1,0,0,0,273,274,1,0,0,0,274,275,1,0,0,0,275,276,5,29,0,0,276,
        65,1,0,0,0,277,278,6,33,-1,0,278,279,3,68,34,0,279,285,1,0,0,0,280,
        281,10,1,0,0,281,282,7,5,0,0,282,284,3,68,34,0,283,280,1,0,0,0,284,
        287,1,0,0,0,285,283,1,0,0,0,285,286,1,0,0,0,286,67,1,0,0,0,287,285,
        1,0,0,0,288,289,6,34,-1,0,289,290,3,70,35,0,290,296,1,0,0,0,291,
        292,10,1,0,0,292,293,7,6,0,0,293,295,3,70,35,0,294,291,1,0,0,0,295,
        298,1,0,0,0,296,294,1,0,0,0,296,297,1,0,0,0,297,69,1,0,0,0,298,296,
        1,0,0,0,299,303,3,72,36,0,300,301,7,5,0,0,301,303,3,72,36,0,302,
        299,1,0,0,0,302,300,1,0,0,0,303,71,1,0,0,0,304,310,3,6,3,0,305,306,
        3,6,3,0,306,307,5,38,0,0,307,308,3,70,35,0,308,310,1,0,0,0,309,304,
        1,0,0,0,309,305,1,0,0,0,310,73,1,0,0,0,21,85,91,93,97,102,111,124,
        129,135,142,148,192,199,244,255,270,273,285,296,302,309
    ]

class AtomicParser ( Parser ):

    grammarFileName = "AtomicParser.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                     "<INVALID>", "'parallel'", "'serial'", "'if'", "'else'", 
                     "'while'", "'with'", "'for'", "'break'", "'continue'", 
                     "'ionreg'", "'beam'", "'pulse'", "'and'", "'&&'", "'or'", 
                     "'||'", "'not'", "'!'", "'true'", "'false'", "':'", 
                     "';'", "','", "'('", "')'", "'['", "']'", "'{'", "'}'", 
                     "'*'", "'/'", "'+'", "'-'", "'^'", "'='", "'=='", "'!='", 
                     "'<'", "'<='", "'>'", "'>='", "<INVALID>", "<INVALID>", 
                     "<INVALID>", "'1j'", "'abs'", "'sin'", "'cos'", "'tan'", 
                     "'exp'", "'log'", "'sinh'", "'cosh'", "'tanh'", "'atan'", 
                     "'acos'", "'asin'", "'atanh'", "'asinh'", "'acosh'", 
                     "'heaviside'", "'conj'", "'real'", "'imag'", "'atan2'" ]

    symbolicNames = [ "<INVALID>", "WHITESPACE", "EOL", "NEWLINE", "COMMENT", 
                      "PARALLEL", "SERIAL", "IF", "ELSE", "WHILE", "WITH", 
                      "FOR", "BREAK", "CONTINUE", "IONREGISTER", "BEAM", 
                      "PULSE", "AND", "AND2", "OR", "OR2", "NOT", "NOT2", 
                      "TRUE", "FALSE", "COLON", "SEMICOLON", "COMMA", "LBRACKET", 
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
    RULE_serial_stmt = 16
    RULE_targets = 17
    RULE_bool_and_op = 18
    RULE_bool_or_op = 19
    RULE_bool_not_op = 20
    RULE_bool_eq_op = 21
    RULE_bool_not_eq_op = 22
    RULE_bool_lt_op = 23
    RULE_bool_lte_op = 24
    RULE_bool_gt_op = 25
    RULE_bool_gte_op = 26
    RULE_bool_literal = 27
    RULE_comparators = 28
    RULE_math_terminal = 29
    RULE_func_names = 30
    RULE_pexpr = 31
    RULE_fexpr = 32
    RULE_aexpr = 33
    RULE_mexpr = 34
    RULE_uexpr = 35
    RULE_eexpr = 36

    ruleNames =  [ "program", "statement", "block", "terminal", "expr", 
                   "cond", "atomic_list", "declaration", "access", "atomic_list_extract", 
                   "break_stmt", "continue_stmt", "while_stmt", "ifelse_stmt", 
                   "ion_register", "parallel_stmt", "serial_stmt", "targets", 
                   "bool_and_op", "bool_or_op", "bool_not_op", "bool_eq_op", 
                   "bool_not_eq_op", "bool_lt_op", "bool_lte_op", "bool_gt_op", 
                   "bool_gte_op", "bool_literal", "comparators", "math_terminal", 
                   "func_names", "pexpr", "fexpr", "aexpr", "mexpr", "uexpr", 
                   "eexpr" ]

    EOF = Token.EOF
    WHITESPACE=1
    EOL=2
    NEWLINE=3
    COMMENT=4
    PARALLEL=5
    SERIAL=6
    IF=7
    ELSE=8
    WHILE=9
    WITH=10
    FOR=11
    BREAK=12
    CONTINUE=13
    IONREGISTER=14
    BEAM=15
    PULSE=16
    AND=17
    AND2=18
    OR=19
    OR2=20
    NOT=21
    NOT2=22
    TRUE=23
    FALSE=24
    COLON=25
    SEMICOLON=26
    COMMA=27
    LBRACKET=28
    RBRACKET=29
    SQUARELBRACKET=30
    SQUARERBRACKET=31
    LBRACE=32
    RBRACE=33
    MULT=34
    DIV=35
    PLUS=36
    MINUS=37
    POWER=38
    ASSIGN=39
    EQ=40
    NEQ=41
    LT=42
    LTE=43
    GT=44
    GTE=45
    INT=46
    FLOAT=47
    MATH_VAR=48
    IMAG=49
    ABS=50
    SIN=51
    COS=52
    TAN=53
    EXP=54
    LOG=55
    SINH=56
    COSH=57
    TANH=58
    ATAN=59
    ACOS=60
    ASIN=61
    ATANH=62
    ASINH=63
    ACOSH=64
    HEAVISIDE=65
    CONJ=66
    REAL=67
    IMAG_FN=68
    ATAN2=69
    ID=70

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
            self.state = 74
            self.block()
            self.state = 75
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


        def serial_stmt(self):
            return self.getTypedRuleContext(AtomicParser.Serial_stmtContext,0)


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
            self.state = 85
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,0,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 77
                self.declaration()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 78
                self.parallel_stmt()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 79
                self.serial_stmt()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 80
                self.while_stmt()
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 81
                self.ifelse_stmt()
                pass

            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 82
                self.break_stmt()
                pass

            elif la_ == 7:
                self.enterOuterAlt(localctx, 7)
                self.state = 83
                self.continue_stmt()
                pass

            elif la_ == 8:
                self.enterOuterAlt(localctx, 8)
                self.state = 84
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
            self.state = 93
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,2,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    self.state = 91
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [5, 6, 7, 9, 12, 13, 14, 15, 16, 21, 22, 23, 24, 28, 30, 36, 37, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]:
                        self.state = 87
                        self.statement()
                        self.state = 88
                        self.match(AtomicParser.EOL)
                        pass
                    elif token in [2]:
                        self.state = 90
                        self.match(AtomicParser.EOL)
                        pass
                    else:
                        raise NoViableAltException(self)
             
                self.state = 95
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,2,self._ctx)

            self.state = 97
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if (((_la) & ~0x3f) == 0 and ((1 << _la) & -70161211985184) != 0) or ((((_la - 64)) & ~0x3f) == 0 and ((1 << (_la - 64)) & 127) != 0):
                self.state = 96
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
            self.state = 102
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [14]:
                self.enterOuterAlt(localctx, 1)
                self.state = 99
                self.ion_register()
                pass
            elif token in [15, 16, 28, 30, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]:
                self.enterOuterAlt(localctx, 2)
                self.state = 100
                self.math_terminal()
                pass
            elif token in [23, 24]:
                self.enterOuterAlt(localctx, 3)
                self.state = 101
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
            self.state = 124
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,6,self._ctx)
            if la_ == 1:
                self.state = 105
                self.aexpr(0)
                self.state = 109 
                self._errHandler.sync(self)
                _alt = 1
                while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 106
                        self.comparators()
                        self.state = 107
                        self.aexpr(0)

                    else:
                        raise NoViableAltException(self)
                    self.state = 111 
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input,5,self._ctx)

                pass

            elif la_ == 2:
                self.state = 113
                self.bool_not_op()
                self.state = 114
                self.expr(6)
                pass

            elif la_ == 3:
                self.state = 116
                self.match(AtomicParser.LBRACKET)
                self.state = 117
                self.expr(0)
                self.state = 118
                self.match(AtomicParser.RBRACKET)
                pass

            elif la_ == 4:
                self.state = 120
                self.atomic_list_extract()
                pass

            elif la_ == 5:
                self.state = 121
                self.atomic_list()
                pass

            elif la_ == 6:
                self.state = 122
                self.terminal()
                pass

            elif la_ == 7:
                self.state = 123
                self.aexpr(0)
                pass


            self._ctx.stop = self._input.LT(-1)
            self.state = 135
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,8,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = AtomicParser.ExprContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                    self.state = 126
                    if not self.precpred(self._ctx, 7):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 7)")
                    self.state = 129
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [17, 18]:
                        self.state = 127
                        self.bool_and_op()
                        pass
                    elif token in [19, 20]:
                        self.state = 128
                        self.bool_or_op()
                        pass
                    else:
                        raise NoViableAltException(self)

                    self.state = 131
                    self.expr(8) 
                self.state = 137
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
            self.state = 138
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
            self.state = 140
            self.match(AtomicParser.SQUARELBRACKET)
            self.state = 142
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if ((((_la - 14)) & ~0x3f) == 0 and ((1 << (_la - 14)) & 144115183793555335) != 0):
                self.state = 141
                self.expr(0)


            self.state = 148
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==27:
                self.state = 144
                self.match(AtomicParser.COMMA)
                self.state = 145
                self.expr(0)
                self.state = 150
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 151
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
            self.state = 153
            self.match(AtomicParser.ID)
            self.state = 154
            self.match(AtomicParser.ASSIGN)
            self.state = 155
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
            self.state = 157
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
            self.state = 159
            self.access()
            self.state = 160
            self.match(AtomicParser.SQUARELBRACKET)
            self.state = 161
            self.match(AtomicParser.INT)
            self.state = 162
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
            self.state = 164
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
            self.state = 166
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
            self.state = 168
            self.match(AtomicParser.WHILE)
            self.state = 169
            self.match(AtomicParser.LBRACKET)
            self.state = 170
            self.cond()
            self.state = 171
            self.match(AtomicParser.RBRACKET)
            self.state = 172
            self.match(AtomicParser.LBRACE)
            self.state = 173
            self.block()
            self.state = 174
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
            self.state = 199
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,12,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 176
                self.match(AtomicParser.IF)
                self.state = 177
                self.match(AtomicParser.LBRACKET)
                self.state = 178
                self.cond()
                self.state = 179
                self.match(AtomicParser.RBRACKET)
                self.state = 180
                self.match(AtomicParser.LBRACE)
                self.state = 181
                self.block()
                self.state = 182
                self.match(AtomicParser.RBRACE)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 184
                self.match(AtomicParser.IF)
                self.state = 185
                self.match(AtomicParser.LBRACKET)
                self.state = 186
                self.cond()
                self.state = 187
                self.match(AtomicParser.RBRACKET)
                self.state = 188
                self.match(AtomicParser.LBRACE)
                self.state = 189
                self.block()
                self.state = 190
                self.match(AtomicParser.RBRACE)
                self.state = 192
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==2:
                    self.state = 191
                    self.match(AtomicParser.EOL)


                self.state = 194
                self.match(AtomicParser.ELSE)
                self.state = 195
                self.match(AtomicParser.LBRACE)
                self.state = 196
                self.block()
                self.state = 197
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
            self.state = 201
            self.match(AtomicParser.IONREGISTER)
            self.state = 202
            self.match(AtomicParser.LBRACKET)
            self.state = 203
            self.match(AtomicParser.INT)
            self.state = 204
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
            self.state = 206
            self.match(AtomicParser.PARALLEL)
            self.state = 207
            self.match(AtomicParser.LBRACE)
            self.state = 208
            self.block()
            self.state = 209
            self.match(AtomicParser.RBRACE)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Serial_stmtContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def SERIAL(self):
            return self.getToken(AtomicParser.SERIAL, 0)

        def LBRACE(self):
            return self.getToken(AtomicParser.LBRACE, 0)

        def block(self):
            return self.getTypedRuleContext(AtomicParser.BlockContext,0)


        def RBRACE(self):
            return self.getToken(AtomicParser.RBRACE, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_serial_stmt

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterSerial_stmt" ):
                listener.enterSerial_stmt(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitSerial_stmt" ):
                listener.exitSerial_stmt(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSerial_stmt" ):
                return visitor.visitSerial_stmt(self)
            else:
                return visitor.visitChildren(self)




    def serial_stmt(self):

        localctx = AtomicParser.Serial_stmtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 32, self.RULE_serial_stmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 211
            self.match(AtomicParser.SERIAL)
            self.state = 212
            self.match(AtomicParser.LBRACE)
            self.state = 213
            self.block()
            self.state = 214
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
        self.enterRule(localctx, 34, self.RULE_targets)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 216
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
        self.enterRule(localctx, 36, self.RULE_bool_and_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 218
            _la = self._input.LA(1)
            if not(_la==17 or _la==18):
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
        self.enterRule(localctx, 38, self.RULE_bool_or_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 220
            _la = self._input.LA(1)
            if not(_la==19 or _la==20):
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
        self.enterRule(localctx, 40, self.RULE_bool_not_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 222
            _la = self._input.LA(1)
            if not(_la==21 or _la==22):
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
        self.enterRule(localctx, 42, self.RULE_bool_eq_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 224
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
        self.enterRule(localctx, 44, self.RULE_bool_not_eq_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 226
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
        self.enterRule(localctx, 46, self.RULE_bool_lt_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 228
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
        self.enterRule(localctx, 48, self.RULE_bool_lte_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 230
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
        self.enterRule(localctx, 50, self.RULE_bool_gt_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 232
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
        self.enterRule(localctx, 52, self.RULE_bool_gte_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 234
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
        self.enterRule(localctx, 54, self.RULE_bool_literal)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 236
            _la = self._input.LA(1)
            if not(_la==23 or _la==24):
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
        self.enterRule(localctx, 56, self.RULE_comparators)
        try:
            self.state = 244
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [40]:
                self.enterOuterAlt(localctx, 1)
                self.state = 238
                self.bool_eq_op()
                pass
            elif token in [41]:
                self.enterOuterAlt(localctx, 2)
                self.state = 239
                self.bool_not_eq_op()
                pass
            elif token in [42]:
                self.enterOuterAlt(localctx, 3)
                self.state = 240
                self.bool_lt_op()
                pass
            elif token in [43]:
                self.enterOuterAlt(localctx, 4)
                self.state = 241
                self.bool_lte_op()
                pass
            elif token in [44]:
                self.enterOuterAlt(localctx, 5)
                self.state = 242
                self.bool_gt_op()
                pass
            elif token in [45]:
                self.enterOuterAlt(localctx, 6)
                self.state = 243
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


        def atomic_list_extract(self):
            return self.getTypedRuleContext(AtomicParser.Atomic_list_extractContext,0)


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
        self.enterRule(localctx, 58, self.RULE_math_terminal)
        try:
            self.state = 255
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,14,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 246
                self.match(AtomicParser.INT)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 247
                self.match(AtomicParser.FLOAT)
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 248
                self.match(AtomicParser.MATH_VAR)
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 249
                self.match(AtomicParser.IMAG)
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 250
                self.access()
                pass

            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 251
                self.pexpr()
                pass

            elif la_ == 7:
                self.enterOuterAlt(localctx, 7)
                self.state = 252
                self.fexpr()
                pass

            elif la_ == 8:
                self.enterOuterAlt(localctx, 8)
                self.state = 253
                self.atomic_list()
                pass

            elif la_ == 9:
                self.enterOuterAlt(localctx, 9)
                self.state = 254
                self.atomic_list_extract()
                pass


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
        self.enterRule(localctx, 60, self.RULE_func_names)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 257
            _la = self._input.LA(1)
            if not(((((_la - 15)) & ~0x3f) == 0 and ((1 << (_la - 15)) & 36028762659225603) != 0)):
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
        self.enterRule(localctx, 62, self.RULE_pexpr)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 259
            self.match(AtomicParser.LBRACKET)
            self.state = 260
            self.aexpr(0)
            self.state = 261
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
        self.enterRule(localctx, 64, self.RULE_fexpr)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 263
            self.func_names()
            self.state = 264
            self.match(AtomicParser.LBRACKET)
            self.state = 273
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if ((((_la - 14)) & ~0x3f) == 0 and ((1 << (_la - 14)) & 144115183793554951) != 0):
                self.state = 265
                self.aexpr(0)
                self.state = 270
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la==27:
                    self.state = 266
                    self.match(AtomicParser.COMMA)
                    self.state = 267
                    self.aexpr(0)
                    self.state = 272
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)



            self.state = 275
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
        _startState = 66
        self.enterRecursionRule(localctx, 66, self.RULE_aexpr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 278
            self.mexpr(0)
            self._ctx.stop = self._input.LT(-1)
            self.state = 285
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,17,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = AtomicParser.AexprContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_aexpr)
                    self.state = 280
                    if not self.precpred(self._ctx, 1):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 1)")
                    self.state = 281
                    _la = self._input.LA(1)
                    if not(_la==36 or _la==37):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 282
                    self.mexpr(0) 
                self.state = 287
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
        _startState = 68
        self.enterRecursionRule(localctx, 68, self.RULE_mexpr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 289
            self.uexpr()
            self._ctx.stop = self._input.LT(-1)
            self.state = 296
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,18,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = AtomicParser.MexprContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_mexpr)
                    self.state = 291
                    if not self.precpred(self._ctx, 1):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 1)")
                    self.state = 292
                    _la = self._input.LA(1)
                    if not(_la==34 or _la==35):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 293
                    self.uexpr() 
                self.state = 298
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
        self.enterRule(localctx, 70, self.RULE_uexpr)
        self._la = 0 # Token type
        try:
            self.state = 302
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [14, 15, 16, 23, 24, 28, 30, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]:
                self.enterOuterAlt(localctx, 1)
                self.state = 299
                self.eexpr()
                pass
            elif token in [36, 37]:
                self.enterOuterAlt(localctx, 2)
                self.state = 300
                _la = self._input.LA(1)
                if not(_la==36 or _la==37):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 301
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
        self.enterRule(localctx, 72, self.RULE_eexpr)
        try:
            self.state = 309
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,20,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 304
                self.terminal()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 305
                self.terminal()
                self.state = 306
                self.match(AtomicParser.POWER)
                self.state = 307
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
        self._predicates[33] = self.aexpr_sempred
        self._predicates[34] = self.mexpr_sempred
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
         




