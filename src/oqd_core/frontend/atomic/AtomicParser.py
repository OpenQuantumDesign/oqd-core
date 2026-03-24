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
        4,1,69,333,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,20,
        7,20,2,21,7,21,2,22,7,22,2,23,7,23,2,24,7,24,2,25,7,25,2,26,7,26,
        2,27,7,27,2,28,7,28,2,29,7,29,2,30,7,30,2,31,7,31,2,32,7,32,2,33,
        7,33,2,34,7,34,2,35,7,35,2,36,7,36,2,37,7,37,2,38,7,38,2,39,7,39,
        1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,91,8,1,1,2,1,2,1,2,1,
        2,5,2,97,8,2,10,2,12,2,100,9,2,1,2,3,2,103,8,2,1,3,1,3,1,3,3,3,108,
        8,3,1,4,1,4,1,4,1,4,1,4,4,4,115,8,4,11,4,12,4,116,1,4,1,4,1,4,1,
        4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,3,4,131,8,4,1,4,1,4,1,4,3,4,136,
        8,4,1,4,1,4,5,4,140,8,4,10,4,12,4,143,9,4,1,5,1,5,1,6,1,6,3,6,149,
        8,6,1,6,1,6,5,6,153,8,6,10,6,12,6,156,9,6,1,6,1,6,1,7,1,7,1,7,1,
        7,1,8,1,8,1,9,1,9,1,9,1,9,1,9,1,10,1,10,1,11,1,11,1,12,1,12,1,12,
        1,12,1,12,1,12,1,12,1,12,1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,
        1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,3,13,199,8,13,1,13,1,13,
        1,13,1,13,1,13,3,13,206,8,13,1,14,1,14,1,14,1,14,1,14,1,15,1,15,
        1,15,1,15,1,15,1,15,1,15,1,15,1,15,1,15,1,15,1,15,1,15,1,16,1,16,
        1,16,1,16,1,16,1,16,1,16,1,16,1,17,1,17,1,17,1,17,1,17,1,18,1,18,
        1,18,1,18,1,18,1,18,1,18,1,18,3,18,247,8,18,1,19,1,19,1,20,1,20,
        1,21,1,21,1,22,1,22,1,23,1,23,1,24,1,24,1,25,1,25,1,26,1,26,1,27,
        1,27,1,28,1,28,1,29,1,29,1,30,1,30,1,31,1,31,1,31,1,31,1,31,1,31,
        3,31,279,8,31,1,32,1,32,1,32,1,32,1,32,1,32,1,32,3,32,288,8,32,1,
        33,1,33,1,34,1,34,1,34,1,34,1,35,1,35,1,35,1,36,1,36,1,36,1,36,1,
        36,1,36,5,36,305,8,36,10,36,12,36,308,9,36,1,37,1,37,1,37,1,37,1,
        37,1,37,5,37,316,8,37,10,37,12,37,319,9,37,1,38,1,38,1,38,3,38,324,
        8,38,1,39,1,39,1,39,1,39,1,39,3,39,331,8,39,1,39,0,3,8,72,74,40,
        0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,
        46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,0,7,1,0,16,17,
        1,0,18,19,1,0,20,21,1,0,22,23,1,0,49,68,1,0,35,36,1,0,33,34,333,
        0,80,1,0,0,0,2,90,1,0,0,0,4,98,1,0,0,0,6,107,1,0,0,0,8,130,1,0,0,
        0,10,144,1,0,0,0,12,146,1,0,0,0,14,159,1,0,0,0,16,163,1,0,0,0,18,
        165,1,0,0,0,20,170,1,0,0,0,22,172,1,0,0,0,24,174,1,0,0,0,26,205,
        1,0,0,0,28,207,1,0,0,0,30,212,1,0,0,0,32,225,1,0,0,0,34,233,1,0,
        0,0,36,238,1,0,0,0,38,248,1,0,0,0,40,250,1,0,0,0,42,252,1,0,0,0,
        44,254,1,0,0,0,46,256,1,0,0,0,48,258,1,0,0,0,50,260,1,0,0,0,52,262,
        1,0,0,0,54,264,1,0,0,0,56,266,1,0,0,0,58,268,1,0,0,0,60,270,1,0,
        0,0,62,278,1,0,0,0,64,287,1,0,0,0,66,289,1,0,0,0,68,291,1,0,0,0,
        70,295,1,0,0,0,72,298,1,0,0,0,74,309,1,0,0,0,76,323,1,0,0,0,78,330,
        1,0,0,0,80,81,3,4,2,0,81,82,5,0,0,1,82,1,1,0,0,0,83,91,3,14,7,0,
        84,91,3,36,18,0,85,91,3,34,17,0,86,91,3,24,12,0,87,91,3,26,13,0,
        88,91,3,20,10,0,89,91,3,22,11,0,90,83,1,0,0,0,90,84,1,0,0,0,90,85,
        1,0,0,0,90,86,1,0,0,0,90,87,1,0,0,0,90,88,1,0,0,0,90,89,1,0,0,0,
        91,3,1,0,0,0,92,93,3,2,1,0,93,94,5,2,0,0,94,97,1,0,0,0,95,97,5,2,
        0,0,96,92,1,0,0,0,96,95,1,0,0,0,97,100,1,0,0,0,98,96,1,0,0,0,98,
        99,1,0,0,0,99,102,1,0,0,0,100,98,1,0,0,0,101,103,3,2,1,0,102,101,
        1,0,0,0,102,103,1,0,0,0,103,5,1,0,0,0,104,108,3,28,14,0,105,108,
        3,64,32,0,106,108,3,60,30,0,107,104,1,0,0,0,107,105,1,0,0,0,107,
        106,1,0,0,0,108,7,1,0,0,0,109,110,6,4,-1,0,110,114,3,72,36,0,111,
        112,3,62,31,0,112,113,3,72,36,0,113,115,1,0,0,0,114,111,1,0,0,0,
        115,116,1,0,0,0,116,114,1,0,0,0,116,117,1,0,0,0,117,131,1,0,0,0,
        118,119,3,46,23,0,119,120,3,8,4,7,120,131,1,0,0,0,121,122,5,27,0,
        0,122,123,3,8,4,0,123,124,5,28,0,0,124,131,1,0,0,0,125,131,3,18,
        9,0,126,131,3,12,6,0,127,131,3,6,3,0,128,131,3,72,36,0,129,131,3,
        30,15,0,130,109,1,0,0,0,130,118,1,0,0,0,130,121,1,0,0,0,130,125,
        1,0,0,0,130,126,1,0,0,0,130,127,1,0,0,0,130,128,1,0,0,0,130,129,
        1,0,0,0,131,141,1,0,0,0,132,135,10,8,0,0,133,136,3,42,21,0,134,136,
        3,44,22,0,135,133,1,0,0,0,135,134,1,0,0,0,136,137,1,0,0,0,137,138,
        3,8,4,9,138,140,1,0,0,0,139,132,1,0,0,0,140,143,1,0,0,0,141,139,
        1,0,0,0,141,142,1,0,0,0,142,9,1,0,0,0,143,141,1,0,0,0,144,145,3,
        8,4,0,145,11,1,0,0,0,146,148,5,29,0,0,147,149,3,8,4,0,148,147,1,
        0,0,0,148,149,1,0,0,0,149,154,1,0,0,0,150,151,5,26,0,0,151,153,3,
        8,4,0,152,150,1,0,0,0,153,156,1,0,0,0,154,152,1,0,0,0,154,155,1,
        0,0,0,155,157,1,0,0,0,156,154,1,0,0,0,157,158,5,30,0,0,158,13,1,
        0,0,0,159,160,5,69,0,0,160,161,5,38,0,0,161,162,3,8,4,0,162,15,1,
        0,0,0,163,164,5,69,0,0,164,17,1,0,0,0,165,166,3,16,8,0,166,167,5,
        29,0,0,167,168,5,45,0,0,168,169,5,30,0,0,169,19,1,0,0,0,170,171,
        5,11,0,0,171,21,1,0,0,0,172,173,5,12,0,0,173,23,1,0,0,0,174,175,
        5,8,0,0,175,176,5,27,0,0,176,177,3,10,5,0,177,178,5,28,0,0,178,179,
        5,31,0,0,179,180,3,4,2,0,180,181,5,32,0,0,181,25,1,0,0,0,182,183,
        5,6,0,0,183,184,5,27,0,0,184,185,3,10,5,0,185,186,5,28,0,0,186,187,
        5,31,0,0,187,188,3,4,2,0,188,189,5,32,0,0,189,206,1,0,0,0,190,191,
        5,6,0,0,191,192,5,27,0,0,192,193,3,10,5,0,193,194,5,28,0,0,194,195,
        5,31,0,0,195,196,3,4,2,0,196,198,5,32,0,0,197,199,5,2,0,0,198,197,
        1,0,0,0,198,199,1,0,0,0,199,200,1,0,0,0,200,201,5,7,0,0,201,202,
        5,31,0,0,202,203,3,4,2,0,203,204,5,32,0,0,204,206,1,0,0,0,205,182,
        1,0,0,0,205,190,1,0,0,0,206,27,1,0,0,0,207,208,5,13,0,0,208,209,
        5,27,0,0,209,210,5,45,0,0,210,211,5,28,0,0,211,29,1,0,0,0,212,213,
        5,14,0,0,213,214,5,27,0,0,214,215,3,8,4,0,215,216,5,26,0,0,216,217,
        3,8,4,0,217,218,5,26,0,0,218,219,3,8,4,0,219,220,5,26,0,0,220,221,
        3,32,16,0,221,222,5,26,0,0,222,223,3,32,16,0,223,224,5,28,0,0,224,
        31,1,0,0,0,225,226,5,29,0,0,226,227,3,8,4,0,227,228,5,26,0,0,228,
        229,3,8,4,0,229,230,5,26,0,0,230,231,3,8,4,0,231,232,5,30,0,0,232,
        33,1,0,0,0,233,234,5,5,0,0,234,235,5,31,0,0,235,236,3,4,2,0,236,
        237,5,32,0,0,237,35,1,0,0,0,238,239,5,15,0,0,239,240,3,40,20,0,240,
        241,5,9,0,0,241,242,3,8,4,0,242,243,5,10,0,0,243,246,3,8,4,0,244,
        245,5,26,0,0,245,247,3,38,19,0,246,244,1,0,0,0,246,247,1,0,0,0,247,
        37,1,0,0,0,248,249,3,8,4,0,249,39,1,0,0,0,250,251,3,8,4,0,251,41,
        1,0,0,0,252,253,7,0,0,0,253,43,1,0,0,0,254,255,7,1,0,0,255,45,1,
        0,0,0,256,257,7,2,0,0,257,47,1,0,0,0,258,259,5,39,0,0,259,49,1,0,
        0,0,260,261,5,40,0,0,261,51,1,0,0,0,262,263,5,41,0,0,263,53,1,0,
        0,0,264,265,5,42,0,0,265,55,1,0,0,0,266,267,5,43,0,0,267,57,1,0,
        0,0,268,269,5,44,0,0,269,59,1,0,0,0,270,271,7,3,0,0,271,61,1,0,0,
        0,272,279,3,48,24,0,273,279,3,50,25,0,274,279,3,52,26,0,275,279,
        3,54,27,0,276,279,3,56,28,0,277,279,3,58,29,0,278,272,1,0,0,0,278,
        273,1,0,0,0,278,274,1,0,0,0,278,275,1,0,0,0,278,276,1,0,0,0,278,
        277,1,0,0,0,279,63,1,0,0,0,280,288,5,45,0,0,281,288,5,46,0,0,282,
        288,5,47,0,0,283,288,5,48,0,0,284,288,3,16,8,0,285,288,3,68,34,0,
        286,288,3,70,35,0,287,280,1,0,0,0,287,281,1,0,0,0,287,282,1,0,0,
        0,287,283,1,0,0,0,287,284,1,0,0,0,287,285,1,0,0,0,287,286,1,0,0,
        0,288,65,1,0,0,0,289,290,7,4,0,0,290,67,1,0,0,0,291,292,5,27,0,0,
        292,293,3,72,36,0,293,294,5,28,0,0,294,69,1,0,0,0,295,296,3,66,33,
        0,296,297,3,68,34,0,297,71,1,0,0,0,298,299,6,36,-1,0,299,300,3,74,
        37,0,300,306,1,0,0,0,301,302,10,1,0,0,302,303,7,5,0,0,303,305,3,
        74,37,0,304,301,1,0,0,0,305,308,1,0,0,0,306,304,1,0,0,0,306,307,
        1,0,0,0,307,73,1,0,0,0,308,306,1,0,0,0,309,310,6,37,-1,0,310,311,
        3,76,38,0,311,317,1,0,0,0,312,313,10,1,0,0,313,314,7,6,0,0,314,316,
        3,76,38,0,315,312,1,0,0,0,316,319,1,0,0,0,317,315,1,0,0,0,317,318,
        1,0,0,0,318,75,1,0,0,0,319,317,1,0,0,0,320,324,3,78,39,0,321,322,
        7,5,0,0,322,324,3,78,39,0,323,320,1,0,0,0,323,321,1,0,0,0,324,77,
        1,0,0,0,325,331,3,6,3,0,326,327,3,6,3,0,327,328,5,37,0,0,328,329,
        3,76,38,0,329,331,1,0,0,0,330,325,1,0,0,0,330,326,1,0,0,0,331,79,
        1,0,0,0,20,90,96,98,102,107,116,130,135,141,148,154,198,205,246,
        278,287,306,317,323,330
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
    RULE_atom = 3
    RULE_expr = 4
    RULE_cond = 5
    RULE_my_list = 6
    RULE_declaration = 7
    RULE_access = 8
    RULE_extract = 9
    RULE_break_stmt = 10
    RULE_continue_stmt = 11
    RULE_while_stmt = 12
    RULE_ifelse_stmt = 13
    RULE_ion_register = 14
    RULE_beam_expr = 15
    RULE_vec3 = 16
    RULE_parallel_stmt = 17
    RULE_pulse_stmt = 18
    RULE_measured = 19
    RULE_targets = 20
    RULE_bool_and_op = 21
    RULE_bool_or_op = 22
    RULE_bool_not_op = 23
    RULE_bool_eq_op = 24
    RULE_bool_not_eq_op = 25
    RULE_bool_lt_op = 26
    RULE_bool_lte_op = 27
    RULE_bool_gt_op = 28
    RULE_bool_gte_op = 29
    RULE_bool_literal = 30
    RULE_comparators = 31
    RULE_math_terminal = 32
    RULE_math_func_name = 33
    RULE_pexpr = 34
    RULE_fexpr = 35
    RULE_aexpr = 36
    RULE_mexpr = 37
    RULE_uexpr = 38
    RULE_eexpr = 39

    ruleNames =  [ "program", "statement", "block", "atom", "expr", "cond", 
                   "my_list", "declaration", "access", "extract", "break_stmt", 
                   "continue_stmt", "while_stmt", "ifelse_stmt", "ion_register", 
                   "beam_expr", "vec3", "parallel_stmt", "pulse_stmt", "measured", 
                   "targets", "bool_and_op", "bool_or_op", "bool_not_op", 
                   "bool_eq_op", "bool_not_eq_op", "bool_lt_op", "bool_lte_op", 
                   "bool_gt_op", "bool_gte_op", "bool_literal", "comparators", 
                   "math_terminal", "math_func_name", "pexpr", "fexpr", 
                   "aexpr", "mexpr", "uexpr", "eexpr" ]

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
            self.state = 80
            self.block()
            self.state = 81
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


        def pulse_stmt(self):
            return self.getTypedRuleContext(AtomicParser.Pulse_stmtContext,0)


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
            self.state = 90
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [69]:
                self.enterOuterAlt(localctx, 1)
                self.state = 83
                self.declaration()
                pass
            elif token in [15]:
                self.enterOuterAlt(localctx, 2)
                self.state = 84
                self.pulse_stmt()
                pass
            elif token in [5]:
                self.enterOuterAlt(localctx, 3)
                self.state = 85
                self.parallel_stmt()
                pass
            elif token in [8]:
                self.enterOuterAlt(localctx, 4)
                self.state = 86
                self.while_stmt()
                pass
            elif token in [6]:
                self.enterOuterAlt(localctx, 5)
                self.state = 87
                self.ifelse_stmt()
                pass
            elif token in [11]:
                self.enterOuterAlt(localctx, 6)
                self.state = 88
                self.break_stmt()
                pass
            elif token in [12]:
                self.enterOuterAlt(localctx, 7)
                self.state = 89
                self.continue_stmt()
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
            self.state = 98
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,2,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    self.state = 96
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [5, 6, 8, 11, 12, 15, 69]:
                        self.state = 92
                        self.statement()
                        self.state = 93
                        self.match(AtomicParser.EOL)
                        pass
                    elif token in [2]:
                        self.state = 95
                        self.match(AtomicParser.EOL)
                        pass
                    else:
                        raise NoViableAltException(self)
             
                self.state = 100
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,2,self._ctx)

            self.state = 102
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if (((_la) & ~0x3f) == 0 and ((1 << _la) & 39264) != 0) or _la==69:
                self.state = 101
                self.statement()


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AtomContext(ParserRuleContext):
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
            return AtomicParser.RULE_atom

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAtom" ):
                listener.enterAtom(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAtom" ):
                listener.exitAtom(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAtom" ):
                return visitor.visitAtom(self)
            else:
                return visitor.visitChildren(self)




    def atom(self):

        localctx = AtomicParser.AtomContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_atom)
        try:
            self.state = 107
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [13]:
                self.enterOuterAlt(localctx, 1)
                self.state = 104
                self.ion_register()
                pass
            elif token in [27, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69]:
                self.enterOuterAlt(localctx, 2)
                self.state = 105
                self.math_terminal()
                pass
            elif token in [22, 23]:
                self.enterOuterAlt(localctx, 3)
                self.state = 106
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

        def extract(self):
            return self.getTypedRuleContext(AtomicParser.ExtractContext,0)


        def my_list(self):
            return self.getTypedRuleContext(AtomicParser.My_listContext,0)


        def atom(self):
            return self.getTypedRuleContext(AtomicParser.AtomContext,0)


        def beam_expr(self):
            return self.getTypedRuleContext(AtomicParser.Beam_exprContext,0)


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
            self.state = 130
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,6,self._ctx)
            if la_ == 1:
                self.state = 110
                self.aexpr(0)
                self.state = 114 
                self._errHandler.sync(self)
                _alt = 1
                while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 111
                        self.comparators()
                        self.state = 112
                        self.aexpr(0)

                    else:
                        raise NoViableAltException(self)
                    self.state = 116 
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input,5,self._ctx)

                pass

            elif la_ == 2:
                self.state = 118
                self.bool_not_op()
                self.state = 119
                self.expr(7)
                pass

            elif la_ == 3:
                self.state = 121
                self.match(AtomicParser.LBRACKET)
                self.state = 122
                self.expr(0)
                self.state = 123
                self.match(AtomicParser.RBRACKET)
                pass

            elif la_ == 4:
                self.state = 125
                self.extract()
                pass

            elif la_ == 5:
                self.state = 126
                self.my_list()
                pass

            elif la_ == 6:
                self.state = 127
                self.atom()
                pass

            elif la_ == 7:
                self.state = 128
                self.aexpr(0)
                pass

            elif la_ == 8:
                self.state = 129
                self.beam_expr()
                pass


            self._ctx.stop = self._input.LT(-1)
            self.state = 141
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,8,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = AtomicParser.ExprContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                    self.state = 132
                    if not self.precpred(self._ctx, 8):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 8)")
                    self.state = 135
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [16, 17]:
                        self.state = 133
                        self.bool_and_op()
                        pass
                    elif token in [18, 19]:
                        self.state = 134
                        self.bool_or_op()
                        pass
                    else:
                        raise NoViableAltException(self)

                    self.state = 137
                    self.expr(9) 
                self.state = 143
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
            self.state = 144
            self.expr(0)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class My_listContext(ParserRuleContext):
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
            return AtomicParser.RULE_my_list

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMy_list" ):
                listener.enterMy_list(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMy_list" ):
                listener.exitMy_list(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMy_list" ):
                return visitor.visitMy_list(self)
            else:
                return visitor.visitChildren(self)




    def my_list(self):

        localctx = AtomicParser.My_listContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_my_list)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 146
            self.match(AtomicParser.SQUARELBRACKET)
            self.state = 148
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if ((((_la - 13)) & ~0x3f) == 0 and ((1 << (_la - 13)) & 144115183793555331) != 0):
                self.state = 147
                self.expr(0)


            self.state = 154
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==26:
                self.state = 150
                self.match(AtomicParser.COMMA)
                self.state = 151
                self.expr(0)
                self.state = 156
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 157
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
            self.state = 159
            self.match(AtomicParser.ID)
            self.state = 160
            self.match(AtomicParser.ASSIGN)
            self.state = 161
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
            self.state = 163
            self.match(AtomicParser.ID)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ExtractContext(ParserRuleContext):
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
            return AtomicParser.RULE_extract

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterExtract" ):
                listener.enterExtract(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitExtract" ):
                listener.exitExtract(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitExtract" ):
                return visitor.visitExtract(self)
            else:
                return visitor.visitChildren(self)




    def extract(self):

        localctx = AtomicParser.ExtractContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_extract)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 165
            self.access()
            self.state = 166
            self.match(AtomicParser.SQUARELBRACKET)
            self.state = 167
            self.match(AtomicParser.INT)
            self.state = 168
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
            self.state = 170
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
            self.state = 172
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
            self.state = 174
            self.match(AtomicParser.WHILE)
            self.state = 175
            self.match(AtomicParser.LBRACKET)
            self.state = 176
            self.cond()
            self.state = 177
            self.match(AtomicParser.RBRACKET)
            self.state = 178
            self.match(AtomicParser.LBRACE)
            self.state = 179
            self.block()
            self.state = 180
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
            self.state = 205
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,12,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 182
                self.match(AtomicParser.IF)
                self.state = 183
                self.match(AtomicParser.LBRACKET)
                self.state = 184
                self.cond()
                self.state = 185
                self.match(AtomicParser.RBRACKET)
                self.state = 186
                self.match(AtomicParser.LBRACE)
                self.state = 187
                self.block()
                self.state = 188
                self.match(AtomicParser.RBRACE)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 190
                self.match(AtomicParser.IF)
                self.state = 191
                self.match(AtomicParser.LBRACKET)
                self.state = 192
                self.cond()
                self.state = 193
                self.match(AtomicParser.RBRACKET)
                self.state = 194
                self.match(AtomicParser.LBRACE)
                self.state = 195
                self.block()
                self.state = 196
                self.match(AtomicParser.RBRACE)
                self.state = 198
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==2:
                    self.state = 197
                    self.match(AtomicParser.EOL)


                self.state = 200
                self.match(AtomicParser.ELSE)
                self.state = 201
                self.match(AtomicParser.LBRACE)
                self.state = 202
                self.block()
                self.state = 203
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
            self.state = 207
            self.match(AtomicParser.IONREGISTER)
            self.state = 208
            self.match(AtomicParser.LBRACKET)
            self.state = 209
            self.match(AtomicParser.INT)
            self.state = 210
            self.match(AtomicParser.RBRACKET)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Beam_exprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def BEAM(self):
            return self.getToken(AtomicParser.BEAM, 0)

        def LBRACKET(self):
            return self.getToken(AtomicParser.LBRACKET, 0)

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

        def vec3(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AtomicParser.Vec3Context)
            else:
                return self.getTypedRuleContext(AtomicParser.Vec3Context,i)


        def RBRACKET(self):
            return self.getToken(AtomicParser.RBRACKET, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_beam_expr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBeam_expr" ):
                listener.enterBeam_expr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBeam_expr" ):
                listener.exitBeam_expr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBeam_expr" ):
                return visitor.visitBeam_expr(self)
            else:
                return visitor.visitChildren(self)




    def beam_expr(self):

        localctx = AtomicParser.Beam_exprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_beam_expr)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 212
            self.match(AtomicParser.BEAM)
            self.state = 213
            self.match(AtomicParser.LBRACKET)
            self.state = 214
            self.expr(0)
            self.state = 215
            self.match(AtomicParser.COMMA)
            self.state = 216
            self.expr(0)
            self.state = 217
            self.match(AtomicParser.COMMA)
            self.state = 218
            self.expr(0)
            self.state = 219
            self.match(AtomicParser.COMMA)
            self.state = 220
            self.vec3()
            self.state = 221
            self.match(AtomicParser.COMMA)
            self.state = 222
            self.vec3()
            self.state = 223
            self.match(AtomicParser.RBRACKET)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Vec3Context(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def SQUARELBRACKET(self):
            return self.getToken(AtomicParser.SQUARELBRACKET, 0)

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

        def SQUARERBRACKET(self):
            return self.getToken(AtomicParser.SQUARERBRACKET, 0)

        def getRuleIndex(self):
            return AtomicParser.RULE_vec3

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterVec3" ):
                listener.enterVec3(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitVec3" ):
                listener.exitVec3(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitVec3" ):
                return visitor.visitVec3(self)
            else:
                return visitor.visitChildren(self)




    def vec3(self):

        localctx = AtomicParser.Vec3Context(self, self._ctx, self.state)
        self.enterRule(localctx, 32, self.RULE_vec3)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 225
            self.match(AtomicParser.SQUARELBRACKET)
            self.state = 226
            self.expr(0)
            self.state = 227
            self.match(AtomicParser.COMMA)
            self.state = 228
            self.expr(0)
            self.state = 229
            self.match(AtomicParser.COMMA)
            self.state = 230
            self.expr(0)
            self.state = 231
            self.match(AtomicParser.SQUARERBRACKET)
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
        self.enterRule(localctx, 34, self.RULE_parallel_stmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 233
            self.match(AtomicParser.PARALLEL)
            self.state = 234
            self.match(AtomicParser.LBRACE)
            self.state = 235
            self.block()
            self.state = 236
            self.match(AtomicParser.RBRACE)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Pulse_stmtContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def PULSE(self):
            return self.getToken(AtomicParser.PULSE, 0)

        def targets(self):
            return self.getTypedRuleContext(AtomicParser.TargetsContext,0)


        def WITH(self):
            return self.getToken(AtomicParser.WITH, 0)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AtomicParser.ExprContext)
            else:
                return self.getTypedRuleContext(AtomicParser.ExprContext,i)


        def FOR(self):
            return self.getToken(AtomicParser.FOR, 0)

        def COMMA(self):
            return self.getToken(AtomicParser.COMMA, 0)

        def measured(self):
            return self.getTypedRuleContext(AtomicParser.MeasuredContext,0)


        def getRuleIndex(self):
            return AtomicParser.RULE_pulse_stmt

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPulse_stmt" ):
                listener.enterPulse_stmt(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPulse_stmt" ):
                listener.exitPulse_stmt(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPulse_stmt" ):
                return visitor.visitPulse_stmt(self)
            else:
                return visitor.visitChildren(self)




    def pulse_stmt(self):

        localctx = AtomicParser.Pulse_stmtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 36, self.RULE_pulse_stmt)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 238
            self.match(AtomicParser.PULSE)
            self.state = 239
            self.targets()
            self.state = 240
            self.match(AtomicParser.WITH)
            self.state = 241
            self.expr(0)
            self.state = 242
            self.match(AtomicParser.FOR)
            self.state = 243
            self.expr(0)
            self.state = 246
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==26:
                self.state = 244
                self.match(AtomicParser.COMMA)
                self.state = 245
                self.measured()


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class MeasuredContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self):
            return self.getTypedRuleContext(AtomicParser.ExprContext,0)


        def getRuleIndex(self):
            return AtomicParser.RULE_measured

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMeasured" ):
                listener.enterMeasured(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMeasured" ):
                listener.exitMeasured(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMeasured" ):
                return visitor.visitMeasured(self)
            else:
                return visitor.visitChildren(self)




    def measured(self):

        localctx = AtomicParser.MeasuredContext(self, self._ctx, self.state)
        self.enterRule(localctx, 38, self.RULE_measured)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 248
            self.expr(0)
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
        self.enterRule(localctx, 40, self.RULE_targets)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 250
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
        self.enterRule(localctx, 42, self.RULE_bool_and_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 252
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
        self.enterRule(localctx, 44, self.RULE_bool_or_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 254
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
        self.enterRule(localctx, 46, self.RULE_bool_not_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 256
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
        self.enterRule(localctx, 48, self.RULE_bool_eq_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 258
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
        self.enterRule(localctx, 50, self.RULE_bool_not_eq_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 260
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
        self.enterRule(localctx, 52, self.RULE_bool_lt_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 262
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
        self.enterRule(localctx, 54, self.RULE_bool_lte_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 264
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
        self.enterRule(localctx, 56, self.RULE_bool_gt_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 266
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
        self.enterRule(localctx, 58, self.RULE_bool_gte_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 268
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
        self.enterRule(localctx, 60, self.RULE_bool_literal)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 270
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
        self.enterRule(localctx, 62, self.RULE_comparators)
        try:
            self.state = 278
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [39]:
                self.enterOuterAlt(localctx, 1)
                self.state = 272
                self.bool_eq_op()
                pass
            elif token in [40]:
                self.enterOuterAlt(localctx, 2)
                self.state = 273
                self.bool_not_eq_op()
                pass
            elif token in [41]:
                self.enterOuterAlt(localctx, 3)
                self.state = 274
                self.bool_lt_op()
                pass
            elif token in [42]:
                self.enterOuterAlt(localctx, 4)
                self.state = 275
                self.bool_lte_op()
                pass
            elif token in [43]:
                self.enterOuterAlt(localctx, 5)
                self.state = 276
                self.bool_gt_op()
                pass
            elif token in [44]:
                self.enterOuterAlt(localctx, 6)
                self.state = 277
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
        self.enterRule(localctx, 64, self.RULE_math_terminal)
        try:
            self.state = 287
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [45]:
                self.enterOuterAlt(localctx, 1)
                self.state = 280
                self.match(AtomicParser.INT)
                pass
            elif token in [46]:
                self.enterOuterAlt(localctx, 2)
                self.state = 281
                self.match(AtomicParser.FLOAT)
                pass
            elif token in [47]:
                self.enterOuterAlt(localctx, 3)
                self.state = 282
                self.match(AtomicParser.MATH_VAR)
                pass
            elif token in [48]:
                self.enterOuterAlt(localctx, 4)
                self.state = 283
                self.match(AtomicParser.IMAG)
                pass
            elif token in [69]:
                self.enterOuterAlt(localctx, 5)
                self.state = 284
                self.access()
                pass
            elif token in [27]:
                self.enterOuterAlt(localctx, 6)
                self.state = 285
                self.pexpr()
                pass
            elif token in [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68]:
                self.enterOuterAlt(localctx, 7)
                self.state = 286
                self.fexpr()
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


    class Math_func_nameContext(ParserRuleContext):
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

        def getRuleIndex(self):
            return AtomicParser.RULE_math_func_name

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMath_func_name" ):
                listener.enterMath_func_name(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMath_func_name" ):
                listener.exitMath_func_name(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMath_func_name" ):
                return visitor.visitMath_func_name(self)
            else:
                return visitor.visitChildren(self)




    def math_func_name(self):

        localctx = AtomicParser.Math_func_nameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 66, self.RULE_math_func_name)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 289
            _la = self._input.LA(1)
            if not(((((_la - 49)) & ~0x3f) == 0 and ((1 << (_la - 49)) & 1048575) != 0)):
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
        self.enterRule(localctx, 68, self.RULE_pexpr)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 291
            self.match(AtomicParser.LBRACKET)
            self.state = 292
            self.aexpr(0)
            self.state = 293
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

        def math_func_name(self):
            return self.getTypedRuleContext(AtomicParser.Math_func_nameContext,0)


        def pexpr(self):
            return self.getTypedRuleContext(AtomicParser.PexprContext,0)


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
        self.enterRule(localctx, 70, self.RULE_fexpr)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 295
            self.math_func_name()
            self.state = 296
            self.pexpr()
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
        _startState = 72
        self.enterRecursionRule(localctx, 72, self.RULE_aexpr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 299
            self.mexpr(0)
            self._ctx.stop = self._input.LT(-1)
            self.state = 306
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,16,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = AtomicParser.AexprContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_aexpr)
                    self.state = 301
                    if not self.precpred(self._ctx, 1):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 1)")
                    self.state = 302
                    _la = self._input.LA(1)
                    if not(_la==35 or _la==36):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 303
                    self.mexpr(0) 
                self.state = 308
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,16,self._ctx)

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
        _startState = 74
        self.enterRecursionRule(localctx, 74, self.RULE_mexpr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 310
            self.uexpr()
            self._ctx.stop = self._input.LT(-1)
            self.state = 317
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,17,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = AtomicParser.MexprContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_mexpr)
                    self.state = 312
                    if not self.precpred(self._ctx, 1):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 1)")
                    self.state = 313
                    _la = self._input.LA(1)
                    if not(_la==33 or _la==34):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 314
                    self.uexpr() 
                self.state = 319
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,17,self._ctx)

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
        self.enterRule(localctx, 76, self.RULE_uexpr)
        self._la = 0 # Token type
        try:
            self.state = 323
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [13, 22, 23, 27, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69]:
                self.enterOuterAlt(localctx, 1)
                self.state = 320
                self.eexpr()
                pass
            elif token in [35, 36]:
                self.enterOuterAlt(localctx, 2)
                self.state = 321
                _la = self._input.LA(1)
                if not(_la==35 or _la==36):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 322
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

        def atom(self):
            return self.getTypedRuleContext(AtomicParser.AtomContext,0)


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
        self.enterRule(localctx, 78, self.RULE_eexpr)
        try:
            self.state = 330
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,19,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 325
                self.atom()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 326
                self.atom()
                self.state = 327
                self.match(AtomicParser.POWER)
                self.state = 328
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
        self._predicates[36] = self.aexpr_sempred
        self._predicates[37] = self.mexpr_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception("No predicate with index:" + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def expr_sempred(self, localctx:ExprContext, predIndex:int):
            if predIndex == 0:
                return self.precpred(self._ctx, 8)
         

    def aexpr_sempred(self, localctx:AexprContext, predIndex:int):
            if predIndex == 1:
                return self.precpred(self._ctx, 1)
         

    def mexpr_sempred(self, localctx:MexprContext, predIndex:int):
            if predIndex == 2:
                return self.precpred(self._ctx, 1)
         




