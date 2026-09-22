# Generated from AnalogParser.g4 by ANTLR 4.13.2
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
        4,1,82,314,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,20,
        7,20,2,21,7,21,2,22,7,22,2,23,7,23,2,24,7,24,2,25,7,25,2,26,7,26,
        2,27,7,27,2,28,7,28,2,29,7,29,2,30,7,30,2,31,7,31,2,32,7,32,2,33,
        7,33,2,34,7,34,1,0,1,0,1,0,1,1,1,1,1,1,1,1,5,1,78,8,1,10,1,12,1,
        81,9,1,1,1,3,1,84,8,1,1,2,1,2,1,2,1,2,1,2,1,2,3,2,92,8,2,1,3,1,3,
        1,3,1,3,1,3,1,3,1,3,1,3,3,3,102,8,3,1,3,1,3,1,3,1,3,1,3,3,3,109,
        8,3,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,4,1,5,1,5,1,6,1,6,1,7,1,7,1,7,
        1,7,1,7,1,7,3,7,129,8,7,1,8,1,8,1,8,1,8,1,9,1,9,1,10,1,10,3,10,139,
        8,10,1,10,1,10,5,10,143,8,10,10,10,12,10,146,9,10,1,10,3,10,149,
        8,10,1,10,1,10,1,11,1,11,1,11,1,11,1,11,1,12,1,12,1,12,3,12,161,
        8,12,1,12,3,12,164,8,12,1,13,1,13,1,14,1,14,3,14,170,8,14,1,15,1,
        15,1,16,1,16,1,17,1,17,1,18,1,18,1,19,1,19,1,19,3,19,183,8,19,1,
        20,1,20,1,20,5,20,188,8,20,10,20,12,20,191,9,20,1,21,1,21,1,21,3,
        21,196,8,21,1,21,1,21,1,22,1,22,3,22,202,8,22,1,22,3,22,205,8,22,
        1,23,1,23,1,23,1,23,1,23,1,23,3,23,213,8,23,1,24,1,24,1,24,1,24,
        1,25,1,25,1,25,1,25,1,25,1,25,5,25,225,8,25,10,25,12,25,228,9,25,
        1,26,1,26,1,26,3,26,233,8,26,1,27,1,27,1,27,1,27,1,27,1,27,5,27,
        241,8,27,10,27,12,27,244,9,27,1,28,1,28,1,28,1,28,1,28,1,28,5,28,
        252,8,28,10,28,12,28,255,9,28,1,29,1,29,1,29,1,29,1,29,1,29,5,29,
        263,8,29,10,29,12,29,266,9,29,1,30,1,30,1,30,1,30,1,30,1,30,5,30,
        274,8,30,10,30,12,30,277,9,30,1,31,1,31,1,31,1,31,1,31,1,31,5,31,
        285,8,31,10,31,12,31,288,9,31,1,32,1,32,1,32,1,32,1,32,1,32,5,32,
        296,8,32,10,32,12,32,299,9,32,1,33,1,33,1,33,1,33,1,33,1,33,5,33,
        307,8,33,10,33,12,33,310,9,33,1,34,1,34,1,34,0,8,50,54,56,58,60,
        62,64,66,35,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,
        38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,0,11,1,0,75,78,1,
        0,79,81,1,0,19,20,1,0,50,70,2,0,5,7,21,22,1,0,71,74,2,0,18,18,34,
        35,2,0,32,33,44,44,1,0,34,35,1,0,40,43,1,0,38,39,319,0,70,1,0,0,
        0,2,79,1,0,0,0,4,91,1,0,0,0,6,93,1,0,0,0,8,110,1,0,0,0,10,118,1,
        0,0,0,12,120,1,0,0,0,14,128,1,0,0,0,16,130,1,0,0,0,18,134,1,0,0,
        0,20,136,1,0,0,0,22,152,1,0,0,0,24,157,1,0,0,0,26,165,1,0,0,0,28,
        169,1,0,0,0,30,171,1,0,0,0,32,173,1,0,0,0,34,175,1,0,0,0,36,177,
        1,0,0,0,38,182,1,0,0,0,40,184,1,0,0,0,42,192,1,0,0,0,44,204,1,0,
        0,0,46,212,1,0,0,0,48,214,1,0,0,0,50,218,1,0,0,0,52,232,1,0,0,0,
        54,234,1,0,0,0,56,245,1,0,0,0,58,256,1,0,0,0,60,267,1,0,0,0,62,278,
        1,0,0,0,64,289,1,0,0,0,66,300,1,0,0,0,68,311,1,0,0,0,70,71,3,2,1,
        0,71,72,5,0,0,1,72,1,1,0,0,0,73,74,3,4,2,0,74,75,5,2,0,0,75,78,1,
        0,0,0,76,78,5,2,0,0,77,73,1,0,0,0,77,76,1,0,0,0,78,81,1,0,0,0,79,
        77,1,0,0,0,79,80,1,0,0,0,80,83,1,0,0,0,81,79,1,0,0,0,82,84,3,4,2,
        0,83,82,1,0,0,0,83,84,1,0,0,0,84,3,1,0,0,0,85,92,3,16,8,0,86,92,
        3,8,4,0,87,92,3,6,3,0,88,92,3,10,5,0,89,92,3,12,6,0,90,92,3,68,34,
        0,91,85,1,0,0,0,91,86,1,0,0,0,91,87,1,0,0,0,91,88,1,0,0,0,91,89,
        1,0,0,0,91,90,1,0,0,0,92,5,1,0,0,0,93,94,5,8,0,0,94,95,5,26,0,0,
        95,96,3,68,34,0,96,97,5,27,0,0,97,98,5,30,0,0,98,99,3,2,1,0,99,108,
        5,31,0,0,100,102,5,2,0,0,101,100,1,0,0,0,101,102,1,0,0,0,102,103,
        1,0,0,0,103,104,5,9,0,0,104,105,5,30,0,0,105,106,3,2,1,0,106,107,
        5,31,0,0,107,109,1,0,0,0,108,101,1,0,0,0,108,109,1,0,0,0,109,7,1,
        0,0,0,110,111,5,10,0,0,111,112,5,26,0,0,112,113,3,68,34,0,113,114,
        5,27,0,0,114,115,5,30,0,0,115,116,3,2,1,0,116,117,5,31,0,0,117,9,
        1,0,0,0,118,119,5,13,0,0,119,11,1,0,0,0,120,121,5,14,0,0,121,13,
        1,0,0,0,122,129,3,22,11,0,123,129,3,28,14,0,124,129,3,46,23,0,125,
        129,3,30,15,0,126,129,3,20,10,0,127,129,3,42,21,0,128,122,1,0,0,
        0,128,123,1,0,0,0,128,124,1,0,0,0,128,125,1,0,0,0,128,126,1,0,0,
        0,128,127,1,0,0,0,129,15,1,0,0,0,130,131,5,82,0,0,131,132,5,37,0,
        0,132,133,3,68,34,0,133,17,1,0,0,0,134,135,5,82,0,0,135,19,1,0,0,
        0,136,138,5,28,0,0,137,139,3,68,34,0,138,137,1,0,0,0,138,139,1,0,
        0,0,139,144,1,0,0,0,140,141,5,25,0,0,141,143,3,68,34,0,142,140,1,
        0,0,0,143,146,1,0,0,0,144,142,1,0,0,0,144,145,1,0,0,0,145,148,1,
        0,0,0,146,144,1,0,0,0,147,149,5,25,0,0,148,147,1,0,0,0,148,149,1,
        0,0,0,149,150,1,0,0,0,150,151,5,29,0,0,151,21,1,0,0,0,152,153,3,
        18,9,0,153,154,5,28,0,0,154,155,3,68,34,0,155,156,5,29,0,0,156,23,
        1,0,0,0,157,163,7,0,0,0,158,160,5,26,0,0,159,161,3,40,20,0,160,159,
        1,0,0,0,160,161,1,0,0,0,161,162,1,0,0,0,162,164,5,27,0,0,163,158,
        1,0,0,0,163,164,1,0,0,0,164,25,1,0,0,0,165,166,7,1,0,0,166,27,1,
        0,0,0,167,170,3,24,12,0,168,170,3,26,13,0,169,167,1,0,0,0,169,168,
        1,0,0,0,170,29,1,0,0,0,171,172,7,2,0,0,172,31,1,0,0,0,173,174,7,
        3,0,0,174,33,1,0,0,0,175,176,7,4,0,0,176,35,1,0,0,0,177,178,7,5,
        0,0,178,37,1,0,0,0,179,183,3,32,16,0,180,183,3,34,17,0,181,183,3,
        36,18,0,182,179,1,0,0,0,182,180,1,0,0,0,182,181,1,0,0,0,183,39,1,
        0,0,0,184,189,3,68,34,0,185,186,5,25,0,0,186,188,3,68,34,0,187,185,
        1,0,0,0,188,191,1,0,0,0,189,187,1,0,0,0,189,190,1,0,0,0,190,41,1,
        0,0,0,191,189,1,0,0,0,192,193,3,38,19,0,193,195,5,26,0,0,194,196,
        3,40,20,0,195,194,1,0,0,0,195,196,1,0,0,0,196,197,1,0,0,0,197,198,
        5,27,0,0,198,43,1,0,0,0,199,205,5,48,0,0,200,202,5,48,0,0,201,200,
        1,0,0,0,201,202,1,0,0,0,202,203,1,0,0,0,203,205,5,49,0,0,204,199,
        1,0,0,0,204,201,1,0,0,0,205,45,1,0,0,0,206,213,5,45,0,0,207,213,
        5,46,0,0,208,213,5,47,0,0,209,213,3,44,22,0,210,213,3,18,9,0,211,
        213,3,48,24,0,212,206,1,0,0,0,212,207,1,0,0,0,212,208,1,0,0,0,212,
        209,1,0,0,0,212,210,1,0,0,0,212,211,1,0,0,0,213,47,1,0,0,0,214,215,
        5,26,0,0,215,216,3,68,34,0,216,217,5,27,0,0,217,49,1,0,0,0,218,219,
        6,25,-1,0,219,220,3,14,7,0,220,226,1,0,0,0,221,222,10,1,0,0,222,
        223,5,36,0,0,223,225,3,14,7,0,224,221,1,0,0,0,225,228,1,0,0,0,226,
        224,1,0,0,0,226,227,1,0,0,0,227,51,1,0,0,0,228,226,1,0,0,0,229,233,
        3,50,25,0,230,231,7,6,0,0,231,233,3,50,25,0,232,229,1,0,0,0,232,
        230,1,0,0,0,233,53,1,0,0,0,234,235,6,27,-1,0,235,236,3,52,26,0,236,
        242,1,0,0,0,237,238,10,1,0,0,238,239,7,7,0,0,239,241,3,52,26,0,240,
        237,1,0,0,0,241,244,1,0,0,0,242,240,1,0,0,0,242,243,1,0,0,0,243,
        55,1,0,0,0,244,242,1,0,0,0,245,246,6,28,-1,0,246,247,3,54,27,0,247,
        253,1,0,0,0,248,249,10,1,0,0,249,250,7,8,0,0,250,252,3,54,27,0,251,
        248,1,0,0,0,252,255,1,0,0,0,253,251,1,0,0,0,253,254,1,0,0,0,254,
        57,1,0,0,0,255,253,1,0,0,0,256,257,6,29,-1,0,257,258,3,56,28,0,258,
        264,1,0,0,0,259,260,10,1,0,0,260,261,7,9,0,0,261,263,3,56,28,0,262,
        259,1,0,0,0,263,266,1,0,0,0,264,262,1,0,0,0,264,265,1,0,0,0,265,
        59,1,0,0,0,266,264,1,0,0,0,267,268,6,30,-1,0,268,269,3,58,29,0,269,
        275,1,0,0,0,270,271,10,1,0,0,271,272,7,10,0,0,272,274,3,58,29,0,
        273,270,1,0,0,0,274,277,1,0,0,0,275,273,1,0,0,0,275,276,1,0,0,0,
        276,61,1,0,0,0,277,275,1,0,0,0,278,279,6,31,-1,0,279,280,3,60,30,
        0,280,286,1,0,0,0,281,282,10,1,0,0,282,283,5,15,0,0,283,285,3,60,
        30,0,284,281,1,0,0,0,285,288,1,0,0,0,286,284,1,0,0,0,286,287,1,0,
        0,0,287,63,1,0,0,0,288,286,1,0,0,0,289,290,6,32,-1,0,290,291,3,62,
        31,0,291,297,1,0,0,0,292,293,10,1,0,0,293,294,5,17,0,0,294,296,3,
        62,31,0,295,292,1,0,0,0,296,299,1,0,0,0,297,295,1,0,0,0,297,298,
        1,0,0,0,298,65,1,0,0,0,299,297,1,0,0,0,300,301,6,33,-1,0,301,302,
        3,64,32,0,302,308,1,0,0,0,303,304,10,1,0,0,304,305,5,16,0,0,305,
        307,3,64,32,0,306,303,1,0,0,0,307,310,1,0,0,0,308,306,1,0,0,0,308,
        309,1,0,0,0,309,67,1,0,0,0,310,308,1,0,0,0,311,312,3,66,33,0,312,
        69,1,0,0,0,28,77,79,83,91,101,108,128,138,144,148,160,163,169,182,
        189,195,201,204,212,226,232,242,253,264,275,286,297,308
    ]

class AnalogParser ( Parser ):

    grammarFileName = "AnalogParser.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                     "<INVALID>", "'evolve'", "'measure'", "'initialize'", 
                     "'if'", "'else'", "'while'", "'with'", "'for'", "'break'", 
                     "'continue'", "<INVALID>", "<INVALID>", "<INVALID>", 
                     "<INVALID>", "'true'", "'false'", "'qreg'", "'qmode'", 
                     "':'", "';'", "','", "'('", "')'", "'['", "']'", "'{'", 
                     "'}'", "'*'", "'/'", "'+'", "'-'", "'^'", "'='", "'=='", 
                     "'!='", "'<'", "'<='", "'>'", "'>='", "'@'", "<INVALID>", 
                     "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                     "'abs'", "'sin'", "'cos'", "'tan'", "'exp'", "'log'", 
                     "'sinh'", "'cosh'", "'tanh'", "'atan'", "'acos'", "'asin'", 
                     "'atanh'", "'asinh'", "'acosh'", "'heaviside'", "'conj'", 
                     "'real'", "'imag'", "'atan2'", "'round'", "'range'", 
                     "'print'", "'len'", "'flatten'", "'%I'", "'%X'", "'%Y'", 
                     "'%Z'", "'%C'", "'%A'", "'%J'" ]

    symbolicNames = [ "<INVALID>", "WHITESPACE", "EOL", "NEWLINE", "COMMENT", 
                      "EVOLVE", "MEASURE", "INITIALIZE", "IF", "ELSE", "WHILE", 
                      "WITH", "FOR", "BREAK", "CONTINUE", "AND", "OR", "XOR", 
                      "NOT", "TRUE", "FALSE", "QUANTUMREGISTER", "MODEREGISTER", 
                      "COLON", "SEMICOLON", "COMMA", "LBRACKET", "RBRACKET", 
                      "SQUARELBRACKET", "SQUARERBRACKET", "LBRACE", "RBRACE", 
                      "MULT", "DIV", "PLUS", "MINUS", "POWER", "ASSIGN", 
                      "EQ", "NEQ", "LT", "LEQ", "GT", "GEQ", "AT", "INT", 
                      "FLOAT", "MATH_VAR", "REAL_PART", "IMAG_PART", "ABS", 
                      "SIN", "COS", "TAN", "EXP", "LOG", "SINH", "COSH", 
                      "TANH", "ATAN", "ACOS", "ASIN", "ATANH", "ASINH", 
                      "ACOSH", "HEAVISIDE", "CONJ", "REAL", "IMAG_FN", "ATAN2", 
                      "ROUND", "RANGE", "PRINT", "LENGTH", "FLATTEN", "PAULI_I", 
                      "PAULI_X", "PAULI_Y", "PAULI_Z", "CREATION", "ANNIHILATION", 
                      "IDENTITY_OP", "ID" ]

    RULE_program = 0
    RULE_block = 1
    RULE_statement = 2
    RULE_ifelse_stmt = 3
    RULE_while_stmt = 4
    RULE_break_stmt = 5
    RULE_continue_stmt = 6
    RULE_terminal = 7
    RULE_declaration = 8
    RULE_access = 9
    RULE_analog_list = 10
    RULE_analog_list_extract = 11
    RULE_pauli_op = 12
    RULE_ladder_op = 13
    RULE_operator_terminal = 14
    RULE_bool_literal = 15
    RULE_math_func = 16
    RULE_quantum_func = 17
    RULE_list_func = 18
    RULE_func_names = 19
    RULE_args = 20
    RULE_func = 21
    RULE_complex = 22
    RULE_math_terminal = 23
    RULE_pexpr = 24
    RULE_eexpr = 25
    RULE_uexpr = 26
    RULE_mexpr = 27
    RULE_aexpr = 28
    RULE_cexpr = 29
    RULE_eqexpr = 30
    RULE_andexpr = 31
    RULE_xorexpr = 32
    RULE_orexpr = 33
    RULE_expr = 34

    ruleNames =  [ "program", "block", "statement", "ifelse_stmt", "while_stmt", 
                   "break_stmt", "continue_stmt", "terminal", "declaration", 
                   "access", "analog_list", "analog_list_extract", "pauli_op", 
                   "ladder_op", "operator_terminal", "bool_literal", "math_func", 
                   "quantum_func", "list_func", "func_names", "args", "func", 
                   "complex", "math_terminal", "pexpr", "eexpr", "uexpr", 
                   "mexpr", "aexpr", "cexpr", "eqexpr", "andexpr", "xorexpr", 
                   "orexpr", "expr" ]

    EOF = Token.EOF
    WHITESPACE=1
    EOL=2
    NEWLINE=3
    COMMENT=4
    EVOLVE=5
    MEASURE=6
    INITIALIZE=7
    IF=8
    ELSE=9
    WHILE=10
    WITH=11
    FOR=12
    BREAK=13
    CONTINUE=14
    AND=15
    OR=16
    XOR=17
    NOT=18
    TRUE=19
    FALSE=20
    QUANTUMREGISTER=21
    MODEREGISTER=22
    COLON=23
    SEMICOLON=24
    COMMA=25
    LBRACKET=26
    RBRACKET=27
    SQUARELBRACKET=28
    SQUARERBRACKET=29
    LBRACE=30
    RBRACE=31
    MULT=32
    DIV=33
    PLUS=34
    MINUS=35
    POWER=36
    ASSIGN=37
    EQ=38
    NEQ=39
    LT=40
    LEQ=41
    GT=42
    GEQ=43
    AT=44
    INT=45
    FLOAT=46
    MATH_VAR=47
    REAL_PART=48
    IMAG_PART=49
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
    ROUND=70
    RANGE=71
    PRINT=72
    LENGTH=73
    FLATTEN=74
    PAULI_I=75
    PAULI_X=76
    PAULI_Y=77
    PAULI_Z=78
    CREATION=79
    ANNIHILATION=80
    IDENTITY_OP=81
    ID=82

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
            return self.getTypedRuleContext(AnalogParser.BlockContext,0)


        def EOF(self):
            return self.getToken(AnalogParser.EOF, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_program

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

        localctx = AnalogParser.ProgramContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_program)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 70
            self.block()
            self.state = 71
            self.match(AnalogParser.EOF)
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
                return self.getTypedRuleContexts(AnalogParser.StatementContext)
            else:
                return self.getTypedRuleContext(AnalogParser.StatementContext,i)


        def EOL(self, i:int=None):
            if i is None:
                return self.getTokens(AnalogParser.EOL)
            else:
                return self.getToken(AnalogParser.EOL, i)

        def getRuleIndex(self):
            return AnalogParser.RULE_block

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

        localctx = AnalogParser.BlockContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_block)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 79
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,1,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    self.state = 77
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [5, 6, 7, 8, 10, 13, 14, 18, 19, 20, 21, 22, 26, 28, 34, 35, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82]:
                        self.state = 73
                        self.statement()
                        self.state = 74
                        self.match(AnalogParser.EOL)
                        pass
                    elif token in [2]:
                        self.state = 76
                        self.match(AnalogParser.EOL)
                        pass
                    else:
                        raise NoViableAltException(self)
             
                self.state = 81
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,1,self._ctx)

            self.state = 83
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if (((_la) & ~0x3f) == 0 and ((1 << _la) & -35132488784416) != 0) or ((((_la - 64)) & ~0x3f) == 0 and ((1 << (_la - 64)) & 524287) != 0):
                self.state = 82
                self.statement()


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
            return self.getTypedRuleContext(AnalogParser.DeclarationContext,0)


        def while_stmt(self):
            return self.getTypedRuleContext(AnalogParser.While_stmtContext,0)


        def ifelse_stmt(self):
            return self.getTypedRuleContext(AnalogParser.Ifelse_stmtContext,0)


        def break_stmt(self):
            return self.getTypedRuleContext(AnalogParser.Break_stmtContext,0)


        def continue_stmt(self):
            return self.getTypedRuleContext(AnalogParser.Continue_stmtContext,0)


        def expr(self):
            return self.getTypedRuleContext(AnalogParser.ExprContext,0)


        def getRuleIndex(self):
            return AnalogParser.RULE_statement

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

        localctx = AnalogParser.StatementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_statement)
        try:
            self.state = 91
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,3,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 85
                self.declaration()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 86
                self.while_stmt()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 87
                self.ifelse_stmt()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 88
                self.break_stmt()
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 89
                self.continue_stmt()
                pass

            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 90
                self.expr()
                pass


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
            return self.getToken(AnalogParser.IF, 0)

        def LBRACKET(self):
            return self.getToken(AnalogParser.LBRACKET, 0)

        def expr(self):
            return self.getTypedRuleContext(AnalogParser.ExprContext,0)


        def RBRACKET(self):
            return self.getToken(AnalogParser.RBRACKET, 0)

        def LBRACE(self, i:int=None):
            if i is None:
                return self.getTokens(AnalogParser.LBRACE)
            else:
                return self.getToken(AnalogParser.LBRACE, i)

        def block(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AnalogParser.BlockContext)
            else:
                return self.getTypedRuleContext(AnalogParser.BlockContext,i)


        def RBRACE(self, i:int=None):
            if i is None:
                return self.getTokens(AnalogParser.RBRACE)
            else:
                return self.getToken(AnalogParser.RBRACE, i)

        def ELSE(self):
            return self.getToken(AnalogParser.ELSE, 0)

        def EOL(self):
            return self.getToken(AnalogParser.EOL, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_ifelse_stmt

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

        localctx = AnalogParser.Ifelse_stmtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_ifelse_stmt)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 93
            self.match(AnalogParser.IF)
            self.state = 94
            self.match(AnalogParser.LBRACKET)
            self.state = 95
            self.expr()
            self.state = 96
            self.match(AnalogParser.RBRACKET)
            self.state = 97
            self.match(AnalogParser.LBRACE)
            self.state = 98
            self.block()
            self.state = 99
            self.match(AnalogParser.RBRACE)
            self.state = 108
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,5,self._ctx)
            if la_ == 1:
                self.state = 101
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==2:
                    self.state = 100
                    self.match(AnalogParser.EOL)


                self.state = 103
                self.match(AnalogParser.ELSE)
                self.state = 104
                self.match(AnalogParser.LBRACE)
                self.state = 105
                self.block()
                self.state = 106
                self.match(AnalogParser.RBRACE)


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
            return self.getToken(AnalogParser.WHILE, 0)

        def LBRACKET(self):
            return self.getToken(AnalogParser.LBRACKET, 0)

        def expr(self):
            return self.getTypedRuleContext(AnalogParser.ExprContext,0)


        def RBRACKET(self):
            return self.getToken(AnalogParser.RBRACKET, 0)

        def LBRACE(self):
            return self.getToken(AnalogParser.LBRACE, 0)

        def block(self):
            return self.getTypedRuleContext(AnalogParser.BlockContext,0)


        def RBRACE(self):
            return self.getToken(AnalogParser.RBRACE, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_while_stmt

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

        localctx = AnalogParser.While_stmtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_while_stmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 110
            self.match(AnalogParser.WHILE)
            self.state = 111
            self.match(AnalogParser.LBRACKET)
            self.state = 112
            self.expr()
            self.state = 113
            self.match(AnalogParser.RBRACKET)
            self.state = 114
            self.match(AnalogParser.LBRACE)
            self.state = 115
            self.block()
            self.state = 116
            self.match(AnalogParser.RBRACE)
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
            return self.getToken(AnalogParser.BREAK, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_break_stmt

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

        localctx = AnalogParser.Break_stmtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_break_stmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 118
            self.match(AnalogParser.BREAK)
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
            return self.getToken(AnalogParser.CONTINUE, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_continue_stmt

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

        localctx = AnalogParser.Continue_stmtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_continue_stmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 120
            self.match(AnalogParser.CONTINUE)
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

        def analog_list_extract(self):
            return self.getTypedRuleContext(AnalogParser.Analog_list_extractContext,0)


        def operator_terminal(self):
            return self.getTypedRuleContext(AnalogParser.Operator_terminalContext,0)


        def math_terminal(self):
            return self.getTypedRuleContext(AnalogParser.Math_terminalContext,0)


        def bool_literal(self):
            return self.getTypedRuleContext(AnalogParser.Bool_literalContext,0)


        def analog_list(self):
            return self.getTypedRuleContext(AnalogParser.Analog_listContext,0)


        def func(self):
            return self.getTypedRuleContext(AnalogParser.FuncContext,0)


        def getRuleIndex(self):
            return AnalogParser.RULE_terminal

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

        localctx = AnalogParser.TerminalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_terminal)
        try:
            self.state = 128
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,6,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 122
                self.analog_list_extract()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 123
                self.operator_terminal()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 124
                self.math_terminal()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 125
                self.bool_literal()
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 126
                self.analog_list()
                pass

            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 127
                self.func()
                pass


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
            return self.getToken(AnalogParser.ID, 0)

        def ASSIGN(self):
            return self.getToken(AnalogParser.ASSIGN, 0)

        def expr(self):
            return self.getTypedRuleContext(AnalogParser.ExprContext,0)


        def getRuleIndex(self):
            return AnalogParser.RULE_declaration

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

        localctx = AnalogParser.DeclarationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_declaration)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 130
            self.match(AnalogParser.ID)
            self.state = 131
            self.match(AnalogParser.ASSIGN)
            self.state = 132
            self.expr()
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
            return self.getToken(AnalogParser.ID, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_access

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

        localctx = AnalogParser.AccessContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_access)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 134
            self.match(AnalogParser.ID)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Analog_listContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def SQUARELBRACKET(self):
            return self.getToken(AnalogParser.SQUARELBRACKET, 0)

        def SQUARERBRACKET(self):
            return self.getToken(AnalogParser.SQUARERBRACKET, 0)

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AnalogParser.ExprContext)
            else:
                return self.getTypedRuleContext(AnalogParser.ExprContext,i)


        def COMMA(self, i:int=None):
            if i is None:
                return self.getTokens(AnalogParser.COMMA)
            else:
                return self.getToken(AnalogParser.COMMA, i)

        def getRuleIndex(self):
            return AnalogParser.RULE_analog_list

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAnalog_list" ):
                listener.enterAnalog_list(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAnalog_list" ):
                listener.exitAnalog_list(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAnalog_list" ):
                return visitor.visitAnalog_list(self)
            else:
                return visitor.visitChildren(self)




    def analog_list(self):

        localctx = AnalogParser.Analog_listContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_analog_list)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 136
            self.match(AnalogParser.SQUARELBRACKET)
            self.state = 138
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if (((_la) & ~0x3f) == 0 and ((1 << _la) & -35132488810272) != 0) or ((((_la - 64)) & ~0x3f) == 0 and ((1 << (_la - 64)) & 524287) != 0):
                self.state = 137
                self.expr()


            self.state = 144
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,8,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    self.state = 140
                    self.match(AnalogParser.COMMA)
                    self.state = 141
                    self.expr() 
                self.state = 146
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,8,self._ctx)

            self.state = 148
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==25:
                self.state = 147
                self.match(AnalogParser.COMMA)


            self.state = 150
            self.match(AnalogParser.SQUARERBRACKET)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Analog_list_extractContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def access(self):
            return self.getTypedRuleContext(AnalogParser.AccessContext,0)


        def SQUARELBRACKET(self):
            return self.getToken(AnalogParser.SQUARELBRACKET, 0)

        def expr(self):
            return self.getTypedRuleContext(AnalogParser.ExprContext,0)


        def SQUARERBRACKET(self):
            return self.getToken(AnalogParser.SQUARERBRACKET, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_analog_list_extract

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAnalog_list_extract" ):
                listener.enterAnalog_list_extract(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAnalog_list_extract" ):
                listener.exitAnalog_list_extract(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAnalog_list_extract" ):
                return visitor.visitAnalog_list_extract(self)
            else:
                return visitor.visitChildren(self)




    def analog_list_extract(self):

        localctx = AnalogParser.Analog_list_extractContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_analog_list_extract)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 152
            self.access()
            self.state = 153
            self.match(AnalogParser.SQUARELBRACKET)
            self.state = 154
            self.expr()
            self.state = 155
            self.match(AnalogParser.SQUARERBRACKET)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Pauli_opContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def PAULI_I(self):
            return self.getToken(AnalogParser.PAULI_I, 0)

        def PAULI_X(self):
            return self.getToken(AnalogParser.PAULI_X, 0)

        def PAULI_Y(self):
            return self.getToken(AnalogParser.PAULI_Y, 0)

        def PAULI_Z(self):
            return self.getToken(AnalogParser.PAULI_Z, 0)

        def LBRACKET(self):
            return self.getToken(AnalogParser.LBRACKET, 0)

        def RBRACKET(self):
            return self.getToken(AnalogParser.RBRACKET, 0)

        def args(self):
            return self.getTypedRuleContext(AnalogParser.ArgsContext,0)


        def getRuleIndex(self):
            return AnalogParser.RULE_pauli_op

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterPauli_op" ):
                listener.enterPauli_op(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitPauli_op" ):
                listener.exitPauli_op(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPauli_op" ):
                return visitor.visitPauli_op(self)
            else:
                return visitor.visitChildren(self)




    def pauli_op(self):

        localctx = AnalogParser.Pauli_opContext(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_pauli_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 157
            _la = self._input.LA(1)
            if not(((((_la - 75)) & ~0x3f) == 0 and ((1 << (_la - 75)) & 15) != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 163
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,11,self._ctx)
            if la_ == 1:
                self.state = 158
                self.match(AnalogParser.LBRACKET)
                self.state = 160
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if (((_la) & ~0x3f) == 0 and ((1 << _la) & -35132488810272) != 0) or ((((_la - 64)) & ~0x3f) == 0 and ((1 << (_la - 64)) & 524287) != 0):
                    self.state = 159
                    self.args()


                self.state = 162
                self.match(AnalogParser.RBRACKET)


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Ladder_opContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def CREATION(self):
            return self.getToken(AnalogParser.CREATION, 0)

        def ANNIHILATION(self):
            return self.getToken(AnalogParser.ANNIHILATION, 0)

        def IDENTITY_OP(self):
            return self.getToken(AnalogParser.IDENTITY_OP, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_ladder_op

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterLadder_op" ):
                listener.enterLadder_op(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitLadder_op" ):
                listener.exitLadder_op(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitLadder_op" ):
                return visitor.visitLadder_op(self)
            else:
                return visitor.visitChildren(self)




    def ladder_op(self):

        localctx = AnalogParser.Ladder_opContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_ladder_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 165
            _la = self._input.LA(1)
            if not(((((_la - 79)) & ~0x3f) == 0 and ((1 << (_la - 79)) & 7) != 0)):
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


    class Operator_terminalContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def pauli_op(self):
            return self.getTypedRuleContext(AnalogParser.Pauli_opContext,0)


        def ladder_op(self):
            return self.getTypedRuleContext(AnalogParser.Ladder_opContext,0)


        def getRuleIndex(self):
            return AnalogParser.RULE_operator_terminal

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterOperator_terminal" ):
                listener.enterOperator_terminal(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitOperator_terminal" ):
                listener.exitOperator_terminal(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitOperator_terminal" ):
                return visitor.visitOperator_terminal(self)
            else:
                return visitor.visitChildren(self)




    def operator_terminal(self):

        localctx = AnalogParser.Operator_terminalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_operator_terminal)
        try:
            self.state = 169
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [75, 76, 77, 78]:
                self.enterOuterAlt(localctx, 1)
                self.state = 167
                self.pauli_op()
                pass
            elif token in [79, 80, 81]:
                self.enterOuterAlt(localctx, 2)
                self.state = 168
                self.ladder_op()
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


    class Bool_literalContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def TRUE(self):
            return self.getToken(AnalogParser.TRUE, 0)

        def FALSE(self):
            return self.getToken(AnalogParser.FALSE, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_bool_literal

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

        localctx = AnalogParser.Bool_literalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_bool_literal)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 171
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


    class Math_funcContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ABS(self):
            return self.getToken(AnalogParser.ABS, 0)

        def SIN(self):
            return self.getToken(AnalogParser.SIN, 0)

        def COS(self):
            return self.getToken(AnalogParser.COS, 0)

        def TAN(self):
            return self.getToken(AnalogParser.TAN, 0)

        def EXP(self):
            return self.getToken(AnalogParser.EXP, 0)

        def LOG(self):
            return self.getToken(AnalogParser.LOG, 0)

        def SINH(self):
            return self.getToken(AnalogParser.SINH, 0)

        def COSH(self):
            return self.getToken(AnalogParser.COSH, 0)

        def TANH(self):
            return self.getToken(AnalogParser.TANH, 0)

        def ATAN(self):
            return self.getToken(AnalogParser.ATAN, 0)

        def ACOS(self):
            return self.getToken(AnalogParser.ACOS, 0)

        def ASIN(self):
            return self.getToken(AnalogParser.ASIN, 0)

        def ATANH(self):
            return self.getToken(AnalogParser.ATANH, 0)

        def ASINH(self):
            return self.getToken(AnalogParser.ASINH, 0)

        def ACOSH(self):
            return self.getToken(AnalogParser.ACOSH, 0)

        def ATAN2(self):
            return self.getToken(AnalogParser.ATAN2, 0)

        def CONJ(self):
            return self.getToken(AnalogParser.CONJ, 0)

        def HEAVISIDE(self):
            return self.getToken(AnalogParser.HEAVISIDE, 0)

        def REAL(self):
            return self.getToken(AnalogParser.REAL, 0)

        def IMAG_FN(self):
            return self.getToken(AnalogParser.IMAG_FN, 0)

        def ROUND(self):
            return self.getToken(AnalogParser.ROUND, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_math_func

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMath_func" ):
                listener.enterMath_func(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMath_func" ):
                listener.exitMath_func(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMath_func" ):
                return visitor.visitMath_func(self)
            else:
                return visitor.visitChildren(self)




    def math_func(self):

        localctx = AnalogParser.Math_funcContext(self, self._ctx, self.state)
        self.enterRule(localctx, 32, self.RULE_math_func)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 173
            _la = self._input.LA(1)
            if not(((((_la - 50)) & ~0x3f) == 0 and ((1 << (_la - 50)) & 2097151) != 0)):
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


    class Quantum_funcContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def QUANTUMREGISTER(self):
            return self.getToken(AnalogParser.QUANTUMREGISTER, 0)

        def MODEREGISTER(self):
            return self.getToken(AnalogParser.MODEREGISTER, 0)

        def EVOLVE(self):
            return self.getToken(AnalogParser.EVOLVE, 0)

        def MEASURE(self):
            return self.getToken(AnalogParser.MEASURE, 0)

        def INITIALIZE(self):
            return self.getToken(AnalogParser.INITIALIZE, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_quantum_func

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterQuantum_func" ):
                listener.enterQuantum_func(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitQuantum_func" ):
                listener.exitQuantum_func(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitQuantum_func" ):
                return visitor.visitQuantum_func(self)
            else:
                return visitor.visitChildren(self)




    def quantum_func(self):

        localctx = AnalogParser.Quantum_funcContext(self, self._ctx, self.state)
        self.enterRule(localctx, 34, self.RULE_quantum_func)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 175
            _la = self._input.LA(1)
            if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 6291680) != 0)):
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


    class List_funcContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def RANGE(self):
            return self.getToken(AnalogParser.RANGE, 0)

        def PRINT(self):
            return self.getToken(AnalogParser.PRINT, 0)

        def LENGTH(self):
            return self.getToken(AnalogParser.LENGTH, 0)

        def FLATTEN(self):
            return self.getToken(AnalogParser.FLATTEN, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_list_func

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterList_func" ):
                listener.enterList_func(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitList_func" ):
                listener.exitList_func(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitList_func" ):
                return visitor.visitList_func(self)
            else:
                return visitor.visitChildren(self)




    def list_func(self):

        localctx = AnalogParser.List_funcContext(self, self._ctx, self.state)
        self.enterRule(localctx, 36, self.RULE_list_func)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 177
            _la = self._input.LA(1)
            if not(((((_la - 71)) & ~0x3f) == 0 and ((1 << (_la - 71)) & 15) != 0)):
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


    class Func_namesContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def math_func(self):
            return self.getTypedRuleContext(AnalogParser.Math_funcContext,0)


        def quantum_func(self):
            return self.getTypedRuleContext(AnalogParser.Quantum_funcContext,0)


        def list_func(self):
            return self.getTypedRuleContext(AnalogParser.List_funcContext,0)


        def getRuleIndex(self):
            return AnalogParser.RULE_func_names

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

        localctx = AnalogParser.Func_namesContext(self, self._ctx, self.state)
        self.enterRule(localctx, 38, self.RULE_func_names)
        try:
            self.state = 182
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]:
                self.enterOuterAlt(localctx, 1)
                self.state = 179
                self.math_func()
                pass
            elif token in [5, 6, 7, 21, 22]:
                self.enterOuterAlt(localctx, 2)
                self.state = 180
                self.quantum_func()
                pass
            elif token in [71, 72, 73, 74]:
                self.enterOuterAlt(localctx, 3)
                self.state = 181
                self.list_func()
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


    class ArgsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AnalogParser.ExprContext)
            else:
                return self.getTypedRuleContext(AnalogParser.ExprContext,i)


        def COMMA(self, i:int=None):
            if i is None:
                return self.getTokens(AnalogParser.COMMA)
            else:
                return self.getToken(AnalogParser.COMMA, i)

        def getRuleIndex(self):
            return AnalogParser.RULE_args

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterArgs" ):
                listener.enterArgs(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitArgs" ):
                listener.exitArgs(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitArgs" ):
                return visitor.visitArgs(self)
            else:
                return visitor.visitChildren(self)




    def args(self):

        localctx = AnalogParser.ArgsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 40, self.RULE_args)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 184
            self.expr()
            self.state = 189
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==25:
                self.state = 185
                self.match(AnalogParser.COMMA)
                self.state = 186
                self.expr()
                self.state = 191
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class FuncContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def func_names(self):
            return self.getTypedRuleContext(AnalogParser.Func_namesContext,0)


        def LBRACKET(self):
            return self.getToken(AnalogParser.LBRACKET, 0)

        def RBRACKET(self):
            return self.getToken(AnalogParser.RBRACKET, 0)

        def args(self):
            return self.getTypedRuleContext(AnalogParser.ArgsContext,0)


        def getRuleIndex(self):
            return AnalogParser.RULE_func

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterFunc" ):
                listener.enterFunc(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitFunc" ):
                listener.exitFunc(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitFunc" ):
                return visitor.visitFunc(self)
            else:
                return visitor.visitChildren(self)




    def func(self):

        localctx = AnalogParser.FuncContext(self, self._ctx, self.state)
        self.enterRule(localctx, 42, self.RULE_func)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 192
            self.func_names()
            self.state = 193
            self.match(AnalogParser.LBRACKET)
            self.state = 195
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if (((_la) & ~0x3f) == 0 and ((1 << _la) & -35132488810272) != 0) or ((((_la - 64)) & ~0x3f) == 0 and ((1 << (_la - 64)) & 524287) != 0):
                self.state = 194
                self.args()


            self.state = 197
            self.match(AnalogParser.RBRACKET)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ComplexContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def REAL_PART(self):
            return self.getToken(AnalogParser.REAL_PART, 0)

        def IMAG_PART(self):
            return self.getToken(AnalogParser.IMAG_PART, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_complex

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterComplex" ):
                listener.enterComplex(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitComplex" ):
                listener.exitComplex(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitComplex" ):
                return visitor.visitComplex(self)
            else:
                return visitor.visitChildren(self)




    def complex_(self):

        localctx = AnalogParser.ComplexContext(self, self._ctx, self.state)
        self.enterRule(localctx, 44, self.RULE_complex)
        self._la = 0 # Token type
        try:
            self.state = 204
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,17,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 199
                self.match(AnalogParser.REAL_PART)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 201
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==48:
                    self.state = 200
                    self.match(AnalogParser.REAL_PART)


                self.state = 203
                self.match(AnalogParser.IMAG_PART)
                pass


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
            return self.getToken(AnalogParser.INT, 0)

        def FLOAT(self):
            return self.getToken(AnalogParser.FLOAT, 0)

        def MATH_VAR(self):
            return self.getToken(AnalogParser.MATH_VAR, 0)

        def complex_(self):
            return self.getTypedRuleContext(AnalogParser.ComplexContext,0)


        def access(self):
            return self.getTypedRuleContext(AnalogParser.AccessContext,0)


        def pexpr(self):
            return self.getTypedRuleContext(AnalogParser.PexprContext,0)


        def getRuleIndex(self):
            return AnalogParser.RULE_math_terminal

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

        localctx = AnalogParser.Math_terminalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 46, self.RULE_math_terminal)
        try:
            self.state = 212
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [45]:
                self.enterOuterAlt(localctx, 1)
                self.state = 206
                self.match(AnalogParser.INT)
                pass
            elif token in [46]:
                self.enterOuterAlt(localctx, 2)
                self.state = 207
                self.match(AnalogParser.FLOAT)
                pass
            elif token in [47]:
                self.enterOuterAlt(localctx, 3)
                self.state = 208
                self.match(AnalogParser.MATH_VAR)
                pass
            elif token in [48, 49]:
                self.enterOuterAlt(localctx, 4)
                self.state = 209
                self.complex_()
                pass
            elif token in [82]:
                self.enterOuterAlt(localctx, 5)
                self.state = 210
                self.access()
                pass
            elif token in [26]:
                self.enterOuterAlt(localctx, 6)
                self.state = 211
                self.pexpr()
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


    class PexprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def LBRACKET(self):
            return self.getToken(AnalogParser.LBRACKET, 0)

        def expr(self):
            return self.getTypedRuleContext(AnalogParser.ExprContext,0)


        def RBRACKET(self):
            return self.getToken(AnalogParser.RBRACKET, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_pexpr

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

        localctx = AnalogParser.PexprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 48, self.RULE_pexpr)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 214
            self.match(AnalogParser.LBRACKET)
            self.state = 215
            self.expr()
            self.state = 216
            self.match(AnalogParser.RBRACKET)
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
            return self.getTypedRuleContext(AnalogParser.TerminalContext,0)


        def eexpr(self):
            return self.getTypedRuleContext(AnalogParser.EexprContext,0)


        def POWER(self):
            return self.getToken(AnalogParser.POWER, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_eexpr

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



    def eexpr(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = AnalogParser.EexprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 50
        self.enterRecursionRule(localctx, 50, self.RULE_eexpr, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 219
            self.terminal()
            self._ctx.stop = self._input.LT(-1)
            self.state = 226
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,19,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = AnalogParser.EexprContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_eexpr)
                    self.state = 221
                    if not self.precpred(self._ctx, 1):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 1)")
                    self.state = 222
                    self.match(AnalogParser.POWER)
                    self.state = 223
                    self.terminal() 
                self.state = 228
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,19,self._ctx)

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
            return self.getTypedRuleContext(AnalogParser.EexprContext,0)


        def PLUS(self):
            return self.getToken(AnalogParser.PLUS, 0)

        def MINUS(self):
            return self.getToken(AnalogParser.MINUS, 0)

        def NOT(self):
            return self.getToken(AnalogParser.NOT, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_uexpr

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

        localctx = AnalogParser.UexprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 52, self.RULE_uexpr)
        self._la = 0 # Token type
        try:
            self.state = 232
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [5, 6, 7, 19, 20, 21, 22, 26, 28, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82]:
                self.enterOuterAlt(localctx, 1)
                self.state = 229
                self.eexpr(0)
                pass
            elif token in [18, 34, 35]:
                self.enterOuterAlt(localctx, 2)
                self.state = 230
                _la = self._input.LA(1)
                if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 51539869696) != 0)):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 231
                self.eexpr(0)
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


    class MexprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def uexpr(self):
            return self.getTypedRuleContext(AnalogParser.UexprContext,0)


        def mexpr(self):
            return self.getTypedRuleContext(AnalogParser.MexprContext,0)


        def MULT(self):
            return self.getToken(AnalogParser.MULT, 0)

        def DIV(self):
            return self.getToken(AnalogParser.DIV, 0)

        def AT(self):
            return self.getToken(AnalogParser.AT, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_mexpr

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
        localctx = AnalogParser.MexprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 54
        self.enterRecursionRule(localctx, 54, self.RULE_mexpr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 235
            self.uexpr()
            self._ctx.stop = self._input.LT(-1)
            self.state = 242
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,21,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = AnalogParser.MexprContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_mexpr)
                    self.state = 237
                    if not self.precpred(self._ctx, 1):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 1)")
                    self.state = 238
                    _la = self._input.LA(1)
                    if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 17605070946304) != 0)):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 239
                    self.uexpr() 
                self.state = 244
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,21,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class AexprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def mexpr(self):
            return self.getTypedRuleContext(AnalogParser.MexprContext,0)


        def aexpr(self):
            return self.getTypedRuleContext(AnalogParser.AexprContext,0)


        def PLUS(self):
            return self.getToken(AnalogParser.PLUS, 0)

        def MINUS(self):
            return self.getToken(AnalogParser.MINUS, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_aexpr

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
        localctx = AnalogParser.AexprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 56
        self.enterRecursionRule(localctx, 56, self.RULE_aexpr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 246
            self.mexpr(0)
            self._ctx.stop = self._input.LT(-1)
            self.state = 253
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,22,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = AnalogParser.AexprContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_aexpr)
                    self.state = 248
                    if not self.precpred(self._ctx, 1):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 1)")
                    self.state = 249
                    _la = self._input.LA(1)
                    if not(_la==34 or _la==35):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 250
                    self.mexpr(0) 
                self.state = 255
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,22,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class CexprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def aexpr(self):
            return self.getTypedRuleContext(AnalogParser.AexprContext,0)


        def cexpr(self):
            return self.getTypedRuleContext(AnalogParser.CexprContext,0)


        def LT(self):
            return self.getToken(AnalogParser.LT, 0)

        def LEQ(self):
            return self.getToken(AnalogParser.LEQ, 0)

        def GT(self):
            return self.getToken(AnalogParser.GT, 0)

        def GEQ(self):
            return self.getToken(AnalogParser.GEQ, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_cexpr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterCexpr" ):
                listener.enterCexpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitCexpr" ):
                listener.exitCexpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitCexpr" ):
                return visitor.visitCexpr(self)
            else:
                return visitor.visitChildren(self)



    def cexpr(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = AnalogParser.CexprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 58
        self.enterRecursionRule(localctx, 58, self.RULE_cexpr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 257
            self.aexpr(0)
            self._ctx.stop = self._input.LT(-1)
            self.state = 264
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,23,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = AnalogParser.CexprContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_cexpr)
                    self.state = 259
                    if not self.precpred(self._ctx, 1):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 1)")
                    self.state = 260
                    _la = self._input.LA(1)
                    if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 16492674416640) != 0)):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 261
                    self.aexpr(0) 
                self.state = 266
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,23,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class EqexprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def cexpr(self):
            return self.getTypedRuleContext(AnalogParser.CexprContext,0)


        def eqexpr(self):
            return self.getTypedRuleContext(AnalogParser.EqexprContext,0)


        def EQ(self):
            return self.getToken(AnalogParser.EQ, 0)

        def NEQ(self):
            return self.getToken(AnalogParser.NEQ, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_eqexpr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterEqexpr" ):
                listener.enterEqexpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitEqexpr" ):
                listener.exitEqexpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitEqexpr" ):
                return visitor.visitEqexpr(self)
            else:
                return visitor.visitChildren(self)



    def eqexpr(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = AnalogParser.EqexprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 60
        self.enterRecursionRule(localctx, 60, self.RULE_eqexpr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 268
            self.cexpr(0)
            self._ctx.stop = self._input.LT(-1)
            self.state = 275
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,24,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = AnalogParser.EqexprContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_eqexpr)
                    self.state = 270
                    if not self.precpred(self._ctx, 1):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 1)")
                    self.state = 271
                    _la = self._input.LA(1)
                    if not(_la==38 or _la==39):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 272
                    self.cexpr(0) 
                self.state = 277
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,24,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class AndexprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def eqexpr(self):
            return self.getTypedRuleContext(AnalogParser.EqexprContext,0)


        def andexpr(self):
            return self.getTypedRuleContext(AnalogParser.AndexprContext,0)


        def AND(self):
            return self.getToken(AnalogParser.AND, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_andexpr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAndexpr" ):
                listener.enterAndexpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAndexpr" ):
                listener.exitAndexpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAndexpr" ):
                return visitor.visitAndexpr(self)
            else:
                return visitor.visitChildren(self)



    def andexpr(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = AnalogParser.AndexprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 62
        self.enterRecursionRule(localctx, 62, self.RULE_andexpr, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 279
            self.eqexpr(0)
            self._ctx.stop = self._input.LT(-1)
            self.state = 286
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,25,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = AnalogParser.AndexprContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_andexpr)
                    self.state = 281
                    if not self.precpred(self._ctx, 1):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 1)")
                    self.state = 282
                    self.match(AnalogParser.AND)
                    self.state = 283
                    self.eqexpr(0) 
                self.state = 288
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,25,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class XorexprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def andexpr(self):
            return self.getTypedRuleContext(AnalogParser.AndexprContext,0)


        def xorexpr(self):
            return self.getTypedRuleContext(AnalogParser.XorexprContext,0)


        def XOR(self):
            return self.getToken(AnalogParser.XOR, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_xorexpr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterXorexpr" ):
                listener.enterXorexpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitXorexpr" ):
                listener.exitXorexpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitXorexpr" ):
                return visitor.visitXorexpr(self)
            else:
                return visitor.visitChildren(self)



    def xorexpr(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = AnalogParser.XorexprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 64
        self.enterRecursionRule(localctx, 64, self.RULE_xorexpr, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 290
            self.andexpr(0)
            self._ctx.stop = self._input.LT(-1)
            self.state = 297
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,26,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = AnalogParser.XorexprContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_xorexpr)
                    self.state = 292
                    if not self.precpred(self._ctx, 1):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 1)")
                    self.state = 293
                    self.match(AnalogParser.XOR)
                    self.state = 294
                    self.andexpr(0) 
                self.state = 299
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,26,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class OrexprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def xorexpr(self):
            return self.getTypedRuleContext(AnalogParser.XorexprContext,0)


        def orexpr(self):
            return self.getTypedRuleContext(AnalogParser.OrexprContext,0)


        def OR(self):
            return self.getToken(AnalogParser.OR, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_orexpr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterOrexpr" ):
                listener.enterOrexpr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitOrexpr" ):
                listener.exitOrexpr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitOrexpr" ):
                return visitor.visitOrexpr(self)
            else:
                return visitor.visitChildren(self)



    def orexpr(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = AnalogParser.OrexprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 66
        self.enterRecursionRule(localctx, 66, self.RULE_orexpr, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 301
            self.xorexpr(0)
            self._ctx.stop = self._input.LT(-1)
            self.state = 308
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,27,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = AnalogParser.OrexprContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_orexpr)
                    self.state = 303
                    if not self.precpred(self._ctx, 1):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 1)")
                    self.state = 304
                    self.match(AnalogParser.OR)
                    self.state = 305
                    self.xorexpr(0) 
                self.state = 310
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,27,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class ExprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def orexpr(self):
            return self.getTypedRuleContext(AnalogParser.OrexprContext,0)


        def getRuleIndex(self):
            return AnalogParser.RULE_expr

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




    def expr(self):

        localctx = AnalogParser.ExprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 68, self.RULE_expr)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 311
            self.orexpr(0)
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
        self._predicates[25] = self.eexpr_sempred
        self._predicates[27] = self.mexpr_sempred
        self._predicates[28] = self.aexpr_sempred
        self._predicates[29] = self.cexpr_sempred
        self._predicates[30] = self.eqexpr_sempred
        self._predicates[31] = self.andexpr_sempred
        self._predicates[32] = self.xorexpr_sempred
        self._predicates[33] = self.orexpr_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception("No predicate with index:" + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def eexpr_sempred(self, localctx:EexprContext, predIndex:int):
            if predIndex == 0:
                return self.precpred(self._ctx, 1)
         

    def mexpr_sempred(self, localctx:MexprContext, predIndex:int):
            if predIndex == 1:
                return self.precpred(self._ctx, 1)
         

    def aexpr_sempred(self, localctx:AexprContext, predIndex:int):
            if predIndex == 2:
                return self.precpred(self._ctx, 1)
         

    def cexpr_sempred(self, localctx:CexprContext, predIndex:int):
            if predIndex == 3:
                return self.precpred(self._ctx, 1)
         

    def eqexpr_sempred(self, localctx:EqexprContext, predIndex:int):
            if predIndex == 4:
                return self.precpred(self._ctx, 1)
         

    def andexpr_sempred(self, localctx:AndexprContext, predIndex:int):
            if predIndex == 5:
                return self.precpred(self._ctx, 1)
         

    def xorexpr_sempred(self, localctx:XorexprContext, predIndex:int):
            if predIndex == 6:
                return self.precpred(self._ctx, 1)
         

    def orexpr_sempred(self, localctx:OrexprContext, predIndex:int):
            if predIndex == 7:
                return self.precpred(self._ctx, 1)
         




