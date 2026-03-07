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
        4,1,68,313,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,20,
        7,20,2,21,7,21,2,22,7,22,2,23,7,23,2,24,7,24,2,25,7,25,2,26,7,26,
        1,0,1,0,1,0,5,0,58,8,0,10,0,12,0,61,9,0,1,0,3,0,64,8,0,1,0,1,0,1,
        1,1,1,1,1,1,1,1,1,1,1,3,1,74,8,1,1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,
        1,3,1,3,1,3,1,3,3,3,88,8,3,1,4,1,4,1,4,1,4,1,4,1,5,1,5,1,5,1,5,1,
        5,1,6,1,6,3,6,102,8,6,1,6,1,6,5,6,106,8,6,10,6,12,6,109,9,6,1,6,
        1,6,1,7,1,7,1,8,1,8,1,8,1,8,1,8,1,9,1,9,1,9,1,9,1,9,1,10,1,10,1,
        10,1,11,1,11,1,11,1,12,1,12,1,13,1,13,1,13,1,13,1,13,1,13,3,13,139,
        8,13,1,13,1,13,1,13,5,13,144,8,13,10,13,12,13,147,9,13,1,13,3,13,
        150,8,13,1,13,1,13,1,13,1,13,3,13,156,8,13,1,13,1,13,1,13,5,13,161,
        8,13,10,13,12,13,164,9,13,1,13,3,13,167,8,13,1,13,3,13,170,8,13,
        1,14,1,14,1,14,1,14,1,14,1,14,3,14,178,8,14,1,14,1,14,1,14,5,14,
        183,8,14,10,14,12,14,186,9,14,1,14,3,14,189,8,14,1,14,1,14,1,15,
        1,15,1,16,1,16,1,17,1,17,1,18,1,18,1,18,1,18,1,18,1,18,1,18,1,18,
        1,18,3,18,208,8,18,1,18,1,18,1,18,1,18,1,18,1,18,1,18,1,18,5,18,
        218,8,18,10,18,12,18,221,9,18,1,19,1,19,1,20,1,20,1,21,1,21,1,21,
        1,21,1,21,1,21,1,21,1,21,1,21,1,21,1,21,3,21,238,8,21,1,21,1,21,
        1,21,1,21,1,21,1,21,1,21,1,21,1,21,1,21,1,21,1,21,1,21,1,21,1,21,
        5,21,255,8,21,10,21,12,21,258,9,21,1,22,1,22,3,22,262,8,22,1,23,
        1,23,1,23,1,23,1,23,1,23,1,23,1,23,1,23,3,23,273,8,23,1,23,1,23,
        1,23,1,23,1,23,1,23,1,23,1,23,1,23,1,23,1,23,1,23,1,23,1,23,1,23,
        5,23,290,8,23,10,23,12,23,293,9,23,1,24,1,24,1,25,1,25,1,26,1,26,
        1,26,1,26,1,26,1,26,1,26,1,26,1,26,1,26,1,26,1,26,3,26,311,8,26,
        1,26,0,3,36,42,46,27,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,
        32,34,36,38,40,42,44,46,48,50,52,0,8,1,0,13,14,1,0,15,16,1,0,17,
        18,1,0,24,27,1,0,21,23,1,0,40,41,2,0,44,47,68,68,1,0,48,66,333,0,
        59,1,0,0,0,2,73,1,0,0,0,4,75,1,0,0,0,6,87,1,0,0,0,8,89,1,0,0,0,10,
        94,1,0,0,0,12,99,1,0,0,0,14,112,1,0,0,0,16,114,1,0,0,0,18,119,1,
        0,0,0,20,124,1,0,0,0,22,127,1,0,0,0,24,130,1,0,0,0,26,132,1,0,0,
        0,28,171,1,0,0,0,30,192,1,0,0,0,32,194,1,0,0,0,34,196,1,0,0,0,36,
        207,1,0,0,0,38,222,1,0,0,0,40,224,1,0,0,0,42,237,1,0,0,0,44,261,
        1,0,0,0,46,272,1,0,0,0,48,294,1,0,0,0,50,296,1,0,0,0,52,310,1,0,
        0,0,54,55,3,2,1,0,55,56,5,2,0,0,56,58,1,0,0,0,57,54,1,0,0,0,58,61,
        1,0,0,0,59,57,1,0,0,0,59,60,1,0,0,0,60,63,1,0,0,0,61,59,1,0,0,0,
        62,64,3,2,1,0,63,62,1,0,0,0,63,64,1,0,0,0,64,65,1,0,0,0,65,66,5,
        0,0,1,66,1,1,0,0,0,67,74,3,4,2,0,68,74,3,18,9,0,69,74,3,20,10,0,
        70,74,3,22,11,0,71,74,3,26,13,0,72,74,3,28,14,0,73,67,1,0,0,0,73,
        68,1,0,0,0,73,69,1,0,0,0,73,70,1,0,0,0,73,71,1,0,0,0,73,72,1,0,0,
        0,74,3,1,0,0,0,75,76,5,68,0,0,76,77,5,43,0,0,77,78,3,6,3,0,78,5,
        1,0,0,0,79,88,3,10,5,0,80,88,3,8,4,0,81,88,3,16,8,0,82,88,3,12,6,
        0,83,88,3,14,7,0,84,88,3,36,18,0,85,88,3,42,21,0,86,88,3,46,23,0,
        87,79,1,0,0,0,87,80,1,0,0,0,87,81,1,0,0,0,87,82,1,0,0,0,87,83,1,
        0,0,0,87,84,1,0,0,0,87,85,1,0,0,0,87,86,1,0,0,0,88,7,1,0,0,0,89,
        90,5,19,0,0,90,91,5,31,0,0,91,92,5,44,0,0,92,93,5,32,0,0,93,9,1,
        0,0,0,94,95,5,20,0,0,95,96,5,31,0,0,96,97,5,44,0,0,97,98,5,32,0,
        0,98,11,1,0,0,0,99,101,5,33,0,0,100,102,3,6,3,0,101,100,1,0,0,0,
        101,102,1,0,0,0,102,107,1,0,0,0,103,104,5,30,0,0,104,106,3,6,3,0,
        105,103,1,0,0,0,106,109,1,0,0,0,107,105,1,0,0,0,107,108,1,0,0,0,
        108,110,1,0,0,0,109,107,1,0,0,0,110,111,5,34,0,0,111,13,1,0,0,0,
        112,113,5,68,0,0,113,15,1,0,0,0,114,115,3,14,7,0,115,116,5,33,0,
        0,116,117,5,44,0,0,117,118,5,34,0,0,118,17,1,0,0,0,119,120,5,5,0,
        0,120,121,3,24,12,0,121,122,5,11,0,0,122,123,3,6,3,0,123,19,1,0,
        0,0,124,125,5,6,0,0,125,126,3,24,12,0,126,21,1,0,0,0,127,128,5,7,
        0,0,128,129,3,24,12,0,129,23,1,0,0,0,130,131,3,6,3,0,131,25,1,0,
        0,0,132,133,5,8,0,0,133,134,5,31,0,0,134,135,3,36,18,0,135,136,5,
        32,0,0,136,138,5,35,0,0,137,139,5,2,0,0,138,137,1,0,0,0,138,139,
        1,0,0,0,139,145,1,0,0,0,140,141,3,2,1,0,141,142,5,2,0,0,142,144,
        1,0,0,0,143,140,1,0,0,0,144,147,1,0,0,0,145,143,1,0,0,0,145,146,
        1,0,0,0,146,149,1,0,0,0,147,145,1,0,0,0,148,150,3,2,1,0,149,148,
        1,0,0,0,149,150,1,0,0,0,150,151,1,0,0,0,151,169,5,36,0,0,152,153,
        5,9,0,0,153,155,5,35,0,0,154,156,5,2,0,0,155,154,1,0,0,0,155,156,
        1,0,0,0,156,162,1,0,0,0,157,158,3,2,1,0,158,159,5,2,0,0,159,161,
        1,0,0,0,160,157,1,0,0,0,161,164,1,0,0,0,162,160,1,0,0,0,162,163,
        1,0,0,0,163,166,1,0,0,0,164,162,1,0,0,0,165,167,3,2,1,0,166,165,
        1,0,0,0,166,167,1,0,0,0,167,168,1,0,0,0,168,170,5,36,0,0,169,152,
        1,0,0,0,169,170,1,0,0,0,170,27,1,0,0,0,171,172,5,10,0,0,172,173,
        5,31,0,0,173,174,3,36,18,0,174,175,5,32,0,0,175,177,5,35,0,0,176,
        178,5,2,0,0,177,176,1,0,0,0,177,178,1,0,0,0,178,184,1,0,0,0,179,
        180,3,2,1,0,180,181,5,2,0,0,181,183,1,0,0,0,182,179,1,0,0,0,183,
        186,1,0,0,0,184,182,1,0,0,0,184,185,1,0,0,0,185,188,1,0,0,0,186,
        184,1,0,0,0,187,189,3,2,1,0,188,187,1,0,0,0,188,189,1,0,0,0,189,
        190,1,0,0,0,190,191,5,36,0,0,191,29,1,0,0,0,192,193,7,0,0,0,193,
        31,1,0,0,0,194,195,7,1,0,0,195,33,1,0,0,0,196,197,7,2,0,0,197,35,
        1,0,0,0,198,199,6,18,-1,0,199,200,3,34,17,0,200,201,3,36,18,3,201,
        208,1,0,0,0,202,208,3,14,7,0,203,204,5,31,0,0,204,205,3,36,18,0,
        205,206,5,32,0,0,206,208,1,0,0,0,207,198,1,0,0,0,207,202,1,0,0,0,
        207,203,1,0,0,0,208,219,1,0,0,0,209,210,10,5,0,0,210,211,3,32,16,
        0,211,212,3,36,18,6,212,218,1,0,0,0,213,214,10,4,0,0,214,215,3,30,
        15,0,215,216,3,36,18,5,216,218,1,0,0,0,217,209,1,0,0,0,217,213,1,
        0,0,0,218,221,1,0,0,0,219,217,1,0,0,0,219,220,1,0,0,0,220,37,1,0,
        0,0,221,219,1,0,0,0,222,223,7,3,0,0,223,39,1,0,0,0,224,225,7,4,0,
        0,225,41,1,0,0,0,226,227,6,21,-1,0,227,228,3,46,23,0,228,229,5,38,
        0,0,229,230,3,42,21,5,230,238,1,0,0,0,231,238,3,44,22,0,232,238,
        3,14,7,0,233,234,5,31,0,0,234,235,3,42,21,0,235,236,5,32,0,0,236,
        238,1,0,0,0,237,226,1,0,0,0,237,231,1,0,0,0,237,232,1,0,0,0,237,
        233,1,0,0,0,238,256,1,0,0,0,239,240,10,9,0,0,240,241,5,40,0,0,241,
        255,3,42,21,10,242,243,10,8,0,0,243,244,5,41,0,0,244,255,3,42,21,
        9,245,246,10,7,0,0,246,247,5,37,0,0,247,255,3,42,21,8,248,249,10,
        6,0,0,249,250,5,38,0,0,250,255,3,42,21,7,251,252,10,4,0,0,252,253,
        5,38,0,0,253,255,3,46,23,0,254,239,1,0,0,0,254,242,1,0,0,0,254,245,
        1,0,0,0,254,248,1,0,0,0,254,251,1,0,0,0,255,258,1,0,0,0,256,254,
        1,0,0,0,256,257,1,0,0,0,257,43,1,0,0,0,258,256,1,0,0,0,259,262,3,
        38,19,0,260,262,3,40,20,0,261,259,1,0,0,0,261,260,1,0,0,0,262,45,
        1,0,0,0,263,264,6,23,-1,0,264,265,7,5,0,0,265,273,3,46,23,4,266,
        273,3,48,24,0,267,273,3,52,26,0,268,269,5,31,0,0,269,270,3,46,23,
        0,270,271,5,32,0,0,271,273,1,0,0,0,272,263,1,0,0,0,272,266,1,0,0,
        0,272,267,1,0,0,0,272,268,1,0,0,0,273,291,1,0,0,0,274,275,10,9,0,
        0,275,276,5,40,0,0,276,290,3,46,23,10,277,278,10,8,0,0,278,279,5,
        41,0,0,279,290,3,46,23,9,280,281,10,7,0,0,281,282,5,38,0,0,282,290,
        3,46,23,8,283,284,10,6,0,0,284,285,5,39,0,0,285,290,3,46,23,7,286,
        287,10,5,0,0,287,288,5,42,0,0,288,290,3,46,23,6,289,274,1,0,0,0,
        289,277,1,0,0,0,289,280,1,0,0,0,289,283,1,0,0,0,289,286,1,0,0,0,
        290,293,1,0,0,0,291,289,1,0,0,0,291,292,1,0,0,0,292,47,1,0,0,0,293,
        291,1,0,0,0,294,295,7,6,0,0,295,49,1,0,0,0,296,297,7,7,0,0,297,51,
        1,0,0,0,298,299,5,67,0,0,299,300,5,31,0,0,300,301,3,46,23,0,301,
        302,5,30,0,0,302,303,3,46,23,0,303,304,5,32,0,0,304,311,1,0,0,0,
        305,306,3,50,25,0,306,307,5,31,0,0,307,308,3,46,23,0,308,309,5,32,
        0,0,309,311,1,0,0,0,310,298,1,0,0,0,310,305,1,0,0,0,311,53,1,0,0,
        0,27,59,63,73,87,101,107,138,145,149,155,162,166,169,177,184,188,
        207,217,219,237,254,256,261,272,289,291,310
    ]

class AnalogParser ( Parser ):

    grammarFileName = "AnalogParser.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                     "<INVALID>", "'evolve'", "'measure'", "'initialize'", 
                     "'if'", "'else'", "'while'", "'with'", "'for'", "'and'", 
                     "'&&'", "'or'", "'||'", "'not'", "'!'", "'qreg'", "'qmode'", 
                     "'%C'", "'%A'", "'%J'", "'%I'", "'%X'", "'%Y'", "'%Z'", 
                     "':'", "';'", "','", "'('", "')'", "'['", "']'", "'{'", 
                     "'}'", "'@'", "'*'", "'/'", "'+'", "'-'", "'^'", "'='", 
                     "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                     "'abs'", "'sin'", "'cos'", "'tan'", "'exp'", "'log'", 
                     "'sinh'", "'cosh'", "'tanh'", "'atan'", "'acos'", "'asin'", 
                     "'atanh'", "'asinh'", "'acosh'", "'heaviside'", "'conj'", 
                     "'real'", "'imag'", "'atan2'" ]

    symbolicNames = [ "<INVALID>", "WHITESPACE", "EOL", "NEWLINE", "COMMENT", 
                      "EVOLVE", "MEASURE", "INITIALIZE", "IF", "ELSE", "WHILE", 
                      "WITH", "FOR", "AND", "AND2", "OR", "OR2", "NOT", 
                      "NOT2", "QUANTUMREGISTER", "MODEREGISTER", "CREATION", 
                      "ANNIHILATION", "IDENTITY_OP", "PAULI_I", "PAULI_X", 
                      "PAULI_Y", "PAULI_Z", "COLON", "SEMICOLON", "COMMA", 
                      "LBRACKET", "RBRACKET", "SQUARELBRACKET", "SQUARERBRACKET", 
                      "LBRACE", "RBRACE", "AT", "MULT", "DIV", "PLUS", "MINUS", 
                      "POWER", "EQ", "INT", "FLOAT", "MATH_VAR", "IMAG", 
                      "ABS", "SIN", "COS", "TAN", "EXP", "LOG", "SINH", 
                      "COSH", "TANH", "ATAN", "ACOS", "ASIN", "ATANH", "ASINH", 
                      "ACOSH", "HEAVISIDE", "CONJ", "REAL", "IMAG_FN", "ATAN2", 
                      "ID" ]

    RULE_program = 0
    RULE_statement = 1
    RULE_declaration = 2
    RULE_atomic_type = 3
    RULE_quantum_register = 4
    RULE_mode_register = 5
    RULE_my_list = 6
    RULE_access = 7
    RULE_extract = 8
    RULE_evolve_stmt = 9
    RULE_measure_stmt = 10
    RULE_init_stmt = 11
    RULE_targets = 12
    RULE_if_else_stmt = 13
    RULE_while_stmt = 14
    RULE_bool_and_op = 15
    RULE_bool_or_op = 16
    RULE_bool_not_op = 17
    RULE_bool_expr = 18
    RULE_pauli_op = 19
    RULE_ladder_op = 20
    RULE_operator_expr = 21
    RULE_operator_terminal = 22
    RULE_math_expr = 23
    RULE_math_terminal = 24
    RULE_math_func_name = 25
    RULE_math_func = 26

    ruleNames =  [ "program", "statement", "declaration", "atomic_type", 
                   "quantum_register", "mode_register", "my_list", "access", 
                   "extract", "evolve_stmt", "measure_stmt", "init_stmt", 
                   "targets", "if_else_stmt", "while_stmt", "bool_and_op", 
                   "bool_or_op", "bool_not_op", "bool_expr", "pauli_op", 
                   "ladder_op", "operator_expr", "operator_terminal", "math_expr", 
                   "math_terminal", "math_func_name", "math_func" ]

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
    AND=13
    AND2=14
    OR=15
    OR2=16
    NOT=17
    NOT2=18
    QUANTUMREGISTER=19
    MODEREGISTER=20
    CREATION=21
    ANNIHILATION=22
    IDENTITY_OP=23
    PAULI_I=24
    PAULI_X=25
    PAULI_Y=26
    PAULI_Z=27
    COLON=28
    SEMICOLON=29
    COMMA=30
    LBRACKET=31
    RBRACKET=32
    SQUARELBRACKET=33
    SQUARERBRACKET=34
    LBRACE=35
    RBRACE=36
    AT=37
    MULT=38
    DIV=39
    PLUS=40
    MINUS=41
    POWER=42
    EQ=43
    INT=44
    FLOAT=45
    MATH_VAR=46
    IMAG=47
    ABS=48
    SIN=49
    COS=50
    TAN=51
    EXP=52
    LOG=53
    SINH=54
    COSH=55
    TANH=56
    ATAN=57
    ACOS=58
    ASIN=59
    ATANH=60
    ASINH=61
    ACOSH=62
    HEAVISIDE=63
    CONJ=64
    REAL=65
    IMAG_FN=66
    ATAN2=67
    ID=68

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

        def EOF(self):
            return self.getToken(AnalogParser.EOF, 0)

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
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 59
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,0,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    self.state = 54
                    self.statement()
                    self.state = 55
                    self.match(AnalogParser.EOL) 
                self.state = 61
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,0,self._ctx)

            self.state = 63
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if ((((_la - 5)) & ~0x3f) == 0 and ((1 << (_la - 5)) & -9223372036854775761) != 0):
                self.state = 62
                self.statement()


            self.state = 65
            self.match(AnalogParser.EOF)
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


        def evolve_stmt(self):
            return self.getTypedRuleContext(AnalogParser.Evolve_stmtContext,0)


        def measure_stmt(self):
            return self.getTypedRuleContext(AnalogParser.Measure_stmtContext,0)


        def init_stmt(self):
            return self.getTypedRuleContext(AnalogParser.Init_stmtContext,0)


        def if_else_stmt(self):
            return self.getTypedRuleContext(AnalogParser.If_else_stmtContext,0)


        def while_stmt(self):
            return self.getTypedRuleContext(AnalogParser.While_stmtContext,0)


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
        self.enterRule(localctx, 2, self.RULE_statement)
        try:
            self.state = 73
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [68]:
                self.enterOuterAlt(localctx, 1)
                self.state = 67
                self.declaration()
                pass
            elif token in [5]:
                self.enterOuterAlt(localctx, 2)
                self.state = 68
                self.evolve_stmt()
                pass
            elif token in [6]:
                self.enterOuterAlt(localctx, 3)
                self.state = 69
                self.measure_stmt()
                pass
            elif token in [7]:
                self.enterOuterAlt(localctx, 4)
                self.state = 70
                self.init_stmt()
                pass
            elif token in [8]:
                self.enterOuterAlt(localctx, 5)
                self.state = 71
                self.if_else_stmt()
                pass
            elif token in [10]:
                self.enterOuterAlt(localctx, 6)
                self.state = 72
                self.while_stmt()
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


    class DeclarationContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(AnalogParser.ID, 0)

        def EQ(self):
            return self.getToken(AnalogParser.EQ, 0)

        def atomic_type(self):
            return self.getTypedRuleContext(AnalogParser.Atomic_typeContext,0)


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
        self.enterRule(localctx, 4, self.RULE_declaration)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 75
            self.match(AnalogParser.ID)
            self.state = 76
            self.match(AnalogParser.EQ)
            self.state = 77
            self.atomic_type()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Atomic_typeContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def mode_register(self):
            return self.getTypedRuleContext(AnalogParser.Mode_registerContext,0)


        def quantum_register(self):
            return self.getTypedRuleContext(AnalogParser.Quantum_registerContext,0)


        def extract(self):
            return self.getTypedRuleContext(AnalogParser.ExtractContext,0)


        def my_list(self):
            return self.getTypedRuleContext(AnalogParser.My_listContext,0)


        def access(self):
            return self.getTypedRuleContext(AnalogParser.AccessContext,0)


        def bool_expr(self):
            return self.getTypedRuleContext(AnalogParser.Bool_exprContext,0)


        def operator_expr(self):
            return self.getTypedRuleContext(AnalogParser.Operator_exprContext,0)


        def math_expr(self):
            return self.getTypedRuleContext(AnalogParser.Math_exprContext,0)


        def getRuleIndex(self):
            return AnalogParser.RULE_atomic_type

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterAtomic_type" ):
                listener.enterAtomic_type(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitAtomic_type" ):
                listener.exitAtomic_type(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAtomic_type" ):
                return visitor.visitAtomic_type(self)
            else:
                return visitor.visitChildren(self)




    def atomic_type(self):

        localctx = AnalogParser.Atomic_typeContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_atomic_type)
        try:
            self.state = 87
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,3,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 79
                self.mode_register()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 80
                self.quantum_register()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 81
                self.extract()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 82
                self.my_list()
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 83
                self.access()
                pass

            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 84
                self.bool_expr(0)
                pass

            elif la_ == 7:
                self.enterOuterAlt(localctx, 7)
                self.state = 85
                self.operator_expr(0)
                pass

            elif la_ == 8:
                self.enterOuterAlt(localctx, 8)
                self.state = 86
                self.math_expr(0)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Quantum_registerContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def QUANTUMREGISTER(self):
            return self.getToken(AnalogParser.QUANTUMREGISTER, 0)

        def LBRACKET(self):
            return self.getToken(AnalogParser.LBRACKET, 0)

        def INT(self):
            return self.getToken(AnalogParser.INT, 0)

        def RBRACKET(self):
            return self.getToken(AnalogParser.RBRACKET, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_quantum_register

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterQuantum_register" ):
                listener.enterQuantum_register(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitQuantum_register" ):
                listener.exitQuantum_register(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitQuantum_register" ):
                return visitor.visitQuantum_register(self)
            else:
                return visitor.visitChildren(self)




    def quantum_register(self):

        localctx = AnalogParser.Quantum_registerContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_quantum_register)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 89
            self.match(AnalogParser.QUANTUMREGISTER)
            self.state = 90
            self.match(AnalogParser.LBRACKET)
            self.state = 91
            self.match(AnalogParser.INT)
            self.state = 92
            self.match(AnalogParser.RBRACKET)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Mode_registerContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def MODEREGISTER(self):
            return self.getToken(AnalogParser.MODEREGISTER, 0)

        def LBRACKET(self):
            return self.getToken(AnalogParser.LBRACKET, 0)

        def INT(self):
            return self.getToken(AnalogParser.INT, 0)

        def RBRACKET(self):
            return self.getToken(AnalogParser.RBRACKET, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_mode_register

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMode_register" ):
                listener.enterMode_register(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMode_register" ):
                listener.exitMode_register(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMode_register" ):
                return visitor.visitMode_register(self)
            else:
                return visitor.visitChildren(self)




    def mode_register(self):

        localctx = AnalogParser.Mode_registerContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_mode_register)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 94
            self.match(AnalogParser.MODEREGISTER)
            self.state = 95
            self.match(AnalogParser.LBRACKET)
            self.state = 96
            self.match(AnalogParser.INT)
            self.state = 97
            self.match(AnalogParser.RBRACKET)
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
            return self.getToken(AnalogParser.SQUARELBRACKET, 0)

        def SQUARERBRACKET(self):
            return self.getToken(AnalogParser.SQUARERBRACKET, 0)

        def atomic_type(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AnalogParser.Atomic_typeContext)
            else:
                return self.getTypedRuleContext(AnalogParser.Atomic_typeContext,i)


        def COMMA(self, i:int=None):
            if i is None:
                return self.getTokens(AnalogParser.COMMA)
            else:
                return self.getToken(AnalogParser.COMMA, i)

        def getRuleIndex(self):
            return AnalogParser.RULE_my_list

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

        localctx = AnalogParser.My_listContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_my_list)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 99
            self.match(AnalogParser.SQUARELBRACKET)
            self.state = 101
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if ((((_la - 17)) & ~0x3f) == 0 and ((1 << (_la - 17)) & 4503599518402559) != 0):
                self.state = 100
                self.atomic_type()


            self.state = 107
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==30:
                self.state = 103
                self.match(AnalogParser.COMMA)
                self.state = 104
                self.atomic_type()
                self.state = 109
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 110
            self.match(AnalogParser.SQUARERBRACKET)
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
        self.enterRule(localctx, 14, self.RULE_access)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 112
            self.match(AnalogParser.ID)
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
            return self.getTypedRuleContext(AnalogParser.AccessContext,0)


        def SQUARELBRACKET(self):
            return self.getToken(AnalogParser.SQUARELBRACKET, 0)

        def INT(self):
            return self.getToken(AnalogParser.INT, 0)

        def SQUARERBRACKET(self):
            return self.getToken(AnalogParser.SQUARERBRACKET, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_extract

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

        localctx = AnalogParser.ExtractContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_extract)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 114
            self.access()
            self.state = 115
            self.match(AnalogParser.SQUARELBRACKET)
            self.state = 116
            self.match(AnalogParser.INT)
            self.state = 117
            self.match(AnalogParser.SQUARERBRACKET)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Evolve_stmtContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EVOLVE(self):
            return self.getToken(AnalogParser.EVOLVE, 0)

        def targets(self):
            return self.getTypedRuleContext(AnalogParser.TargetsContext,0)


        def WITH(self):
            return self.getToken(AnalogParser.WITH, 0)

        def atomic_type(self):
            return self.getTypedRuleContext(AnalogParser.Atomic_typeContext,0)


        def getRuleIndex(self):
            return AnalogParser.RULE_evolve_stmt

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterEvolve_stmt" ):
                listener.enterEvolve_stmt(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitEvolve_stmt" ):
                listener.exitEvolve_stmt(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitEvolve_stmt" ):
                return visitor.visitEvolve_stmt(self)
            else:
                return visitor.visitChildren(self)




    def evolve_stmt(self):

        localctx = AnalogParser.Evolve_stmtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_evolve_stmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 119
            self.match(AnalogParser.EVOLVE)
            self.state = 120
            self.targets()
            self.state = 121
            self.match(AnalogParser.WITH)
            self.state = 122
            self.atomic_type()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Measure_stmtContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def MEASURE(self):
            return self.getToken(AnalogParser.MEASURE, 0)

        def targets(self):
            return self.getTypedRuleContext(AnalogParser.TargetsContext,0)


        def getRuleIndex(self):
            return AnalogParser.RULE_measure_stmt

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMeasure_stmt" ):
                listener.enterMeasure_stmt(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMeasure_stmt" ):
                listener.exitMeasure_stmt(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMeasure_stmt" ):
                return visitor.visitMeasure_stmt(self)
            else:
                return visitor.visitChildren(self)




    def measure_stmt(self):

        localctx = AnalogParser.Measure_stmtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_measure_stmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 124
            self.match(AnalogParser.MEASURE)
            self.state = 125
            self.targets()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Init_stmtContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def INITIALIZE(self):
            return self.getToken(AnalogParser.INITIALIZE, 0)

        def targets(self):
            return self.getTypedRuleContext(AnalogParser.TargetsContext,0)


        def getRuleIndex(self):
            return AnalogParser.RULE_init_stmt

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterInit_stmt" ):
                listener.enterInit_stmt(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitInit_stmt" ):
                listener.exitInit_stmt(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitInit_stmt" ):
                return visitor.visitInit_stmt(self)
            else:
                return visitor.visitChildren(self)




    def init_stmt(self):

        localctx = AnalogParser.Init_stmtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_init_stmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 127
            self.match(AnalogParser.INITIALIZE)
            self.state = 128
            self.targets()
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

        def atomic_type(self):
            return self.getTypedRuleContext(AnalogParser.Atomic_typeContext,0)


        def getRuleIndex(self):
            return AnalogParser.RULE_targets

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

        localctx = AnalogParser.TargetsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_targets)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 130
            self.atomic_type()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class If_else_stmtContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def IF(self):
            return self.getToken(AnalogParser.IF, 0)

        def LBRACKET(self):
            return self.getToken(AnalogParser.LBRACKET, 0)

        def bool_expr(self):
            return self.getTypedRuleContext(AnalogParser.Bool_exprContext,0)


        def RBRACKET(self):
            return self.getToken(AnalogParser.RBRACKET, 0)

        def LBRACE(self, i:int=None):
            if i is None:
                return self.getTokens(AnalogParser.LBRACE)
            else:
                return self.getToken(AnalogParser.LBRACE, i)

        def RBRACE(self, i:int=None):
            if i is None:
                return self.getTokens(AnalogParser.RBRACE)
            else:
                return self.getToken(AnalogParser.RBRACE, i)

        def EOL(self, i:int=None):
            if i is None:
                return self.getTokens(AnalogParser.EOL)
            else:
                return self.getToken(AnalogParser.EOL, i)

        def statement(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AnalogParser.StatementContext)
            else:
                return self.getTypedRuleContext(AnalogParser.StatementContext,i)


        def ELSE(self):
            return self.getToken(AnalogParser.ELSE, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_if_else_stmt

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterIf_else_stmt" ):
                listener.enterIf_else_stmt(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitIf_else_stmt" ):
                listener.exitIf_else_stmt(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitIf_else_stmt" ):
                return visitor.visitIf_else_stmt(self)
            else:
                return visitor.visitChildren(self)




    def if_else_stmt(self):

        localctx = AnalogParser.If_else_stmtContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_if_else_stmt)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 132
            self.match(AnalogParser.IF)
            self.state = 133
            self.match(AnalogParser.LBRACKET)
            self.state = 134
            self.bool_expr(0)
            self.state = 135
            self.match(AnalogParser.RBRACKET)
            self.state = 136
            self.match(AnalogParser.LBRACE)
            self.state = 138
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==2:
                self.state = 137
                self.match(AnalogParser.EOL)


            self.state = 145
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,7,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    self.state = 140
                    self.statement()
                    self.state = 141
                    self.match(AnalogParser.EOL) 
                self.state = 147
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,7,self._ctx)

            self.state = 149
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if ((((_la - 5)) & ~0x3f) == 0 and ((1 << (_la - 5)) & -9223372036854775761) != 0):
                self.state = 148
                self.statement()


            self.state = 151
            self.match(AnalogParser.RBRACE)
            self.state = 169
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==9:
                self.state = 152
                self.match(AnalogParser.ELSE)
                self.state = 153
                self.match(AnalogParser.LBRACE)
                self.state = 155
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==2:
                    self.state = 154
                    self.match(AnalogParser.EOL)


                self.state = 162
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,10,self._ctx)
                while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                    if _alt==1:
                        self.state = 157
                        self.statement()
                        self.state = 158
                        self.match(AnalogParser.EOL) 
                    self.state = 164
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input,10,self._ctx)

                self.state = 166
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if ((((_la - 5)) & ~0x3f) == 0 and ((1 << (_la - 5)) & -9223372036854775761) != 0):
                    self.state = 165
                    self.statement()


                self.state = 168
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

        def bool_expr(self):
            return self.getTypedRuleContext(AnalogParser.Bool_exprContext,0)


        def RBRACKET(self):
            return self.getToken(AnalogParser.RBRACKET, 0)

        def LBRACE(self):
            return self.getToken(AnalogParser.LBRACE, 0)

        def RBRACE(self):
            return self.getToken(AnalogParser.RBRACE, 0)

        def EOL(self, i:int=None):
            if i is None:
                return self.getTokens(AnalogParser.EOL)
            else:
                return self.getToken(AnalogParser.EOL, i)

        def statement(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AnalogParser.StatementContext)
            else:
                return self.getTypedRuleContext(AnalogParser.StatementContext,i)


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
        self.enterRule(localctx, 28, self.RULE_while_stmt)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 171
            self.match(AnalogParser.WHILE)
            self.state = 172
            self.match(AnalogParser.LBRACKET)
            self.state = 173
            self.bool_expr(0)
            self.state = 174
            self.match(AnalogParser.RBRACKET)
            self.state = 175
            self.match(AnalogParser.LBRACE)
            self.state = 177
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==2:
                self.state = 176
                self.match(AnalogParser.EOL)


            self.state = 184
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,14,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    self.state = 179
                    self.statement()
                    self.state = 180
                    self.match(AnalogParser.EOL) 
                self.state = 186
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,14,self._ctx)

            self.state = 188
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if ((((_la - 5)) & ~0x3f) == 0 and ((1 << (_la - 5)) & -9223372036854775761) != 0):
                self.state = 187
                self.statement()


            self.state = 190
            self.match(AnalogParser.RBRACE)
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
            return self.getToken(AnalogParser.AND, 0)

        def AND2(self):
            return self.getToken(AnalogParser.AND2, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_bool_and_op

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

        localctx = AnalogParser.Bool_and_opContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_bool_and_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 192
            _la = self._input.LA(1)
            if not(_la==13 or _la==14):
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
            return self.getToken(AnalogParser.OR, 0)

        def OR2(self):
            return self.getToken(AnalogParser.OR2, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_bool_or_op

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

        localctx = AnalogParser.Bool_or_opContext(self, self._ctx, self.state)
        self.enterRule(localctx, 32, self.RULE_bool_or_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 194
            _la = self._input.LA(1)
            if not(_la==15 or _la==16):
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
            return self.getToken(AnalogParser.NOT, 0)

        def NOT2(self):
            return self.getToken(AnalogParser.NOT2, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_bool_not_op

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

        localctx = AnalogParser.Bool_not_opContext(self, self._ctx, self.state)
        self.enterRule(localctx, 34, self.RULE_bool_not_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 196
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


    class Bool_exprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def bool_not_op(self):
            return self.getTypedRuleContext(AnalogParser.Bool_not_opContext,0)


        def bool_expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AnalogParser.Bool_exprContext)
            else:
                return self.getTypedRuleContext(AnalogParser.Bool_exprContext,i)


        def access(self):
            return self.getTypedRuleContext(AnalogParser.AccessContext,0)


        def LBRACKET(self):
            return self.getToken(AnalogParser.LBRACKET, 0)

        def RBRACKET(self):
            return self.getToken(AnalogParser.RBRACKET, 0)

        def bool_or_op(self):
            return self.getTypedRuleContext(AnalogParser.Bool_or_opContext,0)


        def bool_and_op(self):
            return self.getTypedRuleContext(AnalogParser.Bool_and_opContext,0)


        def getRuleIndex(self):
            return AnalogParser.RULE_bool_expr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBool_expr" ):
                listener.enterBool_expr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBool_expr" ):
                listener.exitBool_expr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBool_expr" ):
                return visitor.visitBool_expr(self)
            else:
                return visitor.visitChildren(self)



    def bool_expr(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = AnalogParser.Bool_exprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 36
        self.enterRecursionRule(localctx, 36, self.RULE_bool_expr, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 207
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [17, 18]:
                self.state = 199
                self.bool_not_op()
                self.state = 200
                self.bool_expr(3)
                pass
            elif token in [68]:
                self.state = 202
                self.access()
                pass
            elif token in [31]:
                self.state = 203
                self.match(AnalogParser.LBRACKET)
                self.state = 204
                self.bool_expr(0)
                self.state = 205
                self.match(AnalogParser.RBRACKET)
                pass
            else:
                raise NoViableAltException(self)

            self._ctx.stop = self._input.LT(-1)
            self.state = 219
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,18,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 217
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,17,self._ctx)
                    if la_ == 1:
                        localctx = AnalogParser.Bool_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_bool_expr)
                        self.state = 209
                        if not self.precpred(self._ctx, 5):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 5)")
                        self.state = 210
                        self.bool_or_op()
                        self.state = 211
                        self.bool_expr(6)
                        pass

                    elif la_ == 2:
                        localctx = AnalogParser.Bool_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_bool_expr)
                        self.state = 213
                        if not self.precpred(self._ctx, 4):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 4)")
                        self.state = 214
                        self.bool_and_op()
                        self.state = 215
                        self.bool_expr(5)
                        pass

             
                self.state = 221
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,18,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
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
        self.enterRule(localctx, 38, self.RULE_pauli_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 222
            _la = self._input.LA(1)
            if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 251658240) != 0)):
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
        self.enterRule(localctx, 40, self.RULE_ladder_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 224
            _la = self._input.LA(1)
            if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 14680064) != 0)):
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


    class Operator_exprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def math_expr(self):
            return self.getTypedRuleContext(AnalogParser.Math_exprContext,0)


        def MULT(self):
            return self.getToken(AnalogParser.MULT, 0)

        def operator_expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AnalogParser.Operator_exprContext)
            else:
                return self.getTypedRuleContext(AnalogParser.Operator_exprContext,i)


        def operator_terminal(self):
            return self.getTypedRuleContext(AnalogParser.Operator_terminalContext,0)


        def access(self):
            return self.getTypedRuleContext(AnalogParser.AccessContext,0)


        def LBRACKET(self):
            return self.getToken(AnalogParser.LBRACKET, 0)

        def RBRACKET(self):
            return self.getToken(AnalogParser.RBRACKET, 0)

        def PLUS(self):
            return self.getToken(AnalogParser.PLUS, 0)

        def MINUS(self):
            return self.getToken(AnalogParser.MINUS, 0)

        def AT(self):
            return self.getToken(AnalogParser.AT, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_operator_expr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterOperator_expr" ):
                listener.enterOperator_expr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitOperator_expr" ):
                listener.exitOperator_expr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitOperator_expr" ):
                return visitor.visitOperator_expr(self)
            else:
                return visitor.visitChildren(self)



    def operator_expr(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = AnalogParser.Operator_exprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 42
        self.enterRecursionRule(localctx, 42, self.RULE_operator_expr, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 237
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,19,self._ctx)
            if la_ == 1:
                self.state = 227
                self.math_expr(0)
                self.state = 228
                self.match(AnalogParser.MULT)
                self.state = 229
                self.operator_expr(5)
                pass

            elif la_ == 2:
                self.state = 231
                self.operator_terminal()
                pass

            elif la_ == 3:
                self.state = 232
                self.access()
                pass

            elif la_ == 4:
                self.state = 233
                self.match(AnalogParser.LBRACKET)
                self.state = 234
                self.operator_expr(0)
                self.state = 235
                self.match(AnalogParser.RBRACKET)
                pass


            self._ctx.stop = self._input.LT(-1)
            self.state = 256
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,21,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 254
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,20,self._ctx)
                    if la_ == 1:
                        localctx = AnalogParser.Operator_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_operator_expr)
                        self.state = 239
                        if not self.precpred(self._ctx, 9):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 9)")
                        self.state = 240
                        self.match(AnalogParser.PLUS)
                        self.state = 241
                        self.operator_expr(10)
                        pass

                    elif la_ == 2:
                        localctx = AnalogParser.Operator_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_operator_expr)
                        self.state = 242
                        if not self.precpred(self._ctx, 8):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 8)")
                        self.state = 243
                        self.match(AnalogParser.MINUS)
                        self.state = 244
                        self.operator_expr(9)
                        pass

                    elif la_ == 3:
                        localctx = AnalogParser.Operator_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_operator_expr)
                        self.state = 245
                        if not self.precpred(self._ctx, 7):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 7)")
                        self.state = 246
                        self.match(AnalogParser.AT)
                        self.state = 247
                        self.operator_expr(8)
                        pass

                    elif la_ == 4:
                        localctx = AnalogParser.Operator_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_operator_expr)
                        self.state = 248
                        if not self.precpred(self._ctx, 6):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 6)")
                        self.state = 249
                        self.match(AnalogParser.MULT)
                        self.state = 250
                        self.operator_expr(7)
                        pass

                    elif la_ == 5:
                        localctx = AnalogParser.Operator_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_operator_expr)
                        self.state = 251
                        if not self.precpred(self._ctx, 4):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 4)")
                        self.state = 252
                        self.match(AnalogParser.MULT)
                        self.state = 253
                        self.math_expr(0)
                        pass

             
                self.state = 258
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,21,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
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
        self.enterRule(localctx, 44, self.RULE_operator_terminal)
        try:
            self.state = 261
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [24, 25, 26, 27]:
                self.enterOuterAlt(localctx, 1)
                self.state = 259
                self.pauli_op()
                pass
            elif token in [21, 22, 23]:
                self.enterOuterAlt(localctx, 2)
                self.state = 260
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


    class Math_exprContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def math_expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AnalogParser.Math_exprContext)
            else:
                return self.getTypedRuleContext(AnalogParser.Math_exprContext,i)


        def PLUS(self):
            return self.getToken(AnalogParser.PLUS, 0)

        def MINUS(self):
            return self.getToken(AnalogParser.MINUS, 0)

        def math_terminal(self):
            return self.getTypedRuleContext(AnalogParser.Math_terminalContext,0)


        def math_func(self):
            return self.getTypedRuleContext(AnalogParser.Math_funcContext,0)


        def LBRACKET(self):
            return self.getToken(AnalogParser.LBRACKET, 0)

        def RBRACKET(self):
            return self.getToken(AnalogParser.RBRACKET, 0)

        def MULT(self):
            return self.getToken(AnalogParser.MULT, 0)

        def DIV(self):
            return self.getToken(AnalogParser.DIV, 0)

        def POWER(self):
            return self.getToken(AnalogParser.POWER, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_math_expr

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterMath_expr" ):
                listener.enterMath_expr(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitMath_expr" ):
                listener.exitMath_expr(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitMath_expr" ):
                return visitor.visitMath_expr(self)
            else:
                return visitor.visitChildren(self)



    def math_expr(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = AnalogParser.Math_exprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 46
        self.enterRecursionRule(localctx, 46, self.RULE_math_expr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 272
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [40, 41]:
                self.state = 264
                _la = self._input.LA(1)
                if not(_la==40 or _la==41):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 265
                self.math_expr(4)
                pass
            elif token in [44, 45, 46, 47, 68]:
                self.state = 266
                self.math_terminal()
                pass
            elif token in [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]:
                self.state = 267
                self.math_func()
                pass
            elif token in [31]:
                self.state = 268
                self.match(AnalogParser.LBRACKET)
                self.state = 269
                self.math_expr(0)
                self.state = 270
                self.match(AnalogParser.RBRACKET)
                pass
            else:
                raise NoViableAltException(self)

            self._ctx.stop = self._input.LT(-1)
            self.state = 291
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,25,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 289
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,24,self._ctx)
                    if la_ == 1:
                        localctx = AnalogParser.Math_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_math_expr)
                        self.state = 274
                        if not self.precpred(self._ctx, 9):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 9)")
                        self.state = 275
                        self.match(AnalogParser.PLUS)
                        self.state = 276
                        self.math_expr(10)
                        pass

                    elif la_ == 2:
                        localctx = AnalogParser.Math_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_math_expr)
                        self.state = 277
                        if not self.precpred(self._ctx, 8):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 8)")
                        self.state = 278
                        self.match(AnalogParser.MINUS)
                        self.state = 279
                        self.math_expr(9)
                        pass

                    elif la_ == 3:
                        localctx = AnalogParser.Math_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_math_expr)
                        self.state = 280
                        if not self.precpred(self._ctx, 7):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 7)")
                        self.state = 281
                        self.match(AnalogParser.MULT)
                        self.state = 282
                        self.math_expr(8)
                        pass

                    elif la_ == 4:
                        localctx = AnalogParser.Math_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_math_expr)
                        self.state = 283
                        if not self.precpred(self._ctx, 6):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 6)")
                        self.state = 284
                        self.match(AnalogParser.DIV)
                        self.state = 285
                        self.math_expr(7)
                        pass

                    elif la_ == 5:
                        localctx = AnalogParser.Math_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_math_expr)
                        self.state = 286
                        if not self.precpred(self._ctx, 5):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 5)")
                        self.state = 287
                        self.match(AnalogParser.POWER)
                        self.state = 288
                        self.math_expr(6)
                        pass

             
                self.state = 293
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,25,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
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

        def IMAG(self):
            return self.getToken(AnalogParser.IMAG, 0)

        def ID(self):
            return self.getToken(AnalogParser.ID, 0)

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
        self.enterRule(localctx, 48, self.RULE_math_terminal)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 294
            _la = self._input.LA(1)
            if not(((((_la - 44)) & ~0x3f) == 0 and ((1 << (_la - 44)) & 16777231) != 0)):
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


    class Math_func_nameContext(ParserRuleContext):
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

        def HEAVISIDE(self):
            return self.getToken(AnalogParser.HEAVISIDE, 0)

        def CONJ(self):
            return self.getToken(AnalogParser.CONJ, 0)

        def REAL(self):
            return self.getToken(AnalogParser.REAL, 0)

        def IMAG_FN(self):
            return self.getToken(AnalogParser.IMAG_FN, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_math_func_name

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

        localctx = AnalogParser.Math_func_nameContext(self, self._ctx, self.state)
        self.enterRule(localctx, 50, self.RULE_math_func_name)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 296
            _la = self._input.LA(1)
            if not(((((_la - 48)) & ~0x3f) == 0 and ((1 << (_la - 48)) & 524287) != 0)):
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

        def ATAN2(self):
            return self.getToken(AnalogParser.ATAN2, 0)

        def LBRACKET(self):
            return self.getToken(AnalogParser.LBRACKET, 0)

        def math_expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AnalogParser.Math_exprContext)
            else:
                return self.getTypedRuleContext(AnalogParser.Math_exprContext,i)


        def COMMA(self):
            return self.getToken(AnalogParser.COMMA, 0)

        def RBRACKET(self):
            return self.getToken(AnalogParser.RBRACKET, 0)

        def math_func_name(self):
            return self.getTypedRuleContext(AnalogParser.Math_func_nameContext,0)


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
        self.enterRule(localctx, 52, self.RULE_math_func)
        try:
            self.state = 310
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [67]:
                self.enterOuterAlt(localctx, 1)
                self.state = 298
                self.match(AnalogParser.ATAN2)
                self.state = 299
                self.match(AnalogParser.LBRACKET)
                self.state = 300
                self.math_expr(0)
                self.state = 301
                self.match(AnalogParser.COMMA)
                self.state = 302
                self.math_expr(0)
                self.state = 303
                self.match(AnalogParser.RBRACKET)
                pass
            elif token in [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]:
                self.enterOuterAlt(localctx, 2)
                self.state = 305
                self.math_func_name()
                self.state = 306
                self.match(AnalogParser.LBRACKET)
                self.state = 307
                self.math_expr(0)
                self.state = 308
                self.match(AnalogParser.RBRACKET)
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



    def sempred(self, localctx:RuleContext, ruleIndex:int, predIndex:int):
        if self._predicates == None:
            self._predicates = dict()
        self._predicates[18] = self.bool_expr_sempred
        self._predicates[21] = self.operator_expr_sempred
        self._predicates[23] = self.math_expr_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception("No predicate with index:" + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def bool_expr_sempred(self, localctx:Bool_exprContext, predIndex:int):
            if predIndex == 0:
                return self.precpred(self._ctx, 5)
         

            if predIndex == 1:
                return self.precpred(self._ctx, 4)
         

    def operator_expr_sempred(self, localctx:Operator_exprContext, predIndex:int):
            if predIndex == 2:
                return self.precpred(self._ctx, 9)
         

            if predIndex == 3:
                return self.precpred(self._ctx, 8)
         

            if predIndex == 4:
                return self.precpred(self._ctx, 7)
         

            if predIndex == 5:
                return self.precpred(self._ctx, 6)
         

            if predIndex == 6:
                return self.precpred(self._ctx, 4)
         

    def math_expr_sempred(self, localctx:Math_exprContext, predIndex:int):
            if predIndex == 7:
                return self.precpred(self._ctx, 9)
         

            if predIndex == 8:
                return self.precpred(self._ctx, 8)
         

            if predIndex == 9:
                return self.precpred(self._ctx, 7)
         

            if predIndex == 10:
                return self.precpred(self._ctx, 6)
         

            if predIndex == 11:
                return self.precpred(self._ctx, 5)
         




