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
        4,1,81,317,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,20,
        7,20,2,21,7,21,2,22,7,22,2,23,7,23,2,24,7,24,2,25,7,25,2,26,7,26,
        2,27,7,27,2,28,7,28,2,29,7,29,2,30,7,30,2,31,7,31,2,32,7,32,2,33,
        7,33,2,34,7,34,2,35,7,35,2,36,7,36,2,37,7,37,2,38,7,38,1,0,1,0,1,
        0,1,1,1,1,1,1,1,1,1,1,1,1,3,1,88,8,1,1,2,1,2,1,2,1,2,5,2,94,8,2,
        10,2,12,2,97,9,2,1,2,3,2,100,8,2,1,3,1,3,1,3,1,3,1,3,3,3,107,8,3,
        1,4,1,4,1,4,1,4,1,4,4,4,114,8,4,11,4,12,4,115,1,4,1,4,1,4,1,4,1,
        4,1,4,1,4,1,4,1,4,1,4,1,4,3,4,129,8,4,1,4,1,4,1,4,3,4,134,8,4,1,
        4,1,4,5,4,138,8,4,10,4,12,4,141,9,4,1,5,1,5,1,6,1,6,3,6,147,8,6,
        1,6,1,6,5,6,151,8,6,10,6,12,6,154,9,6,1,6,1,6,1,7,1,7,1,7,1,7,1,
        8,1,8,1,9,1,9,1,9,1,9,1,9,1,10,1,10,1,11,1,11,1,12,1,12,1,12,1,12,
        1,12,1,12,1,12,1,12,1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,
        1,13,1,13,1,13,1,13,1,13,1,13,1,13,3,13,197,8,13,1,13,1,13,1,13,
        1,13,1,13,3,13,204,8,13,1,14,1,14,1,14,1,14,1,14,1,15,1,15,1,15,
        1,15,1,15,1,16,1,16,1,17,1,17,1,18,1,18,1,19,1,19,1,20,1,20,1,21,
        1,21,1,22,1,22,1,23,1,23,1,24,1,24,1,25,1,25,1,26,1,26,1,27,1,27,
        1,27,1,27,1,27,1,27,3,27,244,8,27,1,28,1,28,1,29,1,29,1,30,1,30,
        3,30,252,8,30,1,31,1,31,1,31,1,31,1,31,1,31,1,31,3,31,261,8,31,1,
        32,1,32,1,33,1,33,1,33,1,33,1,34,1,34,1,34,1,34,1,34,5,34,274,8,
        34,10,34,12,34,277,9,34,3,34,279,8,34,1,34,1,34,1,35,1,35,1,35,1,
        35,1,35,1,35,5,35,289,8,35,10,35,12,35,292,9,35,1,36,1,36,1,36,1,
        36,1,36,1,36,5,36,300,8,36,10,36,12,36,303,9,36,1,37,1,37,1,37,3,
        37,308,8,37,1,38,1,38,1,38,1,38,1,38,3,38,315,8,38,1,38,0,3,8,70,
        72,39,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,
        42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,0,10,1,0,15,
        16,1,0,17,18,1,0,19,20,1,0,21,22,1,0,74,77,1,0,78,80,2,0,5,7,54,
        73,3,0,36,37,47,47,49,49,3,0,34,35,46,46,48,48,1,0,36,37,320,0,78,
        1,0,0,0,2,87,1,0,0,0,4,95,1,0,0,0,6,106,1,0,0,0,8,128,1,0,0,0,10,
        142,1,0,0,0,12,144,1,0,0,0,14,157,1,0,0,0,16,161,1,0,0,0,18,163,
        1,0,0,0,20,168,1,0,0,0,22,170,1,0,0,0,24,172,1,0,0,0,26,203,1,0,
        0,0,28,205,1,0,0,0,30,210,1,0,0,0,32,215,1,0,0,0,34,217,1,0,0,0,
        36,219,1,0,0,0,38,221,1,0,0,0,40,223,1,0,0,0,42,225,1,0,0,0,44,227,
        1,0,0,0,46,229,1,0,0,0,48,231,1,0,0,0,50,233,1,0,0,0,52,235,1,0,
        0,0,54,243,1,0,0,0,56,245,1,0,0,0,58,247,1,0,0,0,60,251,1,0,0,0,
        62,260,1,0,0,0,64,262,1,0,0,0,66,264,1,0,0,0,68,268,1,0,0,0,70,282,
        1,0,0,0,72,293,1,0,0,0,74,307,1,0,0,0,76,314,1,0,0,0,78,79,3,4,2,
        0,79,80,5,0,0,1,80,1,1,0,0,0,81,88,3,14,7,0,82,88,3,24,12,0,83,88,
        3,26,13,0,84,88,3,20,10,0,85,88,3,22,11,0,86,88,3,8,4,0,87,81,1,
        0,0,0,87,82,1,0,0,0,87,83,1,0,0,0,87,84,1,0,0,0,87,85,1,0,0,0,87,
        86,1,0,0,0,88,3,1,0,0,0,89,90,3,2,1,0,90,91,5,2,0,0,91,94,1,0,0,
        0,92,94,5,2,0,0,93,89,1,0,0,0,93,92,1,0,0,0,94,97,1,0,0,0,95,93,
        1,0,0,0,95,96,1,0,0,0,96,99,1,0,0,0,97,95,1,0,0,0,98,100,3,2,1,0,
        99,98,1,0,0,0,99,100,1,0,0,0,100,5,1,0,0,0,101,107,3,30,15,0,102,
        107,3,28,14,0,103,107,3,60,30,0,104,107,3,62,31,0,105,107,3,52,26,
        0,106,101,1,0,0,0,106,102,1,0,0,0,106,103,1,0,0,0,106,104,1,0,0,
        0,106,105,1,0,0,0,107,7,1,0,0,0,108,109,6,4,-1,0,109,113,3,70,35,
        0,110,111,3,54,27,0,111,112,3,70,35,0,112,114,1,0,0,0,113,110,1,
        0,0,0,114,115,1,0,0,0,115,113,1,0,0,0,115,116,1,0,0,0,116,129,1,
        0,0,0,117,118,3,38,19,0,118,119,3,8,4,6,119,129,1,0,0,0,120,121,
        5,28,0,0,121,122,3,8,4,0,122,123,5,29,0,0,123,129,1,0,0,0,124,129,
        3,18,9,0,125,129,3,12,6,0,126,129,3,6,3,0,127,129,3,70,35,0,128,
        108,1,0,0,0,128,117,1,0,0,0,128,120,1,0,0,0,128,124,1,0,0,0,128,
        125,1,0,0,0,128,126,1,0,0,0,128,127,1,0,0,0,129,139,1,0,0,0,130,
        133,10,7,0,0,131,134,3,34,17,0,132,134,3,36,18,0,133,131,1,0,0,0,
        133,132,1,0,0,0,134,135,1,0,0,0,135,136,3,8,4,8,136,138,1,0,0,0,
        137,130,1,0,0,0,138,141,1,0,0,0,139,137,1,0,0,0,139,140,1,0,0,0,
        140,9,1,0,0,0,141,139,1,0,0,0,142,143,3,8,4,0,143,11,1,0,0,0,144,
        146,5,30,0,0,145,147,3,8,4,0,146,145,1,0,0,0,146,147,1,0,0,0,147,
        152,1,0,0,0,148,149,5,27,0,0,149,151,3,8,4,0,150,148,1,0,0,0,151,
        154,1,0,0,0,152,150,1,0,0,0,152,153,1,0,0,0,153,155,1,0,0,0,154,
        152,1,0,0,0,155,156,5,31,0,0,156,13,1,0,0,0,157,158,5,81,0,0,158,
        159,5,39,0,0,159,160,3,8,4,0,160,15,1,0,0,0,161,162,5,81,0,0,162,
        17,1,0,0,0,163,164,3,16,8,0,164,165,5,30,0,0,165,166,5,50,0,0,166,
        167,5,31,0,0,167,19,1,0,0,0,168,169,5,13,0,0,169,21,1,0,0,0,170,
        171,5,14,0,0,171,23,1,0,0,0,172,173,5,10,0,0,173,174,5,28,0,0,174,
        175,3,10,5,0,175,176,5,29,0,0,176,177,5,32,0,0,177,178,3,4,2,0,178,
        179,5,33,0,0,179,25,1,0,0,0,180,181,5,8,0,0,181,182,5,28,0,0,182,
        183,3,10,5,0,183,184,5,29,0,0,184,185,5,32,0,0,185,186,3,4,2,0,186,
        187,5,33,0,0,187,204,1,0,0,0,188,189,5,8,0,0,189,190,5,28,0,0,190,
        191,3,10,5,0,191,192,5,29,0,0,192,193,5,32,0,0,193,194,3,4,2,0,194,
        196,5,33,0,0,195,197,5,2,0,0,196,195,1,0,0,0,196,197,1,0,0,0,197,
        198,1,0,0,0,198,199,5,9,0,0,199,200,5,32,0,0,200,201,3,4,2,0,201,
        202,5,33,0,0,202,204,1,0,0,0,203,180,1,0,0,0,203,188,1,0,0,0,204,
        27,1,0,0,0,205,206,5,23,0,0,206,207,5,28,0,0,207,208,5,50,0,0,208,
        209,5,29,0,0,209,29,1,0,0,0,210,211,5,24,0,0,211,212,5,28,0,0,212,
        213,5,50,0,0,213,214,5,29,0,0,214,31,1,0,0,0,215,216,3,8,4,0,216,
        33,1,0,0,0,217,218,7,0,0,0,218,35,1,0,0,0,219,220,7,1,0,0,220,37,
        1,0,0,0,221,222,7,2,0,0,222,39,1,0,0,0,223,224,5,40,0,0,224,41,1,
        0,0,0,225,226,5,41,0,0,226,43,1,0,0,0,227,228,5,42,0,0,228,45,1,
        0,0,0,229,230,5,43,0,0,230,47,1,0,0,0,231,232,5,44,0,0,232,49,1,
        0,0,0,233,234,5,45,0,0,234,51,1,0,0,0,235,236,7,3,0,0,236,53,1,0,
        0,0,237,244,3,40,20,0,238,244,3,42,21,0,239,244,3,44,22,0,240,244,
        3,46,23,0,241,244,3,48,24,0,242,244,3,50,25,0,243,237,1,0,0,0,243,
        238,1,0,0,0,243,239,1,0,0,0,243,240,1,0,0,0,243,241,1,0,0,0,243,
        242,1,0,0,0,244,55,1,0,0,0,245,246,7,4,0,0,246,57,1,0,0,0,247,248,
        7,5,0,0,248,59,1,0,0,0,249,252,3,56,28,0,250,252,3,58,29,0,251,249,
        1,0,0,0,251,250,1,0,0,0,252,61,1,0,0,0,253,261,5,50,0,0,254,261,
        5,51,0,0,255,261,5,52,0,0,256,261,5,53,0,0,257,261,3,16,8,0,258,
        261,3,66,33,0,259,261,3,68,34,0,260,253,1,0,0,0,260,254,1,0,0,0,
        260,255,1,0,0,0,260,256,1,0,0,0,260,257,1,0,0,0,260,258,1,0,0,0,
        260,259,1,0,0,0,261,63,1,0,0,0,262,263,7,6,0,0,263,65,1,0,0,0,264,
        265,5,28,0,0,265,266,3,70,35,0,266,267,5,29,0,0,267,67,1,0,0,0,268,
        269,3,64,32,0,269,278,5,28,0,0,270,275,3,70,35,0,271,272,5,27,0,
        0,272,274,3,70,35,0,273,271,1,0,0,0,274,277,1,0,0,0,275,273,1,0,
        0,0,275,276,1,0,0,0,276,279,1,0,0,0,277,275,1,0,0,0,278,270,1,0,
        0,0,278,279,1,0,0,0,279,280,1,0,0,0,280,281,5,29,0,0,281,69,1,0,
        0,0,282,283,6,35,-1,0,283,284,3,72,36,0,284,290,1,0,0,0,285,286,
        10,1,0,0,286,287,7,7,0,0,287,289,3,72,36,0,288,285,1,0,0,0,289,292,
        1,0,0,0,290,288,1,0,0,0,290,291,1,0,0,0,291,71,1,0,0,0,292,290,1,
        0,0,0,293,294,6,36,-1,0,294,295,3,74,37,0,295,301,1,0,0,0,296,297,
        10,1,0,0,297,298,7,8,0,0,298,300,3,74,37,0,299,296,1,0,0,0,300,303,
        1,0,0,0,301,299,1,0,0,0,301,302,1,0,0,0,302,73,1,0,0,0,303,301,1,
        0,0,0,304,308,3,76,38,0,305,306,7,9,0,0,306,308,3,76,38,0,307,304,
        1,0,0,0,307,305,1,0,0,0,308,75,1,0,0,0,309,315,3,6,3,0,310,311,3,
        6,3,0,311,312,5,38,0,0,312,313,3,74,37,0,313,315,1,0,0,0,314,309,
        1,0,0,0,314,310,1,0,0,0,315,77,1,0,0,0,22,87,93,95,99,106,115,128,
        133,139,146,152,196,203,243,251,260,275,278,290,301,307,314
    ]

class AnalogParser ( Parser ):

    grammarFileName = "AnalogParser.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                     "<INVALID>", "'evolve'", "'measure'", "'initialize'", 
                     "'if'", "'else'", "'while'", "'with'", "'for'", "'break'", 
                     "'continue'", "'and'", "'&&'", "'or'", "'||'", "'not'", 
                     "'!'", "'true'", "'false'", "'qreg'", "'qmode'", "':'", 
                     "';'", "','", "'('", "')'", "'['", "']'", "'{'", "'}'", 
                     "'*'", "'/'", "'+'", "'-'", "'^'", "'='", "'=='", "'!='", 
                     "'<'", "'<='", "'>'", "'>='", "'%@'", "'%+'", "'%*'", 
                     "'%-'", "<INVALID>", "<INVALID>", "<INVALID>", "'1j'", 
                     "'abs'", "'sin'", "'cos'", "'tan'", "'exp'", "'log'", 
                     "'sinh'", "'cosh'", "'tanh'", "'atan'", "'acos'", "'asin'", 
                     "'atanh'", "'asinh'", "'acosh'", "'heaviside'", "'conj'", 
                     "'real'", "'imag'", "'atan2'", "'%I'", "'%X'", "'%Y'", 
                     "'%Z'", "'%C'", "'%A'", "'%J'" ]

    symbolicNames = [ "<INVALID>", "WHITESPACE", "EOL", "NEWLINE", "COMMENT", 
                      "EVOLVE", "MEASURE", "INITIALIZE", "IF", "ELSE", "WHILE", 
                      "WITH", "FOR", "BREAK", "CONTINUE", "AND", "AND2", 
                      "OR", "OR2", "NOT", "NOT2", "TRUE", "FALSE", "QUANTUMREGISTER", 
                      "MODEREGISTER", "COLON", "SEMICOLON", "COMMA", "LBRACKET", 
                      "RBRACKET", "SQUARELBRACKET", "SQUARERBRACKET", "LBRACE", 
                      "RBRACE", "MULT", "DIV", "PLUS", "MINUS", "POWER", 
                      "ASSIGN", "EQ", "NEQ", "LT", "LTE", "GT", "GTE", "AT", 
                      "OP_ADD", "OP_MUL", "OP_MINUS", "INT", "FLOAT", "MATH_VAR", 
                      "IMAG", "ABS", "SIN", "COS", "TAN", "EXP", "LOG", 
                      "SINH", "COSH", "TANH", "ATAN", "ACOS", "ASIN", "ATANH", 
                      "ASINH", "ACOSH", "HEAVISIDE", "CONJ", "REAL", "IMAG_FN", 
                      "ATAN2", "PAULI_I", "PAULI_X", "PAULI_Y", "PAULI_Z", 
                      "CREATION", "ANNIHILATION", "IDENTITY_OP", "ID" ]

    RULE_program = 0
    RULE_statement = 1
    RULE_block = 2
    RULE_terminal = 3
    RULE_expr = 4
    RULE_cond = 5
    RULE_analog_list = 6
    RULE_declaration = 7
    RULE_access = 8
    RULE_analog_list_extract = 9
    RULE_break_stmt = 10
    RULE_continue_stmt = 11
    RULE_while_stmt = 12
    RULE_ifelse_stmt = 13
    RULE_quantum_register = 14
    RULE_mode_register = 15
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
    RULE_pauli_op = 28
    RULE_ladder_op = 29
    RULE_operator_terminal = 30
    RULE_math_terminal = 31
    RULE_func_names = 32
    RULE_pexpr = 33
    RULE_fexpr = 34
    RULE_aexpr = 35
    RULE_mexpr = 36
    RULE_uexpr = 37
    RULE_eexpr = 38

    ruleNames =  [ "program", "statement", "block", "terminal", "expr", 
                   "cond", "analog_list", "declaration", "access", "analog_list_extract", 
                   "break_stmt", "continue_stmt", "while_stmt", "ifelse_stmt", 
                   "quantum_register", "mode_register", "targets", "bool_and_op", 
                   "bool_or_op", "bool_not_op", "bool_eq_op", "bool_not_eq_op", 
                   "bool_lt_op", "bool_lte_op", "bool_gt_op", "bool_gte_op", 
                   "bool_literal", "comparators", "pauli_op", "ladder_op", 
                   "operator_terminal", "math_terminal", "func_names", "pexpr", 
                   "fexpr", "aexpr", "mexpr", "uexpr", "eexpr" ]

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
    AND2=16
    OR=17
    OR2=18
    NOT=19
    NOT2=20
    TRUE=21
    FALSE=22
    QUANTUMREGISTER=23
    MODEREGISTER=24
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
    AT=46
    OP_ADD=47
    OP_MUL=48
    OP_MINUS=49
    INT=50
    FLOAT=51
    MATH_VAR=52
    IMAG=53
    ABS=54
    SIN=55
    COS=56
    TAN=57
    EXP=58
    LOG=59
    SINH=60
    COSH=61
    TANH=62
    ATAN=63
    ACOS=64
    ASIN=65
    ATANH=66
    ASINH=67
    ACOSH=68
    HEAVISIDE=69
    CONJ=70
    REAL=71
    IMAG_FN=72
    ATAN2=73
    PAULI_I=74
    PAULI_X=75
    PAULI_Y=76
    PAULI_Z=77
    CREATION=78
    ANNIHILATION=79
    IDENTITY_OP=80
    ID=81

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
            self.state = 78
            self.block()
            self.state = 79
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
        self.enterRule(localctx, 2, self.RULE_statement)
        try:
            self.state = 87
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,0,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 81
                self.declaration()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 82
                self.while_stmt()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 83
                self.ifelse_stmt()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 84
                self.break_stmt()
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 85
                self.continue_stmt()
                pass

            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 86
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
        self.enterRule(localctx, 4, self.RULE_block)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 95
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,2,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    self.state = 93
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [5, 6, 7, 8, 10, 13, 14, 19, 20, 21, 22, 23, 24, 28, 30, 36, 37, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81]:
                        self.state = 89
                        self.statement()
                        self.state = 90
                        self.match(AnalogParser.EOL)
                        pass
                    elif token in [2]:
                        self.state = 92
                        self.match(AnalogParser.EOL)
                        pass
                    else:
                        raise NoViableAltException(self)
             
                self.state = 97
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,2,self._ctx)

            self.state = 99
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if (((_la) & ~0x3f) == 0 and ((1 << _la) & -1125692373178912) != 0) or ((((_la - 64)) & ~0x3f) == 0 and ((1 << (_la - 64)) & 262143) != 0):
                self.state = 98
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

        def mode_register(self):
            return self.getTypedRuleContext(AnalogParser.Mode_registerContext,0)


        def quantum_register(self):
            return self.getTypedRuleContext(AnalogParser.Quantum_registerContext,0)


        def operator_terminal(self):
            return self.getTypedRuleContext(AnalogParser.Operator_terminalContext,0)


        def math_terminal(self):
            return self.getTypedRuleContext(AnalogParser.Math_terminalContext,0)


        def bool_literal(self):
            return self.getTypedRuleContext(AnalogParser.Bool_literalContext,0)


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
        self.enterRule(localctx, 6, self.RULE_terminal)
        try:
            self.state = 106
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [24]:
                self.enterOuterAlt(localctx, 1)
                self.state = 101
                self.mode_register()
                pass
            elif token in [23]:
                self.enterOuterAlt(localctx, 2)
                self.state = 102
                self.quantum_register()
                pass
            elif token in [74, 75, 76, 77, 78, 79, 80]:
                self.enterOuterAlt(localctx, 3)
                self.state = 103
                self.operator_terminal()
                pass
            elif token in [5, 6, 7, 28, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 81]:
                self.enterOuterAlt(localctx, 4)
                self.state = 104
                self.math_terminal()
                pass
            elif token in [21, 22]:
                self.enterOuterAlt(localctx, 5)
                self.state = 105
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
                return self.getTypedRuleContexts(AnalogParser.AexprContext)
            else:
                return self.getTypedRuleContext(AnalogParser.AexprContext,i)


        def comparators(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AnalogParser.ComparatorsContext)
            else:
                return self.getTypedRuleContext(AnalogParser.ComparatorsContext,i)


        def bool_not_op(self):
            return self.getTypedRuleContext(AnalogParser.Bool_not_opContext,0)


        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AnalogParser.ExprContext)
            else:
                return self.getTypedRuleContext(AnalogParser.ExprContext,i)


        def LBRACKET(self):
            return self.getToken(AnalogParser.LBRACKET, 0)

        def RBRACKET(self):
            return self.getToken(AnalogParser.RBRACKET, 0)

        def analog_list_extract(self):
            return self.getTypedRuleContext(AnalogParser.Analog_list_extractContext,0)


        def analog_list(self):
            return self.getTypedRuleContext(AnalogParser.Analog_listContext,0)


        def terminal(self):
            return self.getTypedRuleContext(AnalogParser.TerminalContext,0)


        def bool_and_op(self):
            return self.getTypedRuleContext(AnalogParser.Bool_and_opContext,0)


        def bool_or_op(self):
            return self.getTypedRuleContext(AnalogParser.Bool_or_opContext,0)


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



    def expr(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = AnalogParser.ExprContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 8
        self.enterRecursionRule(localctx, 8, self.RULE_expr, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 128
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,6,self._ctx)
            if la_ == 1:
                self.state = 109
                self.aexpr(0)
                self.state = 113 
                self._errHandler.sync(self)
                _alt = 1
                while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                    if _alt == 1:
                        self.state = 110
                        self.comparators()
                        self.state = 111
                        self.aexpr(0)

                    else:
                        raise NoViableAltException(self)
                    self.state = 115 
                    self._errHandler.sync(self)
                    _alt = self._interp.adaptivePredict(self._input,5,self._ctx)

                pass

            elif la_ == 2:
                self.state = 117
                self.bool_not_op()
                self.state = 118
                self.expr(6)
                pass

            elif la_ == 3:
                self.state = 120
                self.match(AnalogParser.LBRACKET)
                self.state = 121
                self.expr(0)
                self.state = 122
                self.match(AnalogParser.RBRACKET)
                pass

            elif la_ == 4:
                self.state = 124
                self.analog_list_extract()
                pass

            elif la_ == 5:
                self.state = 125
                self.analog_list()
                pass

            elif la_ == 6:
                self.state = 126
                self.terminal()
                pass

            elif la_ == 7:
                self.state = 127
                self.aexpr(0)
                pass


            self._ctx.stop = self._input.LT(-1)
            self.state = 139
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,8,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = AnalogParser.ExprContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_expr)
                    self.state = 130
                    if not self.precpred(self._ctx, 7):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 7)")
                    self.state = 133
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [15, 16]:
                        self.state = 131
                        self.bool_and_op()
                        pass
                    elif token in [17, 18]:
                        self.state = 132
                        self.bool_or_op()
                        pass
                    else:
                        raise NoViableAltException(self)

                    self.state = 135
                    self.expr(8) 
                self.state = 141
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
            return self.getTypedRuleContext(AnalogParser.ExprContext,0)


        def getRuleIndex(self):
            return AnalogParser.RULE_cond

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

        localctx = AnalogParser.CondContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_cond)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 142
            self.expr(0)
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
        self.enterRule(localctx, 12, self.RULE_analog_list)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 144
            self.match(AnalogParser.SQUARELBRACKET)
            self.state = 146
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if (((_la) & ~0x3f) == 0 and ((1 << _la) & -1125692373204768) != 0) or ((((_la - 64)) & ~0x3f) == 0 and ((1 << (_la - 64)) & 262143) != 0):
                self.state = 145
                self.expr(0)


            self.state = 152
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==27:
                self.state = 148
                self.match(AnalogParser.COMMA)
                self.state = 149
                self.expr(0)
                self.state = 154
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 155
            self.match(AnalogParser.SQUARERBRACKET)
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
        self.enterRule(localctx, 14, self.RULE_declaration)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 157
            self.match(AnalogParser.ID)
            self.state = 158
            self.match(AnalogParser.ASSIGN)
            self.state = 159
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
        self.enterRule(localctx, 16, self.RULE_access)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 161
            self.match(AnalogParser.ID)
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

        def INT(self):
            return self.getToken(AnalogParser.INT, 0)

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
        self.enterRule(localctx, 18, self.RULE_analog_list_extract)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 163
            self.access()
            self.state = 164
            self.match(AnalogParser.SQUARELBRACKET)
            self.state = 165
            self.match(AnalogParser.INT)
            self.state = 166
            self.match(AnalogParser.SQUARERBRACKET)
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
        self.enterRule(localctx, 20, self.RULE_break_stmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 168
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
        self.enterRule(localctx, 22, self.RULE_continue_stmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 170
            self.match(AnalogParser.CONTINUE)
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

        def cond(self):
            return self.getTypedRuleContext(AnalogParser.CondContext,0)


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
        self.enterRule(localctx, 24, self.RULE_while_stmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 172
            self.match(AnalogParser.WHILE)
            self.state = 173
            self.match(AnalogParser.LBRACKET)
            self.state = 174
            self.cond()
            self.state = 175
            self.match(AnalogParser.RBRACKET)
            self.state = 176
            self.match(AnalogParser.LBRACE)
            self.state = 177
            self.block()
            self.state = 178
            self.match(AnalogParser.RBRACE)
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

        def cond(self):
            return self.getTypedRuleContext(AnalogParser.CondContext,0)


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
        self.enterRule(localctx, 26, self.RULE_ifelse_stmt)
        self._la = 0 # Token type
        try:
            self.state = 203
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,12,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 180
                self.match(AnalogParser.IF)
                self.state = 181
                self.match(AnalogParser.LBRACKET)
                self.state = 182
                self.cond()
                self.state = 183
                self.match(AnalogParser.RBRACKET)
                self.state = 184
                self.match(AnalogParser.LBRACE)
                self.state = 185
                self.block()
                self.state = 186
                self.match(AnalogParser.RBRACE)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 188
                self.match(AnalogParser.IF)
                self.state = 189
                self.match(AnalogParser.LBRACKET)
                self.state = 190
                self.cond()
                self.state = 191
                self.match(AnalogParser.RBRACKET)
                self.state = 192
                self.match(AnalogParser.LBRACE)
                self.state = 193
                self.block()
                self.state = 194
                self.match(AnalogParser.RBRACE)
                self.state = 196
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==2:
                    self.state = 195
                    self.match(AnalogParser.EOL)


                self.state = 198
                self.match(AnalogParser.ELSE)
                self.state = 199
                self.match(AnalogParser.LBRACE)
                self.state = 200
                self.block()
                self.state = 201
                self.match(AnalogParser.RBRACE)
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
        self.enterRule(localctx, 28, self.RULE_quantum_register)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 205
            self.match(AnalogParser.QUANTUMREGISTER)
            self.state = 206
            self.match(AnalogParser.LBRACKET)
            self.state = 207
            self.match(AnalogParser.INT)
            self.state = 208
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
        self.enterRule(localctx, 30, self.RULE_mode_register)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 210
            self.match(AnalogParser.MODEREGISTER)
            self.state = 211
            self.match(AnalogParser.LBRACKET)
            self.state = 212
            self.match(AnalogParser.INT)
            self.state = 213
            self.match(AnalogParser.RBRACKET)
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
            return self.getTypedRuleContext(AnalogParser.ExprContext,0)


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
        self.enterRule(localctx, 32, self.RULE_targets)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 215
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
        self.enterRule(localctx, 34, self.RULE_bool_and_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 217
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
        self.enterRule(localctx, 36, self.RULE_bool_or_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 219
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
        self.enterRule(localctx, 38, self.RULE_bool_not_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 221
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


    class Bool_eq_opContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EQ(self):
            return self.getToken(AnalogParser.EQ, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_bool_eq_op

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

        localctx = AnalogParser.Bool_eq_opContext(self, self._ctx, self.state)
        self.enterRule(localctx, 40, self.RULE_bool_eq_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 223
            self.match(AnalogParser.EQ)
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
            return self.getToken(AnalogParser.NEQ, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_bool_not_eq_op

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

        localctx = AnalogParser.Bool_not_eq_opContext(self, self._ctx, self.state)
        self.enterRule(localctx, 42, self.RULE_bool_not_eq_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 225
            self.match(AnalogParser.NEQ)
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
            return self.getToken(AnalogParser.LT, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_bool_lt_op

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

        localctx = AnalogParser.Bool_lt_opContext(self, self._ctx, self.state)
        self.enterRule(localctx, 44, self.RULE_bool_lt_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 227
            self.match(AnalogParser.LT)
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
            return self.getToken(AnalogParser.LTE, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_bool_lte_op

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

        localctx = AnalogParser.Bool_lte_opContext(self, self._ctx, self.state)
        self.enterRule(localctx, 46, self.RULE_bool_lte_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 229
            self.match(AnalogParser.LTE)
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
            return self.getToken(AnalogParser.GT, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_bool_gt_op

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

        localctx = AnalogParser.Bool_gt_opContext(self, self._ctx, self.state)
        self.enterRule(localctx, 48, self.RULE_bool_gt_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 231
            self.match(AnalogParser.GT)
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
            return self.getToken(AnalogParser.GTE, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_bool_gte_op

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

        localctx = AnalogParser.Bool_gte_opContext(self, self._ctx, self.state)
        self.enterRule(localctx, 50, self.RULE_bool_gte_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 233
            self.match(AnalogParser.GTE)
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
        self.enterRule(localctx, 52, self.RULE_bool_literal)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 235
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


    class ComparatorsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def bool_eq_op(self):
            return self.getTypedRuleContext(AnalogParser.Bool_eq_opContext,0)


        def bool_not_eq_op(self):
            return self.getTypedRuleContext(AnalogParser.Bool_not_eq_opContext,0)


        def bool_lt_op(self):
            return self.getTypedRuleContext(AnalogParser.Bool_lt_opContext,0)


        def bool_lte_op(self):
            return self.getTypedRuleContext(AnalogParser.Bool_lte_opContext,0)


        def bool_gt_op(self):
            return self.getTypedRuleContext(AnalogParser.Bool_gt_opContext,0)


        def bool_gte_op(self):
            return self.getTypedRuleContext(AnalogParser.Bool_gte_opContext,0)


        def getRuleIndex(self):
            return AnalogParser.RULE_comparators

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

        localctx = AnalogParser.ComparatorsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 54, self.RULE_comparators)
        try:
            self.state = 243
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [40]:
                self.enterOuterAlt(localctx, 1)
                self.state = 237
                self.bool_eq_op()
                pass
            elif token in [41]:
                self.enterOuterAlt(localctx, 2)
                self.state = 238
                self.bool_not_eq_op()
                pass
            elif token in [42]:
                self.enterOuterAlt(localctx, 3)
                self.state = 239
                self.bool_lt_op()
                pass
            elif token in [43]:
                self.enterOuterAlt(localctx, 4)
                self.state = 240
                self.bool_lte_op()
                pass
            elif token in [44]:
                self.enterOuterAlt(localctx, 5)
                self.state = 241
                self.bool_gt_op()
                pass
            elif token in [45]:
                self.enterOuterAlt(localctx, 6)
                self.state = 242
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
        self.enterRule(localctx, 56, self.RULE_pauli_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 245
            _la = self._input.LA(1)
            if not(((((_la - 74)) & ~0x3f) == 0 and ((1 << (_la - 74)) & 15) != 0)):
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
        self.enterRule(localctx, 58, self.RULE_ladder_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 247
            _la = self._input.LA(1)
            if not(((((_la - 78)) & ~0x3f) == 0 and ((1 << (_la - 78)) & 7) != 0)):
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
        self.enterRule(localctx, 60, self.RULE_operator_terminal)
        try:
            self.state = 251
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [74, 75, 76, 77]:
                self.enterOuterAlt(localctx, 1)
                self.state = 249
                self.pauli_op()
                pass
            elif token in [78, 79, 80]:
                self.enterOuterAlt(localctx, 2)
                self.state = 250
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

        def access(self):
            return self.getTypedRuleContext(AnalogParser.AccessContext,0)


        def pexpr(self):
            return self.getTypedRuleContext(AnalogParser.PexprContext,0)


        def fexpr(self):
            return self.getTypedRuleContext(AnalogParser.FexprContext,0)


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
        self.enterRule(localctx, 62, self.RULE_math_terminal)
        try:
            self.state = 260
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [50]:
                self.enterOuterAlt(localctx, 1)
                self.state = 253
                self.match(AnalogParser.INT)
                pass
            elif token in [51]:
                self.enterOuterAlt(localctx, 2)
                self.state = 254
                self.match(AnalogParser.FLOAT)
                pass
            elif token in [52]:
                self.enterOuterAlt(localctx, 3)
                self.state = 255
                self.match(AnalogParser.MATH_VAR)
                pass
            elif token in [53]:
                self.enterOuterAlt(localctx, 4)
                self.state = 256
                self.match(AnalogParser.IMAG)
                pass
            elif token in [81]:
                self.enterOuterAlt(localctx, 5)
                self.state = 257
                self.access()
                pass
            elif token in [28]:
                self.enterOuterAlt(localctx, 6)
                self.state = 258
                self.pexpr()
                pass
            elif token in [5, 6, 7, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73]:
                self.enterOuterAlt(localctx, 7)
                self.state = 259
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


    class Func_namesContext(ParserRuleContext):
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

        def EVOLVE(self):
            return self.getToken(AnalogParser.EVOLVE, 0)

        def MEASURE(self):
            return self.getToken(AnalogParser.MEASURE, 0)

        def INITIALIZE(self):
            return self.getToken(AnalogParser.INITIALIZE, 0)

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
        self.enterRule(localctx, 64, self.RULE_func_names)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 262
            _la = self._input.LA(1)
            if not((((_la) & ~0x3f) == 0 and ((1 << _la) & -18014398509481760) != 0) or ((((_la - 64)) & ~0x3f) == 0 and ((1 << (_la - 64)) & 1023) != 0)):
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
            return self.getToken(AnalogParser.LBRACKET, 0)

        def aexpr(self):
            return self.getTypedRuleContext(AnalogParser.AexprContext,0)


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
        self.enterRule(localctx, 66, self.RULE_pexpr)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 264
            self.match(AnalogParser.LBRACKET)
            self.state = 265
            self.aexpr(0)
            self.state = 266
            self.match(AnalogParser.RBRACKET)
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
            return self.getTypedRuleContext(AnalogParser.Func_namesContext,0)


        def LBRACKET(self):
            return self.getToken(AnalogParser.LBRACKET, 0)

        def RBRACKET(self):
            return self.getToken(AnalogParser.RBRACKET, 0)

        def aexpr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AnalogParser.AexprContext)
            else:
                return self.getTypedRuleContext(AnalogParser.AexprContext,i)


        def COMMA(self, i:int=None):
            if i is None:
                return self.getTokens(AnalogParser.COMMA)
            else:
                return self.getToken(AnalogParser.COMMA, i)

        def getRuleIndex(self):
            return AnalogParser.RULE_fexpr

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

        localctx = AnalogParser.FexprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 68, self.RULE_fexpr)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 268
            self.func_names()
            self.state = 269
            self.match(AnalogParser.LBRACKET)
            self.state = 278
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if (((_la) & ~0x3f) == 0 and ((1 << _la) & -1125693448519456) != 0) or ((((_la - 64)) & ~0x3f) == 0 and ((1 << (_la - 64)) & 262143) != 0):
                self.state = 270
                self.aexpr(0)
                self.state = 275
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la==27:
                    self.state = 271
                    self.match(AnalogParser.COMMA)
                    self.state = 272
                    self.aexpr(0)
                    self.state = 277
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)



            self.state = 280
            self.match(AnalogParser.RBRACKET)
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
            return self.getTypedRuleContext(AnalogParser.MexprContext,0)


        def aexpr(self):
            return self.getTypedRuleContext(AnalogParser.AexprContext,0)


        def PLUS(self):
            return self.getToken(AnalogParser.PLUS, 0)

        def MINUS(self):
            return self.getToken(AnalogParser.MINUS, 0)

        def OP_ADD(self):
            return self.getToken(AnalogParser.OP_ADD, 0)

        def OP_MINUS(self):
            return self.getToken(AnalogParser.OP_MINUS, 0)

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
        _startState = 70
        self.enterRecursionRule(localctx, 70, self.RULE_aexpr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 283
            self.mexpr(0)
            self._ctx.stop = self._input.LT(-1)
            self.state = 290
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,18,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = AnalogParser.AexprContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_aexpr)
                    self.state = 285
                    if not self.precpred(self._ctx, 1):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 1)")
                    self.state = 286
                    _la = self._input.LA(1)
                    if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 703893600206848) != 0)):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 287
                    self.mexpr(0) 
                self.state = 292
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,18,self._ctx)

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
            return self.getTypedRuleContext(AnalogParser.UexprContext,0)


        def mexpr(self):
            return self.getTypedRuleContext(AnalogParser.MexprContext,0)


        def MULT(self):
            return self.getToken(AnalogParser.MULT, 0)

        def DIV(self):
            return self.getToken(AnalogParser.DIV, 0)

        def OP_MUL(self):
            return self.getToken(AnalogParser.OP_MUL, 0)

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
        _startState = 72
        self.enterRecursionRule(localctx, 72, self.RULE_mexpr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 294
            self.uexpr()
            self._ctx.stop = self._input.LT(-1)
            self.state = 301
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,19,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = AnalogParser.MexprContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_mexpr)
                    self.state = 296
                    if not self.precpred(self._ctx, 1):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 1)")
                    self.state = 297
                    _la = self._input.LA(1)
                    if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 351895260495872) != 0)):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 298
                    self.uexpr() 
                self.state = 303
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
        self.enterRule(localctx, 74, self.RULE_uexpr)
        self._la = 0 # Token type
        try:
            self.state = 307
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [5, 6, 7, 21, 22, 23, 24, 28, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81]:
                self.enterOuterAlt(localctx, 1)
                self.state = 304
                self.eexpr()
                pass
            elif token in [36, 37]:
                self.enterOuterAlt(localctx, 2)
                self.state = 305
                _la = self._input.LA(1)
                if not(_la==36 or _la==37):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 306
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
            return self.getTypedRuleContext(AnalogParser.TerminalContext,0)


        def POWER(self):
            return self.getToken(AnalogParser.POWER, 0)

        def uexpr(self):
            return self.getTypedRuleContext(AnalogParser.UexprContext,0)


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




    def eexpr(self):

        localctx = AnalogParser.EexprContext(self, self._ctx, self.state)
        self.enterRule(localctx, 76, self.RULE_eexpr)
        try:
            self.state = 314
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,21,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 309
                self.terminal()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 310
                self.terminal()
                self.state = 311
                self.match(AnalogParser.POWER)
                self.state = 312
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
        self._predicates[35] = self.aexpr_sempred
        self._predicates[36] = self.mexpr_sempred
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
         




