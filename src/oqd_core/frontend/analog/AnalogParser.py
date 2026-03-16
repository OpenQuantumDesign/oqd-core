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
        4,1,81,328,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,20,
        7,20,2,21,7,21,2,22,7,22,2,23,7,23,2,24,7,24,2,25,7,25,2,26,7,26,
        2,27,7,27,2,28,7,28,2,29,7,29,2,30,7,30,2,31,7,31,2,32,7,32,2,33,
        7,33,2,34,7,34,2,35,7,35,2,36,7,36,2,37,7,37,2,38,7,38,2,39,7,39,
        2,40,7,40,2,41,7,41,1,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        3,1,96,8,1,1,2,1,2,1,2,1,2,5,2,102,8,2,10,2,12,2,105,9,2,1,2,3,2,
        108,8,2,1,3,1,3,1,3,1,3,1,3,3,3,115,8,3,1,4,1,4,1,4,1,4,1,4,3,4,
        122,8,4,1,5,1,5,1,6,1,6,3,6,128,8,6,1,6,1,6,5,6,132,8,6,10,6,12,
        6,135,9,6,1,6,1,6,1,7,1,7,1,7,1,7,1,8,1,8,1,9,1,9,1,9,1,9,1,9,1,
        10,1,10,1,11,1,11,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,12,1,13,1,
        13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,13,1,
        13,1,13,3,13,178,8,13,1,13,1,13,1,13,1,13,1,13,3,13,185,8,13,1,14,
        1,14,1,14,1,14,1,14,1,15,1,15,1,15,1,15,1,15,1,16,1,16,1,16,1,16,
        1,16,1,16,1,16,1,17,1,17,1,17,1,18,1,18,1,18,1,19,1,19,1,20,1,20,
        1,21,1,21,1,22,1,22,1,23,1,23,1,24,1,24,1,25,1,25,1,26,1,26,1,27,
        1,27,1,28,1,28,1,29,1,29,1,30,1,30,1,30,1,30,1,30,1,30,1,30,1,30,
        1,30,1,30,1,30,3,30,243,8,30,1,30,1,30,1,30,1,30,3,30,249,8,30,1,
        30,1,30,1,30,1,30,1,30,1,30,1,30,1,30,3,30,259,8,30,1,30,1,30,5,
        30,263,8,30,10,30,12,30,266,9,30,1,31,1,31,1,32,1,32,1,33,1,33,3,
        33,274,8,33,1,34,1,34,1,34,1,34,1,34,1,34,1,34,3,34,283,8,34,1,35,
        1,35,1,36,1,36,1,36,1,36,1,37,1,37,1,37,1,38,1,38,1,38,1,38,1,38,
        1,38,5,38,300,8,38,10,38,12,38,303,9,38,1,39,1,39,1,39,1,39,1,39,
        1,39,5,39,311,8,39,10,39,12,39,314,9,39,1,40,1,40,1,40,3,40,319,
        8,40,1,41,1,41,1,41,1,41,1,41,3,41,326,8,41,1,41,0,3,60,76,78,42,
        0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,
        46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,0,10,1,
        0,15,16,1,0,17,18,1,0,19,20,1,0,21,22,1,0,74,77,1,0,78,80,1,0,54,
        73,3,0,36,37,47,47,49,49,3,0,34,35,46,46,48,48,1,0,36,37,330,0,84,
        1,0,0,0,2,95,1,0,0,0,4,103,1,0,0,0,6,114,1,0,0,0,8,121,1,0,0,0,10,
        123,1,0,0,0,12,125,1,0,0,0,14,138,1,0,0,0,16,142,1,0,0,0,18,144,
        1,0,0,0,20,149,1,0,0,0,22,151,1,0,0,0,24,153,1,0,0,0,26,184,1,0,
        0,0,28,186,1,0,0,0,30,191,1,0,0,0,32,196,1,0,0,0,34,203,1,0,0,0,
        36,206,1,0,0,0,38,209,1,0,0,0,40,211,1,0,0,0,42,213,1,0,0,0,44,215,
        1,0,0,0,46,217,1,0,0,0,48,219,1,0,0,0,50,221,1,0,0,0,52,223,1,0,
        0,0,54,225,1,0,0,0,56,227,1,0,0,0,58,229,1,0,0,0,60,242,1,0,0,0,
        62,267,1,0,0,0,64,269,1,0,0,0,66,273,1,0,0,0,68,282,1,0,0,0,70,284,
        1,0,0,0,72,286,1,0,0,0,74,290,1,0,0,0,76,293,1,0,0,0,78,304,1,0,
        0,0,80,318,1,0,0,0,82,325,1,0,0,0,84,85,3,4,2,0,85,86,5,0,0,1,86,
        1,1,0,0,0,87,96,3,14,7,0,88,96,3,32,16,0,89,96,3,34,17,0,90,96,3,
        36,18,0,91,96,3,24,12,0,92,96,3,26,13,0,93,96,3,20,10,0,94,96,3,
        22,11,0,95,87,1,0,0,0,95,88,1,0,0,0,95,89,1,0,0,0,95,90,1,0,0,0,
        95,91,1,0,0,0,95,92,1,0,0,0,95,93,1,0,0,0,95,94,1,0,0,0,96,3,1,0,
        0,0,97,98,3,2,1,0,98,99,5,2,0,0,99,102,1,0,0,0,100,102,5,2,0,0,101,
        97,1,0,0,0,101,100,1,0,0,0,102,105,1,0,0,0,103,101,1,0,0,0,103,104,
        1,0,0,0,104,107,1,0,0,0,105,103,1,0,0,0,106,108,3,2,1,0,107,106,
        1,0,0,0,107,108,1,0,0,0,108,5,1,0,0,0,109,115,3,30,15,0,110,115,
        3,28,14,0,111,115,3,66,33,0,112,115,3,68,34,0,113,115,3,16,8,0,114,
        109,1,0,0,0,114,110,1,0,0,0,114,111,1,0,0,0,114,112,1,0,0,0,114,
        113,1,0,0,0,115,7,1,0,0,0,116,122,3,18,9,0,117,122,3,12,6,0,118,
        122,3,6,3,0,119,122,3,76,38,0,120,122,3,58,29,0,121,116,1,0,0,0,
        121,117,1,0,0,0,121,118,1,0,0,0,121,119,1,0,0,0,121,120,1,0,0,0,
        122,9,1,0,0,0,123,124,3,60,30,0,124,11,1,0,0,0,125,127,5,30,0,0,
        126,128,3,8,4,0,127,126,1,0,0,0,127,128,1,0,0,0,128,133,1,0,0,0,
        129,130,5,27,0,0,130,132,3,8,4,0,131,129,1,0,0,0,132,135,1,0,0,0,
        133,131,1,0,0,0,133,134,1,0,0,0,134,136,1,0,0,0,135,133,1,0,0,0,
        136,137,5,31,0,0,137,13,1,0,0,0,138,139,5,81,0,0,139,140,5,39,0,
        0,140,141,3,8,4,0,141,15,1,0,0,0,142,143,5,81,0,0,143,17,1,0,0,0,
        144,145,3,16,8,0,145,146,5,30,0,0,146,147,5,50,0,0,147,148,5,31,
        0,0,148,19,1,0,0,0,149,150,5,13,0,0,150,21,1,0,0,0,151,152,5,14,
        0,0,152,23,1,0,0,0,153,154,5,10,0,0,154,155,5,28,0,0,155,156,3,10,
        5,0,156,157,5,29,0,0,157,158,5,32,0,0,158,159,3,4,2,0,159,160,5,
        33,0,0,160,25,1,0,0,0,161,162,5,8,0,0,162,163,5,28,0,0,163,164,3,
        10,5,0,164,165,5,29,0,0,165,166,5,32,0,0,166,167,3,4,2,0,167,168,
        5,33,0,0,168,185,1,0,0,0,169,170,5,8,0,0,170,171,5,28,0,0,171,172,
        3,10,5,0,172,173,5,29,0,0,173,174,5,32,0,0,174,175,3,4,2,0,175,177,
        5,33,0,0,176,178,5,2,0,0,177,176,1,0,0,0,177,178,1,0,0,0,178,179,
        1,0,0,0,179,180,5,9,0,0,180,181,5,32,0,0,181,182,3,4,2,0,182,183,
        5,33,0,0,183,185,1,0,0,0,184,161,1,0,0,0,184,169,1,0,0,0,185,27,
        1,0,0,0,186,187,5,23,0,0,187,188,5,28,0,0,188,189,5,50,0,0,189,190,
        5,29,0,0,190,29,1,0,0,0,191,192,5,24,0,0,192,193,5,28,0,0,193,194,
        5,50,0,0,194,195,5,29,0,0,195,31,1,0,0,0,196,197,5,5,0,0,197,198,
        3,38,19,0,198,199,5,11,0,0,199,200,3,8,4,0,200,201,5,12,0,0,201,
        202,3,8,4,0,202,33,1,0,0,0,203,204,5,6,0,0,204,205,3,38,19,0,205,
        35,1,0,0,0,206,207,5,7,0,0,207,208,3,38,19,0,208,37,1,0,0,0,209,
        210,3,8,4,0,210,39,1,0,0,0,211,212,7,0,0,0,212,41,1,0,0,0,213,214,
        7,1,0,0,214,43,1,0,0,0,215,216,7,2,0,0,216,45,1,0,0,0,217,218,5,
        40,0,0,218,47,1,0,0,0,219,220,5,41,0,0,220,49,1,0,0,0,221,222,5,
        42,0,0,222,51,1,0,0,0,223,224,5,43,0,0,224,53,1,0,0,0,225,226,5,
        44,0,0,226,55,1,0,0,0,227,228,5,45,0,0,228,57,1,0,0,0,229,230,7,
        3,0,0,230,59,1,0,0,0,231,232,6,30,-1,0,232,233,3,44,22,0,233,234,
        3,60,30,5,234,243,1,0,0,0,235,243,3,58,29,0,236,243,3,16,8,0,237,
        243,3,6,3,0,238,239,5,28,0,0,239,240,3,60,30,0,240,241,5,29,0,0,
        241,243,1,0,0,0,242,231,1,0,0,0,242,235,1,0,0,0,242,236,1,0,0,0,
        242,237,1,0,0,0,242,238,1,0,0,0,243,264,1,0,0,0,244,248,10,7,0,0,
        245,249,3,40,20,0,246,249,3,42,21,0,247,249,3,46,23,0,248,245,1,
        0,0,0,248,246,1,0,0,0,248,247,1,0,0,0,249,250,1,0,0,0,250,251,3,
        60,30,8,251,263,1,0,0,0,252,258,10,6,0,0,253,259,3,48,24,0,254,259,
        3,50,25,0,255,259,3,52,26,0,256,259,3,54,27,0,257,259,3,56,28,0,
        258,253,1,0,0,0,258,254,1,0,0,0,258,255,1,0,0,0,258,256,1,0,0,0,
        258,257,1,0,0,0,259,260,1,0,0,0,260,261,3,60,30,7,261,263,1,0,0,
        0,262,244,1,0,0,0,262,252,1,0,0,0,263,266,1,0,0,0,264,262,1,0,0,
        0,264,265,1,0,0,0,265,61,1,0,0,0,266,264,1,0,0,0,267,268,7,4,0,0,
        268,63,1,0,0,0,269,270,7,5,0,0,270,65,1,0,0,0,271,274,3,62,31,0,
        272,274,3,64,32,0,273,271,1,0,0,0,273,272,1,0,0,0,274,67,1,0,0,0,
        275,283,5,50,0,0,276,283,5,51,0,0,277,283,5,52,0,0,278,283,5,53,
        0,0,279,283,5,81,0,0,280,283,3,72,36,0,281,283,3,74,37,0,282,275,
        1,0,0,0,282,276,1,0,0,0,282,277,1,0,0,0,282,278,1,0,0,0,282,279,
        1,0,0,0,282,280,1,0,0,0,282,281,1,0,0,0,283,69,1,0,0,0,284,285,7,
        6,0,0,285,71,1,0,0,0,286,287,5,28,0,0,287,288,3,76,38,0,288,289,
        5,29,0,0,289,73,1,0,0,0,290,291,3,70,35,0,291,292,3,72,36,0,292,
        75,1,0,0,0,293,294,6,38,-1,0,294,295,3,78,39,0,295,301,1,0,0,0,296,
        297,10,1,0,0,297,298,7,7,0,0,298,300,3,78,39,0,299,296,1,0,0,0,300,
        303,1,0,0,0,301,299,1,0,0,0,301,302,1,0,0,0,302,77,1,0,0,0,303,301,
        1,0,0,0,304,305,6,39,-1,0,305,306,3,80,40,0,306,312,1,0,0,0,307,
        308,10,1,0,0,308,309,7,8,0,0,309,311,3,80,40,0,310,307,1,0,0,0,311,
        314,1,0,0,0,312,310,1,0,0,0,312,313,1,0,0,0,313,79,1,0,0,0,314,312,
        1,0,0,0,315,319,3,82,41,0,316,317,7,9,0,0,317,319,3,82,41,0,318,
        315,1,0,0,0,318,316,1,0,0,0,319,81,1,0,0,0,320,326,3,6,3,0,321,322,
        3,6,3,0,322,323,5,38,0,0,323,324,3,80,40,0,324,326,1,0,0,0,325,320,
        1,0,0,0,325,321,1,0,0,0,326,83,1,0,0,0,21,95,101,103,107,114,121,
        127,133,177,184,242,248,258,262,264,273,282,301,312,318,325
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
    RULE_quantum_register = 14
    RULE_mode_register = 15
    RULE_evolve_stmt = 16
    RULE_measure_stmt = 17
    RULE_init_stmt = 18
    RULE_targets = 19
    RULE_bool_and_op = 20
    RULE_bool_or_op = 21
    RULE_bool_not_op = 22
    RULE_bool_eq_op = 23
    RULE_bool_not_eq_op = 24
    RULE_bool_lt_op = 25
    RULE_bool_lte_op = 26
    RULE_bool_gt_op = 27
    RULE_bool_gte_op = 28
    RULE_bool_literal = 29
    RULE_bool_expr = 30
    RULE_pauli_op = 31
    RULE_ladder_op = 32
    RULE_operator_terminal = 33
    RULE_math_terminal = 34
    RULE_math_func_name = 35
    RULE_pexpr = 36
    RULE_fexpr = 37
    RULE_aexpr = 38
    RULE_mexpr = 39
    RULE_uexpr = 40
    RULE_eexpr = 41

    ruleNames =  [ "program", "statement", "block", "atom", "expr", "cond", 
                   "my_list", "declaration", "access", "extract", "break_stmt", 
                   "continue_stmt", "while_stmt", "ifelse_stmt", "quantum_register", 
                   "mode_register", "evolve_stmt", "measure_stmt", "init_stmt", 
                   "targets", "bool_and_op", "bool_or_op", "bool_not_op", 
                   "bool_eq_op", "bool_not_eq_op", "bool_lt_op", "bool_lte_op", 
                   "bool_gt_op", "bool_gte_op", "bool_literal", "bool_expr", 
                   "pauli_op", "ladder_op", "operator_terminal", "math_terminal", 
                   "math_func_name", "pexpr", "fexpr", "aexpr", "mexpr", 
                   "uexpr", "eexpr" ]

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
            self.state = 84
            self.block()
            self.state = 85
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


        def while_stmt(self):
            return self.getTypedRuleContext(AnalogParser.While_stmtContext,0)


        def ifelse_stmt(self):
            return self.getTypedRuleContext(AnalogParser.Ifelse_stmtContext,0)


        def break_stmt(self):
            return self.getTypedRuleContext(AnalogParser.Break_stmtContext,0)


        def continue_stmt(self):
            return self.getTypedRuleContext(AnalogParser.Continue_stmtContext,0)


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
            self.state = 95
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [81]:
                self.enterOuterAlt(localctx, 1)
                self.state = 87
                self.declaration()
                pass
            elif token in [5]:
                self.enterOuterAlt(localctx, 2)
                self.state = 88
                self.evolve_stmt()
                pass
            elif token in [6]:
                self.enterOuterAlt(localctx, 3)
                self.state = 89
                self.measure_stmt()
                pass
            elif token in [7]:
                self.enterOuterAlt(localctx, 4)
                self.state = 90
                self.init_stmt()
                pass
            elif token in [10]:
                self.enterOuterAlt(localctx, 5)
                self.state = 91
                self.while_stmt()
                pass
            elif token in [8]:
                self.enterOuterAlt(localctx, 6)
                self.state = 92
                self.ifelse_stmt()
                pass
            elif token in [13]:
                self.enterOuterAlt(localctx, 7)
                self.state = 93
                self.break_stmt()
                pass
            elif token in [14]:
                self.enterOuterAlt(localctx, 8)
                self.state = 94
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
            self.state = 103
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,2,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    self.state = 101
                    self._errHandler.sync(self)
                    token = self._input.LA(1)
                    if token in [5, 6, 7, 8, 10, 13, 14, 81]:
                        self.state = 97
                        self.statement()
                        self.state = 98
                        self.match(AnalogParser.EOL)
                        pass
                    elif token in [2]:
                        self.state = 100
                        self.match(AnalogParser.EOL)
                        pass
                    else:
                        raise NoViableAltException(self)
             
                self.state = 105
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,2,self._ctx)

            self.state = 107
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if (((_la) & ~0x3f) == 0 and ((1 << _la) & 26080) != 0) or _la==81:
                self.state = 106
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

        def mode_register(self):
            return self.getTypedRuleContext(AnalogParser.Mode_registerContext,0)


        def quantum_register(self):
            return self.getTypedRuleContext(AnalogParser.Quantum_registerContext,0)


        def operator_terminal(self):
            return self.getTypedRuleContext(AnalogParser.Operator_terminalContext,0)


        def math_terminal(self):
            return self.getTypedRuleContext(AnalogParser.Math_terminalContext,0)


        def access(self):
            return self.getTypedRuleContext(AnalogParser.AccessContext,0)


        def getRuleIndex(self):
            return AnalogParser.RULE_atom

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

        localctx = AnalogParser.AtomContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_atom)
        try:
            self.state = 114
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,4,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 109
                self.mode_register()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 110
                self.quantum_register()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 111
                self.operator_terminal()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 112
                self.math_terminal()
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 113
                self.access()
                pass


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

        def extract(self):
            return self.getTypedRuleContext(AnalogParser.ExtractContext,0)


        def my_list(self):
            return self.getTypedRuleContext(AnalogParser.My_listContext,0)


        def atom(self):
            return self.getTypedRuleContext(AnalogParser.AtomContext,0)


        def aexpr(self):
            return self.getTypedRuleContext(AnalogParser.AexprContext,0)


        def bool_literal(self):
            return self.getTypedRuleContext(AnalogParser.Bool_literalContext,0)


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
        self.enterRule(localctx, 8, self.RULE_expr)
        try:
            self.state = 121
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,5,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 116
                self.extract()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 117
                self.my_list()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 118
                self.atom()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 119
                self.aexpr(0)
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 120
                self.bool_literal()
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class CondContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def bool_expr(self):
            return self.getTypedRuleContext(AnalogParser.Bool_exprContext,0)


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
            self.state = 123
            self.bool_expr(0)
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
            self.state = 125
            self.match(AnalogParser.SQUARELBRACKET)
            self.state = 127
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if ((((_la - 21)) & ~0x3f) == 0 and ((1 << (_la - 21)) & 2305843008676921999) != 0):
                self.state = 126
                self.expr()


            self.state = 133
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==27:
                self.state = 129
                self.match(AnalogParser.COMMA)
                self.state = 130
                self.expr()
                self.state = 135
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 136
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
            self.state = 138
            self.match(AnalogParser.ID)
            self.state = 139
            self.match(AnalogParser.ASSIGN)
            self.state = 140
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
        self.enterRule(localctx, 16, self.RULE_access)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 142
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
        self.enterRule(localctx, 18, self.RULE_extract)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 144
            self.access()
            self.state = 145
            self.match(AnalogParser.SQUARELBRACKET)
            self.state = 146
            self.match(AnalogParser.INT)
            self.state = 147
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
            self.state = 149
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
            self.state = 151
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
            self.state = 153
            self.match(AnalogParser.WHILE)
            self.state = 154
            self.match(AnalogParser.LBRACKET)
            self.state = 155
            self.cond()
            self.state = 156
            self.match(AnalogParser.RBRACKET)
            self.state = 157
            self.match(AnalogParser.LBRACE)
            self.state = 158
            self.block()
            self.state = 159
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
            self.state = 184
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,9,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 161
                self.match(AnalogParser.IF)
                self.state = 162
                self.match(AnalogParser.LBRACKET)
                self.state = 163
                self.cond()
                self.state = 164
                self.match(AnalogParser.RBRACKET)
                self.state = 165
                self.match(AnalogParser.LBRACE)
                self.state = 166
                self.block()
                self.state = 167
                self.match(AnalogParser.RBRACE)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 169
                self.match(AnalogParser.IF)
                self.state = 170
                self.match(AnalogParser.LBRACKET)
                self.state = 171
                self.cond()
                self.state = 172
                self.match(AnalogParser.RBRACKET)
                self.state = 173
                self.match(AnalogParser.LBRACE)
                self.state = 174
                self.block()
                self.state = 175
                self.match(AnalogParser.RBRACE)
                self.state = 177
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==2:
                    self.state = 176
                    self.match(AnalogParser.EOL)


                self.state = 179
                self.match(AnalogParser.ELSE)
                self.state = 180
                self.match(AnalogParser.LBRACE)
                self.state = 181
                self.block()
                self.state = 182
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
            self.state = 186
            self.match(AnalogParser.QUANTUMREGISTER)
            self.state = 187
            self.match(AnalogParser.LBRACKET)
            self.state = 188
            self.match(AnalogParser.INT)
            self.state = 189
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
            self.state = 191
            self.match(AnalogParser.MODEREGISTER)
            self.state = 192
            self.match(AnalogParser.LBRACKET)
            self.state = 193
            self.match(AnalogParser.INT)
            self.state = 194
            self.match(AnalogParser.RBRACKET)
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

        def expr(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AnalogParser.ExprContext)
            else:
                return self.getTypedRuleContext(AnalogParser.ExprContext,i)


        def FOR(self):
            return self.getToken(AnalogParser.FOR, 0)

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
        self.enterRule(localctx, 32, self.RULE_evolve_stmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 196
            self.match(AnalogParser.EVOLVE)
            self.state = 197
            self.targets()
            self.state = 198
            self.match(AnalogParser.WITH)
            self.state = 199
            self.expr()
            self.state = 200
            self.match(AnalogParser.FOR)
            self.state = 201
            self.expr()
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
        self.enterRule(localctx, 34, self.RULE_measure_stmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 203
            self.match(AnalogParser.MEASURE)
            self.state = 204
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
        self.enterRule(localctx, 36, self.RULE_init_stmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 206
            self.match(AnalogParser.INITIALIZE)
            self.state = 207
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
        self.enterRule(localctx, 38, self.RULE_targets)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 209
            self.expr()
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
        self.enterRule(localctx, 40, self.RULE_bool_and_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 211
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
        self.enterRule(localctx, 42, self.RULE_bool_or_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 213
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
        self.enterRule(localctx, 44, self.RULE_bool_not_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 215
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
        self.enterRule(localctx, 46, self.RULE_bool_eq_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 217
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
        self.enterRule(localctx, 48, self.RULE_bool_not_eq_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 219
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
        self.enterRule(localctx, 50, self.RULE_bool_lt_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 221
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
        self.enterRule(localctx, 52, self.RULE_bool_lte_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 223
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
        self.enterRule(localctx, 54, self.RULE_bool_gt_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 225
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
        self.enterRule(localctx, 56, self.RULE_bool_gte_op)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 227
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
        self.enterRule(localctx, 58, self.RULE_bool_literal)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 229
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


        def bool_literal(self):
            return self.getTypedRuleContext(AnalogParser.Bool_literalContext,0)


        def access(self):
            return self.getTypedRuleContext(AnalogParser.AccessContext,0)


        def atom(self):
            return self.getTypedRuleContext(AnalogParser.AtomContext,0)


        def LBRACKET(self):
            return self.getToken(AnalogParser.LBRACKET, 0)

        def RBRACKET(self):
            return self.getToken(AnalogParser.RBRACKET, 0)

        def bool_and_op(self):
            return self.getTypedRuleContext(AnalogParser.Bool_and_opContext,0)


        def bool_or_op(self):
            return self.getTypedRuleContext(AnalogParser.Bool_or_opContext,0)


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
        _startState = 60
        self.enterRecursionRule(localctx, 60, self.RULE_bool_expr, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 242
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,10,self._ctx)
            if la_ == 1:
                self.state = 232
                self.bool_not_op()
                self.state = 233
                self.bool_expr(5)
                pass

            elif la_ == 2:
                self.state = 235
                self.bool_literal()
                pass

            elif la_ == 3:
                self.state = 236
                self.access()
                pass

            elif la_ == 4:
                self.state = 237
                self.atom()
                pass

            elif la_ == 5:
                self.state = 238
                self.match(AnalogParser.LBRACKET)
                self.state = 239
                self.bool_expr(0)
                self.state = 240
                self.match(AnalogParser.RBRACKET)
                pass


            self._ctx.stop = self._input.LT(-1)
            self.state = 264
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,14,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 262
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,13,self._ctx)
                    if la_ == 1:
                        localctx = AnalogParser.Bool_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_bool_expr)
                        self.state = 244
                        if not self.precpred(self._ctx, 7):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 7)")
                        self.state = 248
                        self._errHandler.sync(self)
                        token = self._input.LA(1)
                        if token in [15, 16]:
                            self.state = 245
                            self.bool_and_op()
                            pass
                        elif token in [17, 18]:
                            self.state = 246
                            self.bool_or_op()
                            pass
                        elif token in [40]:
                            self.state = 247
                            self.bool_eq_op()
                            pass
                        else:
                            raise NoViableAltException(self)

                        self.state = 250
                        self.bool_expr(8)
                        pass

                    elif la_ == 2:
                        localctx = AnalogParser.Bool_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_bool_expr)
                        self.state = 252
                        if not self.precpred(self._ctx, 6):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 6)")
                        self.state = 258
                        self._errHandler.sync(self)
                        token = self._input.LA(1)
                        if token in [41]:
                            self.state = 253
                            self.bool_not_eq_op()
                            pass
                        elif token in [42]:
                            self.state = 254
                            self.bool_lt_op()
                            pass
                        elif token in [43]:
                            self.state = 255
                            self.bool_lte_op()
                            pass
                        elif token in [44]:
                            self.state = 256
                            self.bool_gt_op()
                            pass
                        elif token in [45]:
                            self.state = 257
                            self.bool_gte_op()
                            pass
                        else:
                            raise NoViableAltException(self)

                        self.state = 260
                        self.bool_expr(7)
                        pass

             
                self.state = 266
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,14,self._ctx)

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
        self.enterRule(localctx, 62, self.RULE_pauli_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 267
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
        self.enterRule(localctx, 64, self.RULE_ladder_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 269
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
        self.enterRule(localctx, 66, self.RULE_operator_terminal)
        try:
            self.state = 273
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [74, 75, 76, 77]:
                self.enterOuterAlt(localctx, 1)
                self.state = 271
                self.pauli_op()
                pass
            elif token in [78, 79, 80]:
                self.enterOuterAlt(localctx, 2)
                self.state = 272
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

        def ID(self):
            return self.getToken(AnalogParser.ID, 0)

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
        self.enterRule(localctx, 68, self.RULE_math_terminal)
        try:
            self.state = 282
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [50]:
                self.enterOuterAlt(localctx, 1)
                self.state = 275
                self.match(AnalogParser.INT)
                pass
            elif token in [51]:
                self.enterOuterAlt(localctx, 2)
                self.state = 276
                self.match(AnalogParser.FLOAT)
                pass
            elif token in [52]:
                self.enterOuterAlt(localctx, 3)
                self.state = 277
                self.match(AnalogParser.MATH_VAR)
                pass
            elif token in [53]:
                self.enterOuterAlt(localctx, 4)
                self.state = 278
                self.match(AnalogParser.IMAG)
                pass
            elif token in [81]:
                self.enterOuterAlt(localctx, 5)
                self.state = 279
                self.match(AnalogParser.ID)
                pass
            elif token in [28]:
                self.enterOuterAlt(localctx, 6)
                self.state = 280
                self.pexpr()
                pass
            elif token in [54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73]:
                self.enterOuterAlt(localctx, 7)
                self.state = 281
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
        self.enterRule(localctx, 70, self.RULE_math_func_name)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 284
            _la = self._input.LA(1)
            if not(((((_la - 54)) & ~0x3f) == 0 and ((1 << (_la - 54)) & 1048575) != 0)):
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
        self.enterRule(localctx, 72, self.RULE_pexpr)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 286
            self.match(AnalogParser.LBRACKET)
            self.state = 287
            self.aexpr(0)
            self.state = 288
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

        def math_func_name(self):
            return self.getTypedRuleContext(AnalogParser.Math_func_nameContext,0)


        def pexpr(self):
            return self.getTypedRuleContext(AnalogParser.PexprContext,0)


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
        self.enterRule(localctx, 74, self.RULE_fexpr)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 290
            self.math_func_name()
            self.state = 291
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
        _startState = 76
        self.enterRecursionRule(localctx, 76, self.RULE_aexpr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 294
            self.mexpr(0)
            self._ctx.stop = self._input.LT(-1)
            self.state = 301
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,17,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = AnalogParser.AexprContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_aexpr)
                    self.state = 296
                    if not self.precpred(self._ctx, 1):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 1)")
                    self.state = 297
                    _la = self._input.LA(1)
                    if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 703893600206848) != 0)):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 298
                    self.mexpr(0) 
                self.state = 303
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
        _startState = 78
        self.enterRecursionRule(localctx, 78, self.RULE_mexpr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 305
            self.uexpr()
            self._ctx.stop = self._input.LT(-1)
            self.state = 312
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,18,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = AnalogParser.MexprContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_mexpr)
                    self.state = 307
                    if not self.precpred(self._ctx, 1):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 1)")
                    self.state = 308
                    _la = self._input.LA(1)
                    if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 351895260495872) != 0)):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 309
                    self.uexpr() 
                self.state = 314
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
        self.enterRule(localctx, 80, self.RULE_uexpr)
        self._la = 0 # Token type
        try:
            self.state = 318
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [23, 24, 28, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81]:
                self.enterOuterAlt(localctx, 1)
                self.state = 315
                self.eexpr()
                pass
            elif token in [36, 37]:
                self.enterOuterAlt(localctx, 2)
                self.state = 316
                _la = self._input.LA(1)
                if not(_la==36 or _la==37):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 317
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
            return self.getTypedRuleContext(AnalogParser.AtomContext,0)


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
        self.enterRule(localctx, 82, self.RULE_eexpr)
        try:
            self.state = 325
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,20,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 320
                self.atom()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 321
                self.atom()
                self.state = 322
                self.match(AnalogParser.POWER)
                self.state = 323
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
        self._predicates[30] = self.bool_expr_sempred
        self._predicates[38] = self.aexpr_sempred
        self._predicates[39] = self.mexpr_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception("No predicate with index:" + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def bool_expr_sempred(self, localctx:Bool_exprContext, predIndex:int):
            if predIndex == 0:
                return self.precpred(self._ctx, 7)
         

            if predIndex == 1:
                return self.precpred(self._ctx, 6)
         

    def aexpr_sempred(self, localctx:AexprContext, predIndex:int):
            if predIndex == 2:
                return self.precpred(self._ctx, 1)
         

    def mexpr_sempred(self, localctx:MexprContext, predIndex:int):
            if predIndex == 3:
                return self.precpred(self._ctx, 1)
         




