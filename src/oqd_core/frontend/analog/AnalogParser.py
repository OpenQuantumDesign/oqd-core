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
        4,1,77,293,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,20,
        7,20,2,21,7,21,2,22,7,22,2,23,7,23,2,24,7,24,2,25,7,25,2,26,7,26,
        2,27,7,27,2,28,7,28,2,29,7,29,2,30,7,30,2,31,7,31,1,0,1,0,3,0,67,
        8,0,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,3,1,77,8,1,1,2,1,2,1,2,5,2,82,
        8,2,10,2,12,2,85,9,2,1,3,1,3,1,3,1,3,1,3,3,3,92,8,3,1,4,1,4,1,4,
        1,4,1,4,3,4,99,8,4,1,5,1,5,3,5,103,8,5,1,5,1,5,5,5,107,8,5,10,5,
        12,5,110,9,5,1,5,1,5,1,6,1,6,1,6,1,6,1,7,1,7,1,8,1,8,1,8,1,8,1,8,
        1,9,1,9,1,9,1,9,3,9,129,8,9,1,9,1,9,1,9,1,9,1,10,1,10,1,10,1,10,
        3,10,139,8,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,3,10,149,8,
        10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,3,10,159,8,10,1,11,1,
        11,1,11,1,11,1,11,1,12,1,12,1,12,1,12,1,12,1,13,1,13,1,13,1,13,1,
        13,1,13,1,13,1,14,1,14,1,14,1,15,1,15,1,15,1,16,1,16,1,17,1,17,1,
        18,1,18,1,19,1,19,1,20,1,20,1,20,1,20,1,20,1,20,1,20,1,20,3,20,200,
        8,20,1,20,1,20,1,20,1,20,1,20,1,20,1,20,1,20,5,20,210,8,20,10,20,
        12,20,213,9,20,1,21,1,21,1,22,1,22,1,23,1,23,3,23,221,8,23,1,24,
        1,24,1,24,1,24,1,24,1,24,1,24,3,24,230,8,24,1,25,1,25,1,26,1,26,
        1,26,1,26,1,27,1,27,1,27,1,28,1,28,1,28,1,28,1,28,3,28,246,8,28,
        1,28,1,28,3,28,250,8,28,1,28,5,28,253,8,28,10,28,12,28,256,9,28,
        1,29,1,29,1,29,1,29,1,29,3,29,263,8,29,1,29,1,29,3,29,267,8,29,1,
        29,5,29,270,8,29,10,29,12,29,273,9,29,1,30,1,30,1,30,3,30,278,8,
        30,1,31,1,31,1,31,3,31,283,8,31,1,31,1,31,3,31,287,8,31,1,31,1,31,
        3,31,291,8,31,1,31,0,3,40,56,58,32,0,2,4,6,8,10,12,14,16,18,20,22,
        24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,0,9,
        1,0,13,14,1,0,15,16,1,0,17,18,1,0,70,73,1,0,74,76,1,0,50,69,3,0,
        33,34,43,43,45,45,2,0,30,32,44,44,1,0,33,34,301,0,64,1,0,0,0,2,76,
        1,0,0,0,4,83,1,0,0,0,6,91,1,0,0,0,8,98,1,0,0,0,10,100,1,0,0,0,12,
        113,1,0,0,0,14,117,1,0,0,0,16,119,1,0,0,0,18,124,1,0,0,0,20,158,
        1,0,0,0,22,160,1,0,0,0,24,165,1,0,0,0,26,170,1,0,0,0,28,177,1,0,
        0,0,30,180,1,0,0,0,32,183,1,0,0,0,34,185,1,0,0,0,36,187,1,0,0,0,
        38,189,1,0,0,0,40,199,1,0,0,0,42,214,1,0,0,0,44,216,1,0,0,0,46,220,
        1,0,0,0,48,229,1,0,0,0,50,231,1,0,0,0,52,233,1,0,0,0,54,237,1,0,
        0,0,56,240,1,0,0,0,58,257,1,0,0,0,60,277,1,0,0,0,62,290,1,0,0,0,
        64,66,3,4,2,0,65,67,3,2,1,0,66,65,1,0,0,0,66,67,1,0,0,0,67,68,1,
        0,0,0,68,69,5,0,0,1,69,1,1,0,0,0,70,77,3,12,6,0,71,77,3,26,13,0,
        72,77,3,28,14,0,73,77,3,30,15,0,74,77,3,18,9,0,75,77,3,20,10,0,76,
        70,1,0,0,0,76,71,1,0,0,0,76,72,1,0,0,0,76,73,1,0,0,0,76,74,1,0,0,
        0,76,75,1,0,0,0,77,3,1,0,0,0,78,79,3,2,1,0,79,80,5,2,0,0,80,82,1,
        0,0,0,81,78,1,0,0,0,82,85,1,0,0,0,83,81,1,0,0,0,83,84,1,0,0,0,84,
        5,1,0,0,0,85,83,1,0,0,0,86,92,3,24,12,0,87,92,3,22,11,0,88,92,3,
        46,23,0,89,92,3,48,24,0,90,92,3,14,7,0,91,86,1,0,0,0,91,87,1,0,0,
        0,91,88,1,0,0,0,91,89,1,0,0,0,91,90,1,0,0,0,92,7,1,0,0,0,93,99,3,
        16,8,0,94,99,3,10,5,0,95,99,3,56,28,0,96,99,3,6,3,0,97,99,3,40,20,
        0,98,93,1,0,0,0,98,94,1,0,0,0,98,95,1,0,0,0,98,96,1,0,0,0,98,97,
        1,0,0,0,99,9,1,0,0,0,100,102,5,26,0,0,101,103,3,8,4,0,102,101,1,
        0,0,0,102,103,1,0,0,0,103,108,1,0,0,0,104,105,5,23,0,0,105,107,3,
        8,4,0,106,104,1,0,0,0,107,110,1,0,0,0,108,106,1,0,0,0,108,109,1,
        0,0,0,109,111,1,0,0,0,110,108,1,0,0,0,111,112,5,27,0,0,112,11,1,
        0,0,0,113,114,5,77,0,0,114,115,5,36,0,0,115,116,3,8,4,0,116,13,1,
        0,0,0,117,118,5,77,0,0,118,15,1,0,0,0,119,120,3,14,7,0,120,121,5,
        26,0,0,121,122,5,46,0,0,122,123,5,27,0,0,123,17,1,0,0,0,124,125,
        5,10,0,0,125,126,5,1,0,0,126,128,3,8,4,0,127,129,5,1,0,0,128,127,
        1,0,0,0,128,129,1,0,0,0,129,130,1,0,0,0,130,131,5,21,0,0,131,132,
        5,2,0,0,132,133,3,4,2,0,133,19,1,0,0,0,134,135,5,8,0,0,135,136,5,
        1,0,0,136,138,3,8,4,0,137,139,5,1,0,0,138,137,1,0,0,0,138,139,1,
        0,0,0,139,140,1,0,0,0,140,141,5,21,0,0,141,142,5,2,0,0,142,143,3,
        4,2,0,143,159,1,0,0,0,144,145,5,8,0,0,145,146,5,1,0,0,146,148,3,
        8,4,0,147,149,5,1,0,0,148,147,1,0,0,0,148,149,1,0,0,0,149,150,1,
        0,0,0,150,151,5,21,0,0,151,152,5,2,0,0,152,153,3,4,2,0,153,154,5,
        9,0,0,154,155,5,21,0,0,155,156,5,2,0,0,156,157,3,4,2,0,157,159,1,
        0,0,0,158,134,1,0,0,0,158,144,1,0,0,0,159,21,1,0,0,0,160,161,5,19,
        0,0,161,162,5,24,0,0,162,163,5,46,0,0,163,164,5,25,0,0,164,23,1,
        0,0,0,165,166,5,20,0,0,166,167,5,24,0,0,167,168,5,46,0,0,168,169,
        5,25,0,0,169,25,1,0,0,0,170,171,5,5,0,0,171,172,3,32,16,0,172,173,
        5,11,0,0,173,174,3,8,4,0,174,175,5,12,0,0,175,176,3,8,4,0,176,27,
        1,0,0,0,177,178,5,6,0,0,178,179,3,32,16,0,179,29,1,0,0,0,180,181,
        5,7,0,0,181,182,3,32,16,0,182,31,1,0,0,0,183,184,3,8,4,0,184,33,
        1,0,0,0,185,186,7,0,0,0,186,35,1,0,0,0,187,188,7,1,0,0,188,37,1,
        0,0,0,189,190,7,2,0,0,190,39,1,0,0,0,191,192,6,20,-1,0,192,193,3,
        38,19,0,193,194,3,40,20,2,194,200,1,0,0,0,195,196,5,24,0,0,196,197,
        3,40,20,0,197,198,5,25,0,0,198,200,1,0,0,0,199,191,1,0,0,0,199,195,
        1,0,0,0,200,211,1,0,0,0,201,202,10,4,0,0,202,203,3,36,18,0,203,204,
        3,40,20,5,204,210,1,0,0,0,205,206,10,3,0,0,206,207,3,34,17,0,207,
        208,3,40,20,4,208,210,1,0,0,0,209,201,1,0,0,0,209,205,1,0,0,0,210,
        213,1,0,0,0,211,209,1,0,0,0,211,212,1,0,0,0,212,41,1,0,0,0,213,211,
        1,0,0,0,214,215,7,3,0,0,215,43,1,0,0,0,216,217,7,4,0,0,217,45,1,
        0,0,0,218,221,3,42,21,0,219,221,3,44,22,0,220,218,1,0,0,0,220,219,
        1,0,0,0,221,47,1,0,0,0,222,230,5,46,0,0,223,230,5,47,0,0,224,230,
        5,48,0,0,225,230,5,49,0,0,226,230,5,77,0,0,227,230,3,52,26,0,228,
        230,3,54,27,0,229,222,1,0,0,0,229,223,1,0,0,0,229,224,1,0,0,0,229,
        225,1,0,0,0,229,226,1,0,0,0,229,227,1,0,0,0,229,228,1,0,0,0,230,
        49,1,0,0,0,231,232,7,5,0,0,232,51,1,0,0,0,233,234,5,24,0,0,234,235,
        3,56,28,0,235,236,5,25,0,0,236,53,1,0,0,0,237,238,3,50,25,0,238,
        239,3,52,26,0,239,55,1,0,0,0,240,241,6,28,-1,0,241,242,3,58,29,0,
        242,254,1,0,0,0,243,245,10,1,0,0,244,246,5,1,0,0,245,244,1,0,0,0,
        245,246,1,0,0,0,246,247,1,0,0,0,247,249,7,6,0,0,248,250,5,1,0,0,
        249,248,1,0,0,0,249,250,1,0,0,0,250,251,1,0,0,0,251,253,3,58,29,
        0,252,243,1,0,0,0,253,256,1,0,0,0,254,252,1,0,0,0,254,255,1,0,0,
        0,255,57,1,0,0,0,256,254,1,0,0,0,257,258,6,29,-1,0,258,259,3,60,
        30,0,259,271,1,0,0,0,260,262,10,1,0,0,261,263,5,1,0,0,262,261,1,
        0,0,0,262,263,1,0,0,0,263,264,1,0,0,0,264,266,7,7,0,0,265,267,5,
        1,0,0,266,265,1,0,0,0,266,267,1,0,0,0,267,268,1,0,0,0,268,270,3,
        60,30,0,269,260,1,0,0,0,270,273,1,0,0,0,271,269,1,0,0,0,271,272,
        1,0,0,0,272,59,1,0,0,0,273,271,1,0,0,0,274,278,3,62,31,0,275,276,
        7,8,0,0,276,278,3,62,31,0,277,274,1,0,0,0,277,275,1,0,0,0,278,61,
        1,0,0,0,279,291,3,6,3,0,280,282,3,6,3,0,281,283,5,1,0,0,282,281,
        1,0,0,0,282,283,1,0,0,0,283,284,1,0,0,0,284,286,5,35,0,0,285,287,
        5,1,0,0,286,285,1,0,0,0,286,287,1,0,0,0,287,288,1,0,0,0,288,289,
        3,60,30,0,289,291,1,0,0,0,290,279,1,0,0,0,290,280,1,0,0,0,291,63,
        1,0,0,0,26,66,76,83,91,98,102,108,128,138,148,158,199,209,211,220,
        229,245,249,254,262,266,271,277,282,286,290
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
                     "':'", "';'", "','", "'('", "')'", "'['", "']'", "'{'", 
                     "'}'", "'@'", "'*'", "'/'", "'+'", "'-'", "'^'", "'='", 
                     "'=='", "'!='", "'<'", "'<='", "'>'", "'>='", "'%+'", 
                     "'%*'", "'%-'", "<INVALID>", "<INVALID>", "<INVALID>", 
                     "'1j'", "'abs'", "'sin'", "'cos'", "'tan'", "'exp'", 
                     "'log'", "'sinh'", "'cosh'", "'tanh'", "'atan'", "'acos'", 
                     "'asin'", "'atanh'", "'asinh'", "'acosh'", "'heaviside'", 
                     "'conj'", "'real'", "'imag'", "'atan2'", "'%I'", "'%X'", 
                     "'%Y'", "'%Z'", "'%C'", "'%A'", "'%J'" ]

    symbolicNames = [ "<INVALID>", "WHITESPACE", "EOL", "NEWLINE", "COMMENT", 
                      "EVOLVE", "MEASURE", "INITIALIZE", "IF", "ELSE", "WHILE", 
                      "WITH", "FOR", "AND", "AND2", "OR", "OR2", "NOT", 
                      "NOT2", "QUANTUMREGISTER", "MODEREGISTER", "COLON", 
                      "SEMICOLON", "COMMA", "LBRACKET", "RBRACKET", "SQUARELBRACKET", 
                      "SQUARERBRACKET", "LBRACE", "RBRACE", "AT", "MULT", 
                      "DIV", "PLUS", "MINUS", "POWER", "ASSIGN", "EQ", "NEQ", 
                      "LT", "LTE", "GT", "GTE", "OP_ADD", "OP_MUL", "OP_MINUS", 
                      "INT", "FLOAT", "MATH_VAR", "IMAG", "ABS", "SIN", 
                      "COS", "TAN", "EXP", "LOG", "SINH", "COSH", "TANH", 
                      "ATAN", "ACOS", "ASIN", "ATANH", "ASINH", "ACOSH", 
                      "HEAVISIDE", "CONJ", "REAL", "IMAG_FN", "ATAN2", "PAULI_I", 
                      "PAULI_X", "PAULI_Y", "PAULI_Z", "CREATION", "ANNIHILATION", 
                      "IDENTITY_OP", "ID" ]

    RULE_program = 0
    RULE_statement = 1
    RULE_block = 2
    RULE_atom = 3
    RULE_expr = 4
    RULE_my_list = 5
    RULE_declaration = 6
    RULE_access = 7
    RULE_extract = 8
    RULE_while_stmt = 9
    RULE_ifelse_stmt = 10
    RULE_quantum_register = 11
    RULE_mode_register = 12
    RULE_evolve_stmt = 13
    RULE_measure_stmt = 14
    RULE_init_stmt = 15
    RULE_targets = 16
    RULE_bool_and_op = 17
    RULE_bool_or_op = 18
    RULE_bool_not_op = 19
    RULE_bool_expr = 20
    RULE_pauli_op = 21
    RULE_ladder_op = 22
    RULE_operator_terminal = 23
    RULE_math_terminal = 24
    RULE_math_func_name = 25
    RULE_pexpr = 26
    RULE_fexpr = 27
    RULE_aexpr = 28
    RULE_mexpr = 29
    RULE_uexpr = 30
    RULE_eexpr = 31

    ruleNames =  [ "program", "statement", "block", "atom", "expr", "my_list", 
                   "declaration", "access", "extract", "while_stmt", "ifelse_stmt", 
                   "quantum_register", "mode_register", "evolve_stmt", "measure_stmt", 
                   "init_stmt", "targets", "bool_and_op", "bool_or_op", 
                   "bool_not_op", "bool_expr", "pauli_op", "ladder_op", 
                   "operator_terminal", "math_terminal", "math_func_name", 
                   "pexpr", "fexpr", "aexpr", "mexpr", "uexpr", "eexpr" ]

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
    COLON=21
    SEMICOLON=22
    COMMA=23
    LBRACKET=24
    RBRACKET=25
    SQUARELBRACKET=26
    SQUARERBRACKET=27
    LBRACE=28
    RBRACE=29
    AT=30
    MULT=31
    DIV=32
    PLUS=33
    MINUS=34
    POWER=35
    ASSIGN=36
    EQ=37
    NEQ=38
    LT=39
    LTE=40
    GT=41
    GTE=42
    OP_ADD=43
    OP_MUL=44
    OP_MINUS=45
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
    PAULI_I=70
    PAULI_X=71
    PAULI_Y=72
    PAULI_Z=73
    CREATION=74
    ANNIHILATION=75
    IDENTITY_OP=76
    ID=77

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

        def statement(self):
            return self.getTypedRuleContext(AnalogParser.StatementContext,0)


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
            self.state = 64
            self.block()
            self.state = 66
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if (((_la) & ~0x3f) == 0 and ((1 << _la) & 1504) != 0) or _la==77:
                self.state = 65
                self.statement()


            self.state = 68
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
            self.state = 76
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [77]:
                self.enterOuterAlt(localctx, 1)
                self.state = 70
                self.declaration()
                pass
            elif token in [5]:
                self.enterOuterAlt(localctx, 2)
                self.state = 71
                self.evolve_stmt()
                pass
            elif token in [6]:
                self.enterOuterAlt(localctx, 3)
                self.state = 72
                self.measure_stmt()
                pass
            elif token in [7]:
                self.enterOuterAlt(localctx, 4)
                self.state = 73
                self.init_stmt()
                pass
            elif token in [10]:
                self.enterOuterAlt(localctx, 5)
                self.state = 74
                self.while_stmt()
                pass
            elif token in [8]:
                self.enterOuterAlt(localctx, 6)
                self.state = 75
                self.ifelse_stmt()
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
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 83
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,2,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    self.state = 78
                    self.statement()
                    self.state = 79
                    self.match(AnalogParser.EOL) 
                self.state = 85
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,2,self._ctx)

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
            self.state = 91
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,3,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 86
                self.mode_register()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 87
                self.quantum_register()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 88
                self.operator_terminal()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 89
                self.math_terminal()
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 90
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


        def aexpr(self):
            return self.getTypedRuleContext(AnalogParser.AexprContext,0)


        def atom(self):
            return self.getTypedRuleContext(AnalogParser.AtomContext,0)


        def bool_expr(self):
            return self.getTypedRuleContext(AnalogParser.Bool_exprContext,0)


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
            self.state = 98
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,4,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 93
                self.extract()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 94
                self.my_list()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 95
                self.aexpr(0)
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 96
                self.atom()
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 97
                self.bool_expr(0)
                pass


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
        self.enterRule(localctx, 10, self.RULE_my_list)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 100
            self.match(AnalogParser.SQUARELBRACKET)
            self.state = 102
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if ((((_la - 17)) & ~0x3f) == 0 and ((1 << (_la - 17)) & 2305843008677020303) != 0):
                self.state = 101
                self.expr()


            self.state = 108
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==23:
                self.state = 104
                self.match(AnalogParser.COMMA)
                self.state = 105
                self.expr()
                self.state = 110
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 111
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
        self.enterRule(localctx, 12, self.RULE_declaration)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 113
            self.match(AnalogParser.ID)
            self.state = 114
            self.match(AnalogParser.ASSIGN)
            self.state = 115
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
        self.enterRule(localctx, 14, self.RULE_access)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 117
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
            self.state = 119
            self.access()
            self.state = 120
            self.match(AnalogParser.SQUARELBRACKET)
            self.state = 121
            self.match(AnalogParser.INT)
            self.state = 122
            self.match(AnalogParser.SQUARERBRACKET)
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

        def WHITESPACE(self, i:int=None):
            if i is None:
                return self.getTokens(AnalogParser.WHITESPACE)
            else:
                return self.getToken(AnalogParser.WHITESPACE, i)

        def expr(self):
            return self.getTypedRuleContext(AnalogParser.ExprContext,0)


        def COLON(self):
            return self.getToken(AnalogParser.COLON, 0)

        def EOL(self):
            return self.getToken(AnalogParser.EOL, 0)

        def block(self):
            return self.getTypedRuleContext(AnalogParser.BlockContext,0)


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
        self.enterRule(localctx, 18, self.RULE_while_stmt)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 124
            self.match(AnalogParser.WHILE)
            self.state = 125
            self.match(AnalogParser.WHITESPACE)
            self.state = 126
            self.expr()
            self.state = 128
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==1:
                self.state = 127
                self.match(AnalogParser.WHITESPACE)


            self.state = 130
            self.match(AnalogParser.COLON)
            self.state = 131
            self.match(AnalogParser.EOL)
            self.state = 132
            self.block()
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

        def WHITESPACE(self, i:int=None):
            if i is None:
                return self.getTokens(AnalogParser.WHITESPACE)
            else:
                return self.getToken(AnalogParser.WHITESPACE, i)

        def expr(self):
            return self.getTypedRuleContext(AnalogParser.ExprContext,0)


        def COLON(self, i:int=None):
            if i is None:
                return self.getTokens(AnalogParser.COLON)
            else:
                return self.getToken(AnalogParser.COLON, i)

        def EOL(self, i:int=None):
            if i is None:
                return self.getTokens(AnalogParser.EOL)
            else:
                return self.getToken(AnalogParser.EOL, i)

        def block(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AnalogParser.BlockContext)
            else:
                return self.getTypedRuleContext(AnalogParser.BlockContext,i)


        def ELSE(self):
            return self.getToken(AnalogParser.ELSE, 0)

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
        self.enterRule(localctx, 20, self.RULE_ifelse_stmt)
        self._la = 0 # Token type
        try:
            self.state = 158
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,10,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 134
                self.match(AnalogParser.IF)
                self.state = 135
                self.match(AnalogParser.WHITESPACE)
                self.state = 136
                self.expr()
                self.state = 138
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==1:
                    self.state = 137
                    self.match(AnalogParser.WHITESPACE)


                self.state = 140
                self.match(AnalogParser.COLON)
                self.state = 141
                self.match(AnalogParser.EOL)
                self.state = 142
                self.block()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 144
                self.match(AnalogParser.IF)
                self.state = 145
                self.match(AnalogParser.WHITESPACE)
                self.state = 146
                self.expr()
                self.state = 148
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==1:
                    self.state = 147
                    self.match(AnalogParser.WHITESPACE)


                self.state = 150
                self.match(AnalogParser.COLON)
                self.state = 151
                self.match(AnalogParser.EOL)
                self.state = 152
                self.block()
                self.state = 153
                self.match(AnalogParser.ELSE)
                self.state = 154
                self.match(AnalogParser.COLON)
                self.state = 155
                self.match(AnalogParser.EOL)
                self.state = 156
                self.block()
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
        self.enterRule(localctx, 22, self.RULE_quantum_register)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 160
            self.match(AnalogParser.QUANTUMREGISTER)
            self.state = 161
            self.match(AnalogParser.LBRACKET)
            self.state = 162
            self.match(AnalogParser.INT)
            self.state = 163
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
        self.enterRule(localctx, 24, self.RULE_mode_register)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 165
            self.match(AnalogParser.MODEREGISTER)
            self.state = 166
            self.match(AnalogParser.LBRACKET)
            self.state = 167
            self.match(AnalogParser.INT)
            self.state = 168
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
        self.enterRule(localctx, 26, self.RULE_evolve_stmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 170
            self.match(AnalogParser.EVOLVE)
            self.state = 171
            self.targets()
            self.state = 172
            self.match(AnalogParser.WITH)
            self.state = 173
            self.expr()
            self.state = 174
            self.match(AnalogParser.FOR)
            self.state = 175
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
        self.enterRule(localctx, 28, self.RULE_measure_stmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 177
            self.match(AnalogParser.MEASURE)
            self.state = 178
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
        self.enterRule(localctx, 30, self.RULE_init_stmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 180
            self.match(AnalogParser.INITIALIZE)
            self.state = 181
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
        self.enterRule(localctx, 32, self.RULE_targets)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 183
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
        self.enterRule(localctx, 34, self.RULE_bool_and_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 185
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
        self.enterRule(localctx, 36, self.RULE_bool_or_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 187
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
        self.enterRule(localctx, 38, self.RULE_bool_not_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 189
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
        _startState = 40
        self.enterRecursionRule(localctx, 40, self.RULE_bool_expr, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 199
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [17, 18]:
                self.state = 192
                self.bool_not_op()
                self.state = 193
                self.bool_expr(2)
                pass
            elif token in [24]:
                self.state = 195
                self.match(AnalogParser.LBRACKET)
                self.state = 196
                self.bool_expr(0)
                self.state = 197
                self.match(AnalogParser.RBRACKET)
                pass
            else:
                raise NoViableAltException(self)

            self._ctx.stop = self._input.LT(-1)
            self.state = 211
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,13,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 209
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,12,self._ctx)
                    if la_ == 1:
                        localctx = AnalogParser.Bool_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_bool_expr)
                        self.state = 201
                        if not self.precpred(self._ctx, 4):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 4)")
                        self.state = 202
                        self.bool_or_op()
                        self.state = 203
                        self.bool_expr(5)
                        pass

                    elif la_ == 2:
                        localctx = AnalogParser.Bool_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_bool_expr)
                        self.state = 205
                        if not self.precpred(self._ctx, 3):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 3)")
                        self.state = 206
                        self.bool_and_op()
                        self.state = 207
                        self.bool_expr(4)
                        pass

             
                self.state = 213
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,13,self._ctx)

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
        self.enterRule(localctx, 42, self.RULE_pauli_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 214
            _la = self._input.LA(1)
            if not(((((_la - 70)) & ~0x3f) == 0 and ((1 << (_la - 70)) & 15) != 0)):
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
        self.enterRule(localctx, 44, self.RULE_ladder_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 216
            _la = self._input.LA(1)
            if not(((((_la - 74)) & ~0x3f) == 0 and ((1 << (_la - 74)) & 7) != 0)):
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
        self.enterRule(localctx, 46, self.RULE_operator_terminal)
        try:
            self.state = 220
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [70, 71, 72, 73]:
                self.enterOuterAlt(localctx, 1)
                self.state = 218
                self.pauli_op()
                pass
            elif token in [74, 75, 76]:
                self.enterOuterAlt(localctx, 2)
                self.state = 219
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
        self.enterRule(localctx, 48, self.RULE_math_terminal)
        try:
            self.state = 229
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [46]:
                self.enterOuterAlt(localctx, 1)
                self.state = 222
                self.match(AnalogParser.INT)
                pass
            elif token in [47]:
                self.enterOuterAlt(localctx, 2)
                self.state = 223
                self.match(AnalogParser.FLOAT)
                pass
            elif token in [48]:
                self.enterOuterAlt(localctx, 3)
                self.state = 224
                self.match(AnalogParser.MATH_VAR)
                pass
            elif token in [49]:
                self.enterOuterAlt(localctx, 4)
                self.state = 225
                self.match(AnalogParser.IMAG)
                pass
            elif token in [77]:
                self.enterOuterAlt(localctx, 5)
                self.state = 226
                self.match(AnalogParser.ID)
                pass
            elif token in [24]:
                self.enterOuterAlt(localctx, 6)
                self.state = 227
                self.pexpr()
                pass
            elif token in [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69]:
                self.enterOuterAlt(localctx, 7)
                self.state = 228
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
        self.enterRule(localctx, 50, self.RULE_math_func_name)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 231
            _la = self._input.LA(1)
            if not(((((_la - 50)) & ~0x3f) == 0 and ((1 << (_la - 50)) & 1048575) != 0)):
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
        self.enterRule(localctx, 52, self.RULE_pexpr)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 233
            self.match(AnalogParser.LBRACKET)
            self.state = 234
            self.aexpr(0)
            self.state = 235
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
        self.enterRule(localctx, 54, self.RULE_fexpr)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 237
            self.math_func_name()
            self.state = 238
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

        def WHITESPACE(self, i:int=None):
            if i is None:
                return self.getTokens(AnalogParser.WHITESPACE)
            else:
                return self.getToken(AnalogParser.WHITESPACE, i)

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
            self.state = 241
            self.mexpr(0)
            self._ctx.stop = self._input.LT(-1)
            self.state = 254
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,18,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = AnalogParser.AexprContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_aexpr)
                    self.state = 243
                    if not self.precpred(self._ctx, 1):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 1)")
                    self.state = 245
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la==1:
                        self.state = 244
                        self.match(AnalogParser.WHITESPACE)


                    self.state = 247
                    _la = self._input.LA(1)
                    if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 44006234914816) != 0)):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 249
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la==1:
                        self.state = 248
                        self.match(AnalogParser.WHITESPACE)


                    self.state = 251
                    self.mexpr(0) 
                self.state = 256
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

        def WHITESPACE(self, i:int=None):
            if i is None:
                return self.getTokens(AnalogParser.WHITESPACE)
            else:
                return self.getToken(AnalogParser.WHITESPACE, i)

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
        _startState = 58
        self.enterRecursionRule(localctx, 58, self.RULE_mexpr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 258
            self.uexpr()
            self._ctx.stop = self._input.LT(-1)
            self.state = 271
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,21,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    localctx = AnalogParser.MexprContext(self, _parentctx, _parentState)
                    self.pushNewRecursionContext(localctx, _startState, self.RULE_mexpr)
                    self.state = 260
                    if not self.precpred(self._ctx, 1):
                        from antlr4.error.Errors import FailedPredicateException
                        raise FailedPredicateException(self, "self.precpred(self._ctx, 1)")
                    self.state = 262
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la==1:
                        self.state = 261
                        self.match(AnalogParser.WHITESPACE)


                    self.state = 264
                    _la = self._input.LA(1)
                    if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 17599702237184) != 0)):
                        self._errHandler.recoverInline(self)
                    else:
                        self._errHandler.reportMatch(self)
                        self.consume()
                    self.state = 266
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)
                    if _la==1:
                        self.state = 265
                        self.match(AnalogParser.WHITESPACE)


                    self.state = 268
                    self.uexpr() 
                self.state = 273
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,21,self._ctx)

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
        self.enterRule(localctx, 60, self.RULE_uexpr)
        self._la = 0 # Token type
        try:
            self.state = 277
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [19, 20, 24, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77]:
                self.enterOuterAlt(localctx, 1)
                self.state = 274
                self.eexpr()
                pass
            elif token in [33, 34]:
                self.enterOuterAlt(localctx, 2)
                self.state = 275
                _la = self._input.LA(1)
                if not(_la==33 or _la==34):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 276
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


        def WHITESPACE(self, i:int=None):
            if i is None:
                return self.getTokens(AnalogParser.WHITESPACE)
            else:
                return self.getToken(AnalogParser.WHITESPACE, i)

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
        self.enterRule(localctx, 62, self.RULE_eexpr)
        self._la = 0 # Token type
        try:
            self.state = 290
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,25,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 279
                self.atom()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 280
                self.atom()
                self.state = 282
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==1:
                    self.state = 281
                    self.match(AnalogParser.WHITESPACE)


                self.state = 284
                self.match(AnalogParser.POWER)
                self.state = 286
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==1:
                    self.state = 285
                    self.match(AnalogParser.WHITESPACE)


                self.state = 288
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
        self._predicates[20] = self.bool_expr_sempred
        self._predicates[28] = self.aexpr_sempred
        self._predicates[29] = self.mexpr_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception("No predicate with index:" + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def bool_expr_sempred(self, localctx:Bool_exprContext, predIndex:int):
            if predIndex == 0:
                return self.precpred(self._ctx, 4)
         

            if predIndex == 1:
                return self.precpred(self._ctx, 3)
         

    def aexpr_sempred(self, localctx:AexprContext, predIndex:int):
            if predIndex == 2:
                return self.precpred(self._ctx, 1)
         

    def mexpr_sempred(self, localctx:MexprContext, predIndex:int):
            if predIndex == 3:
                return self.precpred(self._ctx, 1)
         




