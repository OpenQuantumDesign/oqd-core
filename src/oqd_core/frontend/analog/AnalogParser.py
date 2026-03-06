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
        4,1,68,244,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,
        6,2,7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,
        2,14,7,14,2,15,7,15,2,16,7,16,2,17,7,17,2,18,7,18,2,19,7,19,2,20,
        7,20,2,21,7,21,2,22,7,22,2,23,7,23,2,24,7,24,2,25,7,25,1,0,1,0,1,
        0,5,0,56,8,0,10,0,12,0,59,9,0,1,0,3,0,62,8,0,1,0,1,0,1,1,1,1,1,1,
        1,1,3,1,70,8,1,1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,3,3,3,80,8,3,1,4,1,
        4,1,4,1,4,3,4,86,8,4,1,5,1,5,1,5,1,6,1,6,1,6,1,6,1,6,1,7,1,7,1,7,
        1,7,5,7,100,8,7,10,7,12,7,103,9,7,1,7,1,7,1,8,1,8,1,9,1,9,1,9,1,
        9,1,9,1,9,1,9,1,10,1,10,1,11,1,11,1,12,1,12,1,13,1,13,1,14,1,14,
        1,15,1,15,1,16,1,16,1,16,1,16,1,16,1,16,1,16,1,16,1,16,3,16,137,
        8,16,1,16,1,16,1,16,1,16,1,16,1,16,1,16,1,16,5,16,147,8,16,10,16,
        12,16,150,9,16,1,17,1,17,1,18,1,18,1,19,1,19,1,20,1,20,1,20,1,20,
        1,20,1,20,1,20,1,20,1,20,1,20,1,20,3,20,169,8,20,1,20,1,20,1,20,
        1,20,1,20,1,20,1,20,1,20,1,20,1,20,1,20,1,20,1,20,1,20,1,20,5,20,
        186,8,20,10,20,12,20,189,9,20,1,21,1,21,3,21,193,8,21,1,22,1,22,
        1,22,1,22,1,22,1,22,1,22,1,22,1,22,3,22,204,8,22,1,22,1,22,1,22,
        1,22,1,22,1,22,1,22,1,22,1,22,1,22,1,22,1,22,1,22,1,22,1,22,5,22,
        221,8,22,10,22,12,22,224,9,22,1,23,1,23,1,24,1,24,1,25,1,25,1,25,
        1,25,1,25,1,25,1,25,1,25,1,25,1,25,1,25,1,25,3,25,242,8,25,1,25,
        0,3,32,40,44,26,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,
        36,38,40,42,44,46,48,50,0,9,2,0,46,46,68,68,1,0,13,14,1,0,15,16,
        1,0,17,18,1,0,24,27,1,0,20,23,1,0,40,41,2,0,44,47,68,68,1,0,48,66,
        251,0,57,1,0,0,0,2,69,1,0,0,0,4,71,1,0,0,0,6,79,1,0,0,0,8,85,1,0,
        0,0,10,87,1,0,0,0,12,90,1,0,0,0,14,95,1,0,0,0,16,106,1,0,0,0,18,
        108,1,0,0,0,20,115,1,0,0,0,22,117,1,0,0,0,24,119,1,0,0,0,26,121,
        1,0,0,0,28,123,1,0,0,0,30,125,1,0,0,0,32,136,1,0,0,0,34,151,1,0,
        0,0,36,153,1,0,0,0,38,155,1,0,0,0,40,168,1,0,0,0,42,192,1,0,0,0,
        44,203,1,0,0,0,46,225,1,0,0,0,48,227,1,0,0,0,50,241,1,0,0,0,52,53,
        3,2,1,0,53,54,5,2,0,0,54,56,1,0,0,0,55,52,1,0,0,0,56,59,1,0,0,0,
        57,55,1,0,0,0,57,58,1,0,0,0,58,61,1,0,0,0,59,57,1,0,0,0,60,62,3,
        2,1,0,61,60,1,0,0,0,61,62,1,0,0,0,62,63,1,0,0,0,63,64,5,0,0,1,64,
        1,1,0,0,0,65,70,3,4,2,0,66,70,3,18,9,0,67,70,3,22,11,0,68,70,3,24,
        12,0,69,65,1,0,0,0,69,66,1,0,0,0,69,67,1,0,0,0,69,68,1,0,0,0,70,
        3,1,0,0,0,71,72,7,0,0,0,72,73,5,43,0,0,73,74,3,6,3,0,74,5,1,0,0,
        0,75,80,3,32,16,0,76,80,3,8,4,0,77,80,3,40,20,0,78,80,3,44,22,0,
        79,75,1,0,0,0,79,76,1,0,0,0,79,77,1,0,0,0,79,78,1,0,0,0,80,7,1,0,
        0,0,81,86,3,10,5,0,82,86,3,12,6,0,83,86,3,14,7,0,84,86,3,16,8,0,
        85,81,1,0,0,0,85,82,1,0,0,0,85,83,1,0,0,0,85,84,1,0,0,0,86,9,1,0,
        0,0,87,88,5,19,0,0,88,89,5,44,0,0,89,11,1,0,0,0,90,91,5,68,0,0,91,
        92,5,33,0,0,92,93,5,44,0,0,93,94,5,34,0,0,94,13,1,0,0,0,95,96,5,
        33,0,0,96,101,3,8,4,0,97,98,5,30,0,0,98,100,3,8,4,0,99,97,1,0,0,
        0,100,103,1,0,0,0,101,99,1,0,0,0,101,102,1,0,0,0,102,104,1,0,0,0,
        103,101,1,0,0,0,104,105,5,34,0,0,105,15,1,0,0,0,106,107,5,68,0,0,
        107,17,1,0,0,0,108,109,5,5,0,0,109,110,3,40,20,0,110,111,5,11,0,
        0,111,112,3,44,22,0,112,113,5,12,0,0,113,114,3,20,10,0,114,19,1,
        0,0,0,115,116,3,8,4,0,116,21,1,0,0,0,117,118,5,6,0,0,118,23,1,0,
        0,0,119,120,5,7,0,0,120,25,1,0,0,0,121,122,7,1,0,0,122,27,1,0,0,
        0,123,124,7,2,0,0,124,29,1,0,0,0,125,126,7,3,0,0,126,31,1,0,0,0,
        127,128,6,16,-1,0,128,129,3,30,15,0,129,130,3,32,16,3,130,137,1,
        0,0,0,131,137,3,34,17,0,132,133,5,31,0,0,133,134,3,32,16,0,134,135,
        5,32,0,0,135,137,1,0,0,0,136,127,1,0,0,0,136,131,1,0,0,0,136,132,
        1,0,0,0,137,148,1,0,0,0,138,139,10,5,0,0,139,140,3,28,14,0,140,141,
        3,32,16,6,141,147,1,0,0,0,142,143,10,4,0,0,143,144,3,26,13,0,144,
        145,3,32,16,5,145,147,1,0,0,0,146,138,1,0,0,0,146,142,1,0,0,0,147,
        150,1,0,0,0,148,146,1,0,0,0,148,149,1,0,0,0,149,33,1,0,0,0,150,148,
        1,0,0,0,151,152,5,68,0,0,152,35,1,0,0,0,153,154,7,4,0,0,154,37,1,
        0,0,0,155,156,7,5,0,0,156,39,1,0,0,0,157,158,6,20,-1,0,158,159,3,
        44,22,0,159,160,5,38,0,0,160,161,3,40,20,5,161,169,1,0,0,0,162,169,
        3,42,21,0,163,169,3,16,8,0,164,165,5,31,0,0,165,166,3,40,20,0,166,
        167,5,32,0,0,167,169,1,0,0,0,168,157,1,0,0,0,168,162,1,0,0,0,168,
        163,1,0,0,0,168,164,1,0,0,0,169,187,1,0,0,0,170,171,10,9,0,0,171,
        172,5,40,0,0,172,186,3,40,20,10,173,174,10,8,0,0,174,175,5,41,0,
        0,175,186,3,40,20,9,176,177,10,7,0,0,177,178,5,37,0,0,178,186,3,
        40,20,8,179,180,10,6,0,0,180,181,5,38,0,0,181,186,3,40,20,7,182,
        183,10,4,0,0,183,184,5,38,0,0,184,186,3,44,22,0,185,170,1,0,0,0,
        185,173,1,0,0,0,185,176,1,0,0,0,185,179,1,0,0,0,185,182,1,0,0,0,
        186,189,1,0,0,0,187,185,1,0,0,0,187,188,1,0,0,0,188,41,1,0,0,0,189,
        187,1,0,0,0,190,193,3,36,18,0,191,193,3,38,19,0,192,190,1,0,0,0,
        192,191,1,0,0,0,193,43,1,0,0,0,194,195,6,22,-1,0,195,196,7,6,0,0,
        196,204,3,44,22,4,197,204,3,46,23,0,198,204,3,50,25,0,199,200,5,
        31,0,0,200,201,3,44,22,0,201,202,5,32,0,0,202,204,1,0,0,0,203,194,
        1,0,0,0,203,197,1,0,0,0,203,198,1,0,0,0,203,199,1,0,0,0,204,222,
        1,0,0,0,205,206,10,9,0,0,206,207,5,40,0,0,207,221,3,44,22,10,208,
        209,10,8,0,0,209,210,5,41,0,0,210,221,3,44,22,9,211,212,10,7,0,0,
        212,213,5,38,0,0,213,221,3,44,22,8,214,215,10,6,0,0,215,216,5,39,
        0,0,216,221,3,44,22,7,217,218,10,5,0,0,218,219,5,42,0,0,219,221,
        3,44,22,6,220,205,1,0,0,0,220,208,1,0,0,0,220,211,1,0,0,0,220,214,
        1,0,0,0,220,217,1,0,0,0,221,224,1,0,0,0,222,220,1,0,0,0,222,223,
        1,0,0,0,223,45,1,0,0,0,224,222,1,0,0,0,225,226,7,7,0,0,226,47,1,
        0,0,0,227,228,7,8,0,0,228,49,1,0,0,0,229,230,5,67,0,0,230,231,5,
        31,0,0,231,232,3,44,22,0,232,233,5,30,0,0,233,234,3,44,22,0,234,
        235,5,32,0,0,235,242,1,0,0,0,236,237,3,48,24,0,237,238,5,31,0,0,
        238,239,3,44,22,0,239,240,5,32,0,0,240,242,1,0,0,0,241,229,1,0,0,
        0,241,236,1,0,0,0,242,51,1,0,0,0,17,57,61,69,79,85,101,136,146,148,
        168,185,187,192,203,220,222,241
    ]

class AnalogParser ( Parser ):

    grammarFileName = "AnalogParser.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                     "<INVALID>", "'evolve'", "'measure'", "'initialize'", 
                     "'if'", "'else'", "'while'", "'for'", "'on'", "'and'", 
                     "'&&'", "'or'", "'||'", "'not'", "'!'", "'register'", 
                     "'creation'", "'a_dag'", "'annihilation'", "'identity'", 
                     "'%I'", "'%X'", "'%Y'", "'%Z'", "':'", "';'", "','", 
                     "'('", "')'", "'['", "']'", "'{'", "'}'", "'@'", "'*'", 
                     "'/'", "'+'", "'-'", "'^'", "'='", "<INVALID>", "<INVALID>", 
                     "<INVALID>", "<INVALID>", "'abs'", "'sin'", "'cos'", 
                     "'tan'", "'exp'", "'log'", "'sinh'", "'cosh'", "'tanh'", 
                     "'atan'", "'acos'", "'asin'", "'atanh'", "'asinh'", 
                     "'acosh'", "'heaviside'", "'conj'", "'real'", "'imag'", 
                     "'atan2'" ]

    symbolicNames = [ "<INVALID>", "WHITESPACE", "EOL", "NEWLINE", "COMMENT", 
                      "EVOLVE", "MEASURE", "INITIALIZE", "IF", "ELSE", "WHILE", 
                      "FOR", "ON", "AND", "AND2", "OR", "OR2", "NOT", "NOT2", 
                      "REGISTER", "CREATION", "A_DAG", "ANNIHILATION", "IDENTITY_OP", 
                      "PAULI_I", "PAULI_X", "PAULI_Y", "PAULI_Z", "COLON", 
                      "SEMICOLON", "COMMA", "LBRACKET", "RBRACKET", "SQUARELBRACKET", 
                      "SQUARERBRACKET", "LBRACE", "RBRACE", "AT", "MULT", 
                      "DIV", "PLUS", "MINUS", "POWER", "EQ", "INT", "FLOAT", 
                      "MATH_VAR", "IMAG", "ABS", "SIN", "COS", "TAN", "EXP", 
                      "LOG", "SINH", "COSH", "TANH", "ATAN", "ACOS", "ASIN", 
                      "ATANH", "ASINH", "ACOSH", "HEAVISIDE", "CONJ", "REAL", 
                      "IMAG_FN", "ATAN2", "ID" ]

    RULE_program = 0
    RULE_statement = 1
    RULE_declaration = 2
    RULE_decl_value = 3
    RULE_atomic_type = 4
    RULE_quantum_register = 5
    RULE_quantum_bit = 6
    RULE_my_list = 7
    RULE_access = 8
    RULE_evolve_stmt = 9
    RULE_targets = 10
    RULE_measure_stmt = 11
    RULE_init_stmt = 12
    RULE_bool_and_op = 13
    RULE_bool_or_op = 14
    RULE_bool_not_op = 15
    RULE_bool_expr = 16
    RULE_bool_ref = 17
    RULE_pauli_op = 18
    RULE_ladder_op = 19
    RULE_operator_expr = 20
    RULE_operator_terminal = 21
    RULE_math_expr = 22
    RULE_math_terminal = 23
    RULE_math_func_name = 24
    RULE_math_func = 25

    ruleNames =  [ "program", "statement", "declaration", "decl_value", 
                   "atomic_type", "quantum_register", "quantum_bit", "my_list", 
                   "access", "evolve_stmt", "targets", "measure_stmt", "init_stmt", 
                   "bool_and_op", "bool_or_op", "bool_not_op", "bool_expr", 
                   "bool_ref", "pauli_op", "ladder_op", "operator_expr", 
                   "operator_terminal", "math_expr", "math_terminal", "math_func_name", 
                   "math_func" ]

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
    FOR=11
    ON=12
    AND=13
    AND2=14
    OR=15
    OR2=16
    NOT=17
    NOT2=18
    REGISTER=19
    CREATION=20
    A_DAG=21
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
            self.state = 57
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,0,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    self.state = 52
                    self.statement()
                    self.state = 53
                    self.match(AnalogParser.EOL) 
                self.state = 59
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,0,self._ctx)

            self.state = 61
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if ((((_la - 5)) & ~0x3f) == 0 and ((1 << (_la - 5)) & -9223369837831520249) != 0):
                self.state = 60
                self.statement()


            self.state = 63
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
            self.state = 69
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [46, 68]:
                self.enterOuterAlt(localctx, 1)
                self.state = 65
                self.declaration()
                pass
            elif token in [5]:
                self.enterOuterAlt(localctx, 2)
                self.state = 66
                self.evolve_stmt()
                pass
            elif token in [6]:
                self.enterOuterAlt(localctx, 3)
                self.state = 67
                self.measure_stmt()
                pass
            elif token in [7]:
                self.enterOuterAlt(localctx, 4)
                self.state = 68
                self.init_stmt()
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

        def EQ(self):
            return self.getToken(AnalogParser.EQ, 0)

        def decl_value(self):
            return self.getTypedRuleContext(AnalogParser.Decl_valueContext,0)


        def ID(self):
            return self.getToken(AnalogParser.ID, 0)

        def MATH_VAR(self):
            return self.getToken(AnalogParser.MATH_VAR, 0)

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
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 71
            _la = self._input.LA(1)
            if not(_la==46 or _la==68):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 72
            self.match(AnalogParser.EQ)
            self.state = 73
            self.decl_value()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Decl_valueContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def bool_expr(self):
            return self.getTypedRuleContext(AnalogParser.Bool_exprContext,0)


        def atomic_type(self):
            return self.getTypedRuleContext(AnalogParser.Atomic_typeContext,0)


        def operator_expr(self):
            return self.getTypedRuleContext(AnalogParser.Operator_exprContext,0)


        def math_expr(self):
            return self.getTypedRuleContext(AnalogParser.Math_exprContext,0)


        def getRuleIndex(self):
            return AnalogParser.RULE_decl_value

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterDecl_value" ):
                listener.enterDecl_value(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitDecl_value" ):
                listener.exitDecl_value(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitDecl_value" ):
                return visitor.visitDecl_value(self)
            else:
                return visitor.visitChildren(self)




    def decl_value(self):

        localctx = AnalogParser.Decl_valueContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_decl_value)
        try:
            self.state = 79
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,3,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 75
                self.bool_expr(0)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 76
                self.atomic_type()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 77
                self.operator_expr(0)
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 78
                self.math_expr(0)
                pass


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

        def quantum_register(self):
            return self.getTypedRuleContext(AnalogParser.Quantum_registerContext,0)


        def quantum_bit(self):
            return self.getTypedRuleContext(AnalogParser.Quantum_bitContext,0)


        def my_list(self):
            return self.getTypedRuleContext(AnalogParser.My_listContext,0)


        def access(self):
            return self.getTypedRuleContext(AnalogParser.AccessContext,0)


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
        self.enterRule(localctx, 8, self.RULE_atomic_type)
        try:
            self.state = 85
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,4,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 81
                self.quantum_register()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 82
                self.quantum_bit()
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 83
                self.my_list()
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 84
                self.access()
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

        def REGISTER(self):
            return self.getToken(AnalogParser.REGISTER, 0)

        def INT(self):
            return self.getToken(AnalogParser.INT, 0)

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
        self.enterRule(localctx, 10, self.RULE_quantum_register)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 87
            self.match(AnalogParser.REGISTER)
            self.state = 88
            self.match(AnalogParser.INT)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Quantum_bitContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(AnalogParser.ID, 0)

        def SQUARELBRACKET(self):
            return self.getToken(AnalogParser.SQUARELBRACKET, 0)

        def INT(self):
            return self.getToken(AnalogParser.INT, 0)

        def SQUARERBRACKET(self):
            return self.getToken(AnalogParser.SQUARERBRACKET, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_quantum_bit

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterQuantum_bit" ):
                listener.enterQuantum_bit(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitQuantum_bit" ):
                listener.exitQuantum_bit(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitQuantum_bit" ):
                return visitor.visitQuantum_bit(self)
            else:
                return visitor.visitChildren(self)




    def quantum_bit(self):

        localctx = AnalogParser.Quantum_bitContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_quantum_bit)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 90
            self.match(AnalogParser.ID)
            self.state = 91
            self.match(AnalogParser.SQUARELBRACKET)
            self.state = 92
            self.match(AnalogParser.INT)
            self.state = 93
            self.match(AnalogParser.SQUARERBRACKET)
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

        def atomic_type(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(AnalogParser.Atomic_typeContext)
            else:
                return self.getTypedRuleContext(AnalogParser.Atomic_typeContext,i)


        def SQUARERBRACKET(self):
            return self.getToken(AnalogParser.SQUARERBRACKET, 0)

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
        self.enterRule(localctx, 14, self.RULE_my_list)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 95
            self.match(AnalogParser.SQUARELBRACKET)
            self.state = 96
            self.atomic_type()
            self.state = 101
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==30:
                self.state = 97
                self.match(AnalogParser.COMMA)
                self.state = 98
                self.atomic_type()
                self.state = 103
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 104
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
        self.enterRule(localctx, 16, self.RULE_access)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 106
            self.match(AnalogParser.ID)
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

        def operator_expr(self):
            return self.getTypedRuleContext(AnalogParser.Operator_exprContext,0)


        def FOR(self):
            return self.getToken(AnalogParser.FOR, 0)

        def math_expr(self):
            return self.getTypedRuleContext(AnalogParser.Math_exprContext,0)


        def ON(self):
            return self.getToken(AnalogParser.ON, 0)

        def targets(self):
            return self.getTypedRuleContext(AnalogParser.TargetsContext,0)


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
            self.state = 108
            self.match(AnalogParser.EVOLVE)
            self.state = 109
            self.operator_expr(0)
            self.state = 110
            self.match(AnalogParser.FOR)
            self.state = 111
            self.math_expr(0)
            self.state = 112
            self.match(AnalogParser.ON)
            self.state = 113
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
        self.enterRule(localctx, 20, self.RULE_targets)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 115
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
        self.enterRule(localctx, 22, self.RULE_measure_stmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 117
            self.match(AnalogParser.MEASURE)
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
        self.enterRule(localctx, 24, self.RULE_init_stmt)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 119
            self.match(AnalogParser.INITIALIZE)
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
        self.enterRule(localctx, 26, self.RULE_bool_and_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 121
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
        self.enterRule(localctx, 28, self.RULE_bool_or_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 123
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
        self.enterRule(localctx, 30, self.RULE_bool_not_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 125
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


        def bool_ref(self):
            return self.getTypedRuleContext(AnalogParser.Bool_refContext,0)


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
        _startState = 32
        self.enterRecursionRule(localctx, 32, self.RULE_bool_expr, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 136
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [17, 18]:
                self.state = 128
                self.bool_not_op()
                self.state = 129
                self.bool_expr(3)
                pass
            elif token in [68]:
                self.state = 131
                self.bool_ref()
                pass
            elif token in [31]:
                self.state = 132
                self.match(AnalogParser.LBRACKET)
                self.state = 133
                self.bool_expr(0)
                self.state = 134
                self.match(AnalogParser.RBRACKET)
                pass
            else:
                raise NoViableAltException(self)

            self._ctx.stop = self._input.LT(-1)
            self.state = 148
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,8,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 146
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,7,self._ctx)
                    if la_ == 1:
                        localctx = AnalogParser.Bool_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_bool_expr)
                        self.state = 138
                        if not self.precpred(self._ctx, 5):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 5)")
                        self.state = 139
                        self.bool_or_op()
                        self.state = 140
                        self.bool_expr(6)
                        pass

                    elif la_ == 2:
                        localctx = AnalogParser.Bool_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_bool_expr)
                        self.state = 142
                        if not self.precpred(self._ctx, 4):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 4)")
                        self.state = 143
                        self.bool_and_op()
                        self.state = 144
                        self.bool_expr(5)
                        pass

             
                self.state = 150
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,8,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class Bool_refContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def ID(self):
            return self.getToken(AnalogParser.ID, 0)

        def getRuleIndex(self):
            return AnalogParser.RULE_bool_ref

        def enterRule(self, listener:ParseTreeListener):
            if hasattr( listener, "enterBool_ref" ):
                listener.enterBool_ref(self)

        def exitRule(self, listener:ParseTreeListener):
            if hasattr( listener, "exitBool_ref" ):
                listener.exitBool_ref(self)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitBool_ref" ):
                return visitor.visitBool_ref(self)
            else:
                return visitor.visitChildren(self)




    def bool_ref(self):

        localctx = AnalogParser.Bool_refContext(self, self._ctx, self.state)
        self.enterRule(localctx, 34, self.RULE_bool_ref)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 151
            self.match(AnalogParser.ID)
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
        self.enterRule(localctx, 36, self.RULE_pauli_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 153
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

        def A_DAG(self):
            return self.getToken(AnalogParser.A_DAG, 0)

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
        self.enterRule(localctx, 38, self.RULE_ladder_op)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 155
            _la = self._input.LA(1)
            if not((((_la) & ~0x3f) == 0 and ((1 << _la) & 15728640) != 0)):
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
        _startState = 40
        self.enterRecursionRule(localctx, 40, self.RULE_operator_expr, _p)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 168
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,9,self._ctx)
            if la_ == 1:
                self.state = 158
                self.math_expr(0)
                self.state = 159
                self.match(AnalogParser.MULT)
                self.state = 160
                self.operator_expr(5)
                pass

            elif la_ == 2:
                self.state = 162
                self.operator_terminal()
                pass

            elif la_ == 3:
                self.state = 163
                self.access()
                pass

            elif la_ == 4:
                self.state = 164
                self.match(AnalogParser.LBRACKET)
                self.state = 165
                self.operator_expr(0)
                self.state = 166
                self.match(AnalogParser.RBRACKET)
                pass


            self._ctx.stop = self._input.LT(-1)
            self.state = 187
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,11,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 185
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,10,self._ctx)
                    if la_ == 1:
                        localctx = AnalogParser.Operator_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_operator_expr)
                        self.state = 170
                        if not self.precpred(self._ctx, 9):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 9)")
                        self.state = 171
                        self.match(AnalogParser.PLUS)
                        self.state = 172
                        self.operator_expr(10)
                        pass

                    elif la_ == 2:
                        localctx = AnalogParser.Operator_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_operator_expr)
                        self.state = 173
                        if not self.precpred(self._ctx, 8):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 8)")
                        self.state = 174
                        self.match(AnalogParser.MINUS)
                        self.state = 175
                        self.operator_expr(9)
                        pass

                    elif la_ == 3:
                        localctx = AnalogParser.Operator_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_operator_expr)
                        self.state = 176
                        if not self.precpred(self._ctx, 7):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 7)")
                        self.state = 177
                        self.match(AnalogParser.AT)
                        self.state = 178
                        self.operator_expr(8)
                        pass

                    elif la_ == 4:
                        localctx = AnalogParser.Operator_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_operator_expr)
                        self.state = 179
                        if not self.precpred(self._ctx, 6):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 6)")
                        self.state = 180
                        self.match(AnalogParser.MULT)
                        self.state = 181
                        self.operator_expr(7)
                        pass

                    elif la_ == 5:
                        localctx = AnalogParser.Operator_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_operator_expr)
                        self.state = 182
                        if not self.precpred(self._ctx, 4):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 4)")
                        self.state = 183
                        self.match(AnalogParser.MULT)
                        self.state = 184
                        self.math_expr(0)
                        pass

             
                self.state = 189
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,11,self._ctx)

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
        self.enterRule(localctx, 42, self.RULE_operator_terminal)
        try:
            self.state = 192
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [24, 25, 26, 27]:
                self.enterOuterAlt(localctx, 1)
                self.state = 190
                self.pauli_op()
                pass
            elif token in [20, 21, 22, 23]:
                self.enterOuterAlt(localctx, 2)
                self.state = 191
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
        _startState = 44
        self.enterRecursionRule(localctx, 44, self.RULE_math_expr, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 203
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [40, 41]:
                self.state = 195
                _la = self._input.LA(1)
                if not(_la==40 or _la==41):
                    self._errHandler.recoverInline(self)
                else:
                    self._errHandler.reportMatch(self)
                    self.consume()
                self.state = 196
                self.math_expr(4)
                pass
            elif token in [44, 45, 46, 47, 68]:
                self.state = 197
                self.math_terminal()
                pass
            elif token in [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]:
                self.state = 198
                self.math_func()
                pass
            elif token in [31]:
                self.state = 199
                self.match(AnalogParser.LBRACKET)
                self.state = 200
                self.math_expr(0)
                self.state = 201
                self.match(AnalogParser.RBRACKET)
                pass
            else:
                raise NoViableAltException(self)

            self._ctx.stop = self._input.LT(-1)
            self.state = 222
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,15,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 220
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,14,self._ctx)
                    if la_ == 1:
                        localctx = AnalogParser.Math_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_math_expr)
                        self.state = 205
                        if not self.precpred(self._ctx, 9):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 9)")
                        self.state = 206
                        self.match(AnalogParser.PLUS)
                        self.state = 207
                        self.math_expr(10)
                        pass

                    elif la_ == 2:
                        localctx = AnalogParser.Math_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_math_expr)
                        self.state = 208
                        if not self.precpred(self._ctx, 8):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 8)")
                        self.state = 209
                        self.match(AnalogParser.MINUS)
                        self.state = 210
                        self.math_expr(9)
                        pass

                    elif la_ == 3:
                        localctx = AnalogParser.Math_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_math_expr)
                        self.state = 211
                        if not self.precpred(self._ctx, 7):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 7)")
                        self.state = 212
                        self.match(AnalogParser.MULT)
                        self.state = 213
                        self.math_expr(8)
                        pass

                    elif la_ == 4:
                        localctx = AnalogParser.Math_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_math_expr)
                        self.state = 214
                        if not self.precpred(self._ctx, 6):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 6)")
                        self.state = 215
                        self.match(AnalogParser.DIV)
                        self.state = 216
                        self.math_expr(7)
                        pass

                    elif la_ == 5:
                        localctx = AnalogParser.Math_exprContext(self, _parentctx, _parentState)
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_math_expr)
                        self.state = 217
                        if not self.precpred(self._ctx, 5):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 5)")
                        self.state = 218
                        self.match(AnalogParser.POWER)
                        self.state = 219
                        self.math_expr(6)
                        pass

             
                self.state = 224
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,15,self._ctx)

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
        self.enterRule(localctx, 46, self.RULE_math_terminal)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 225
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
        self.enterRule(localctx, 48, self.RULE_math_func_name)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 227
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
        self.enterRule(localctx, 50, self.RULE_math_func)
        try:
            self.state = 241
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [67]:
                self.enterOuterAlt(localctx, 1)
                self.state = 229
                self.match(AnalogParser.ATAN2)
                self.state = 230
                self.match(AnalogParser.LBRACKET)
                self.state = 231
                self.math_expr(0)
                self.state = 232
                self.match(AnalogParser.COMMA)
                self.state = 233
                self.math_expr(0)
                self.state = 234
                self.match(AnalogParser.RBRACKET)
                pass
            elif token in [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]:
                self.enterOuterAlt(localctx, 2)
                self.state = 236
                self.math_func_name()
                self.state = 237
                self.match(AnalogParser.LBRACKET)
                self.state = 238
                self.math_expr(0)
                self.state = 239
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
        self._predicates[16] = self.bool_expr_sempred
        self._predicates[20] = self.operator_expr_sempred
        self._predicates[22] = self.math_expr_sempred
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
         




