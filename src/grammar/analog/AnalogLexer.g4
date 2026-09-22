lexer grammar AnalogLexer;

/** ================================================================================= */

WHITESPACE: [ \t]+ -> channel(HIDDEN);
EOL: ([\r\n]+) | COMMENT;
NEWLINE: [\r\n]+ -> channel(HIDDEN);
COMMENT: '//' ~[\n\r]* NEWLINE -> skip;

/** ================================================================================= */

// Statement keyword
EVOLVE: 'evolve';
MEASURE: 'measure';
INITIALIZE: 'initialize';
IF: 'if';
ELSE: 'else';
WHILE: 'while';
WITH: 'with';
FOR: 'for';
BREAK: 'break';
CONTINUE: 'continue';

/** ================================================================================= */

// Boolean
AND: 'and';
AND2: '&&';
OR: 'or';
OR2: '||';
XOR: 'xor';
XOR2: '^^';
NOT: 'not';
NOT2: '!';
TRUE: 'true';
FALSE: 'false';

/** ================================================================================= */

// Register

QUANTUMREGISTER: 'qreg';
MODEREGISTER: 'qmode';


/** ================================================================================= */

// Punctuation
COLON: ':';
SEMICOLON: ';';
COMMA: ',';
LBRACKET: '(';
RBRACKET: ')';
SQUARELBRACKET: '[';
SQUARERBRACKET: ']';
LBRACE: '{';
RBRACE: '}';

// Math Operators

MULT: '*';
DIV: '/';
PLUS: '+';
MINUS: '-';
POWER: '^';
ASSIGN: '=';
EQ: '==';
NEQ: '!=';
LT: '<';
LEQ: '<=';
GT: '>';
GEQ: '>=';
AT: '@';

// Analog Operators


/** ================================================================================= */

// Math

fragment NONZERODIGIT: [1-9];
fragment ZERO: '0';
fragment DIGIT: ZERO | NONZERODIGIT;
fragment DIGITSEQ: DIGIT+;

INT: ZERO | NONZERODIGIT DIGITSEQ?;

FLOAT: INT? '.' DIGITSEQ (('e' | 'E') (PLUS | MINUS)? INT)?
     | INT '.' (DIGITSEQ)? (('e' | 'E') (PLUS | MINUS)? INT)?
     | INT (('e' | 'E') (PLUS | MINUS)? INT);

MATH_VAR: '#' ID;
REAL_UNIT: 'r';
IMAG_UNIT: 'j';

// Math functions
ABS: 'abs';
SIN: 'sin';
COS: 'cos';
TAN: 'tan';
EXP: 'exp';
LOG: 'log';
SINH: 'sinh';
COSH: 'cosh';
TANH: 'tanh';
ATAN: 'atan';
ACOS: 'acos';
ASIN: 'asin';
ATANH: 'atanh';
ASINH: 'asinh';
ACOSH: 'acosh';
HEAVISIDE: 'heavwwwiside';
CONJ: 'conj';
REAL: 'real';
IMAG_FN: 'imag';
ATAN2: 'atan2';
ROUND: 'round';

/** ================================================================================= */

// Builtin functions
RANGE: 'range';
PRINT: 'print';
LENGTH: 'len';
FLATTEN: 'flatten';

/** ================================================================================= */

// Quantum Operator

PAULI_I: '%I';
PAULI_X: '%X';
PAULI_Y: '%Y';
PAULI_Z: '%Z';

CREATION: '%C';
ANNIHILATION: '%A';
IDENTITY_OP: '%J';


/** ================================================================================= */

// Identifier

fragment ID_START: [a-zA-Z_];
fragment ID_CONTINUE: [a-zA-Z0-9_];

ID: ID_START ID_CONTINUE*;
