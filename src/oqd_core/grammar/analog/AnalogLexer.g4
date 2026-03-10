lexer grammar AnalogLexer;

/** ================================================================================= */

WHITESPACE: [ \t]+ -> channel(HIDDEN);
EOL: (';' [\r\n]* | [\r\n]+);
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

/** ================================================================================= */

// Boolean
AND: 'and';
AND2: '&&';
OR: 'or';
OR2: '||';
NOT: 'not';
NOT2: '!';

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
LTE: '<=';
GT: '>';
GTE: '>=';

// Analog Operators

AT: '%@';
OP_ADD: '%+';
OP_MUL: '%*';
OP_MINUS: '%-';

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
IMAG: '1j';

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
HEAVISIDE: 'heaviside';
CONJ: 'conj';
REAL: 'real';
IMAG_FN: 'imag';
ATAN2: 'atan2';


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
