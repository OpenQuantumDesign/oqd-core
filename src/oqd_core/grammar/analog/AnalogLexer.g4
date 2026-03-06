lexer grammar AnalogLexer;

/** ================================================================================= */

WHITESPACE: [ \t]+ -> channel(HIDDEN);
EOL: ';'? [\r\n]+;
NEWLINE: [\r\n]+ -> channel(HIDDEN);
COMMENT: '//' ~[\n\r]* NEWLINE -> skip;

/** ================================================================================= */

// Statements
EVOLVE: 'evolve';
MEASURE: 'measure';
INITIALIZE: 'initialize';
IF: 'if';
ELSE: 'else';
WHILE: 'while';
FOR: 'for';
ON: 'on';

/** ================================================================================= */

// Boolean
AND: 'and';
AND2: '&&';
OR: 'or';
OR2: '||';
NOT: 'not';
NOT2: '!';

/** ================================================================================= */

REGISTER: 'register';
CREATION: 'creation';
A_DAG: 'a_dag';
ANNIHILATION: 'annihilation';
IDENTITY_OP: 'identity';

PAULI_I: '%I';
PAULI_X: '%X';
PAULI_Y: '%Y';
PAULI_Z: '%Z';

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
AT: '@';

MULT: '*';
DIV: '/';
PLUS: '+';
MINUS: '-';
POWER: '^';
EQ: '=';

/** ================================================================================= */

fragment NONZERODIGIT: [1-9];
fragment ZERO: '0';
fragment DIGIT: ZERO | NONZERODIGIT;
fragment DIGITSEQ: DIGIT+;

INT: ZERO | NONZERODIGIT DIGITSEQ?;

FLOAT: INT? '.' DIGITSEQ (('e' | 'E') (PLUS | MINUS)? INT)?
     | INT '.' (DIGITSEQ)? (('e' | 'E') (PLUS | MINUS)? INT)?
     | INT (('e' | 'E') (PLUS | MINUS)? INT);

MATH_VAR: '#' [a-zA-Z] [a-zA-Z0-9_]*;
IMAG: INT? '.'? DIGITSEQ? 'j' | INT 'j';

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

fragment ID_START: [a-zA-Z_];
fragment ID_CONTINUE: [a-zA-Z0-9_];

ID: ID_START ID_CONTINUE*;
