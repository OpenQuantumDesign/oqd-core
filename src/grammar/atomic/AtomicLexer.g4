lexer grammar AtomicLexer;

/** ================================================================================= */

WHITESPACE: [ \t]+ -> channel(HIDDEN);
EOL: (';' [\r\n]* | [\r\n]+);
NEWLINE: [\r\n]+ -> channel(HIDDEN);
COMMENT: '//' ~[\n\r]* NEWLINE -> skip;

/** ================================================================================= */

// Statement keyword
MEASURE: 'measure';
PARALLEL: 'parallel';
IF: 'if';
ELSE: 'else';
WHILE: 'while';
WITH: 'with';
FOR: 'for';
BREAK: 'break';
CONTINUE: 'continue';
IONREGISTER: 'ionreg';
BEAM: 'beam';
PULSE: 'pulse';

/** ================================================================================= */

// Boolean
AND: 'and';
AND2: '&&';
OR: 'or';
OR2: '||';
NOT: 'not';
NOT2: '!';
TRUE: 'true';
FALSE: 'false';

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

// Identifier

fragment ID_START: [a-zA-Z_];
fragment ID_CONTINUE: [a-zA-Z0-9_];

ID: ID_START ID_CONTINUE*;
