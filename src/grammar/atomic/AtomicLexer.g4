lexer grammar AtomicLexer;

/** ================================================================================= */

WHITESPACE: [ \t]+ -> channel(HIDDEN);
EOL: ([\r\n]+ | COMMENT);
NEWLINE: [\r\n]+ -> channel(HIDDEN);
COMMENT: '//' ~[\n\r]* NEWLINE -> skip;

/** ================================================================================= */

// Statement keyword
PARALLEL: 'parallel';
SERIAL: 'serial';
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
AND: 'and' | '&&';
OR: 'or' | '||';
XOR: 'xor' | '^^';
NOT: 'not' | '!';
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
LEQ: '<=';
GT: '>';
GEQ: '>=';

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
REAL_PART:  (INT | FLOAT) 'r';
IMAG_PART:  (INT | FLOAT) 'j';

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
ROUND: 'round';
POSF: '_pos';
NEGF: '_neg';
ADDF: '_add';
SUBF: '_sub';
MULF: '_mul';
DIVF: '_div';
POWF: '_pow';
EQF: '_eq';
NEQF: '_neq';
LTF: '_lt';
LEQF: '_leq';
GTF: '_gt';
GEQF: '_geq';
ANDF: '_and';
ORF: '_or';
XORF: '_xor';
NOTF: '_not';

/** ================================================================================= */

// Builtin functions
RANGE: 'range';
PRINT: 'print';
LENGTH: 'len';
FLATTEN: 'flatten';

/** ================================================================================= */

// Identifier

fragment ID_START: [a-zA-Z_];
fragment ID_CONTINUE: [a-zA-Z0-9_];

ID: ID_START ID_CONTINUE*;
