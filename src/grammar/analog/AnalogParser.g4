parser grammar AnalogParser;

options { tokenVocab = AnalogLexer; }

/** ================================================================================= */

program: block EOF;

block: (statement EOL | EOL)* (statement)?;

statement
    : declaration
    | while_stmt
    | ifelse_stmt
    | break_stmt
    | continue_stmt
    | expr
    ;

/** ================================================================================= */

// Structural control flow


ifelse_stmt
    : IF LBRACKET expr RBRACKET LBRACE block RBRACE (EOL? ELSE LBRACE block RBRACE)?;

while_stmt: WHILE LBRACKET expr RBRACKET LBRACE block RBRACE;

break_stmt: BREAK;
continue_stmt: CONTINUE;

/** ================================================================================= */

// Atom

terminal: analog_list_extract | operator_terminal | math_terminal | bool_literal | analog_list | func;

/** ================================================================================= */

// Variable

declaration: ID ASSIGN expr;
access: ID;

/** ================================================================================= */

// List

analog_list: SQUARELBRACKET expr? (COMMA expr)* COMMA? SQUARERBRACKET;
analog_list_extract: access SQUARELBRACKET expr SQUARERBRACKET;

/** ================================================================================= */

// Quantum operator

pauli_op: (PAULI_I | PAULI_X | PAULI_Y | PAULI_Z) (LBRACKET args? RBRACKET)?;
ladder_op: CREATION | ANNIHILATION | IDENTITY_OP;

operator_terminal: pauli_op | ladder_op;

/** ================================================================================= */

// Boolean

bool_literal: TRUE | FALSE;

not: NOT | NOT2;
and: AND | AND2;
or: OR | OR2;
xor: XOR |XOR2;

/** ================================================================================= */

// Function

math_func: ABS | SIN | COS | TAN | EXP | LOG | SINH | COSH | TANH
    | ATAN | ACOS | ASIN | ATANH | ASINH | ACOSH | ATAN2 | CONJ
    | HEAVISIDE | REAL | IMAG_FN | ROUND;


quantum_func: QUANTUMREGISTER | MODEREGISTER | EVOLVE | MEASURE | INITIALIZE;

list_func: RANGE | PRINT | LENGTH | FLATTEN;

func_names: math_func | quantum_func | list_func;

args: expr (COMMA expr)*;

func: func_names LBRACKET args? RBRACKET;

/** ================================================================================= */

// Arithmetic

real_part: ((INT | FLOAT) REAL_UNIT);

imag_part: ((INT | FLOAT) IMAG_UNIT);

complex: real_part | real_part? imag_part;

math_terminal: INT | FLOAT | MATH_VAR | complex | access | pexpr;

pexpr: LBRACKET expr RBRACKET;

eexpr: terminal | eexpr POWER terminal;

uexpr: eexpr | (PLUS|MINUS|NOT) eexpr;

mexpr: uexpr | mexpr (MULT|DIV|AT) uexpr;

aexpr: mexpr | aexpr (PLUS|MINUS) mexpr;

cexpr: aexpr | cexpr (LT | LEQ | GT | GEQ) aexpr;

eqexpr: cexpr | eqexpr (EQ | NEQ) cexpr;

andexpr: eqexpr | andexpr and eqexpr;

xorexpr: andexpr | xorexpr and andexpr;

orexpr: xorexpr | orexpr and xorexpr;

expr: orexpr;