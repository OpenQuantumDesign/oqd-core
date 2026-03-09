parser grammar AnalogParser;

options { tokenVocab = AnalogLexer; }

/** ================================================================================= */

program: block statement? EOF;

statement
    : declaration
    | evolve_stmt
    | measure_stmt
    | init_stmt
    | while_stmt
    | ifelse_stmt
    ;

block: (statement EOL)*;

/** ================================================================================= */

atom: mode_register | quantum_register | operator_terminal | math_terminal | access;
expr: extract | my_list | bool_expr | operator_expr | math_expr | atom;
my_list: SQUARELBRACKET expr? (COMMA expr)* SQUARERBRACKET;

declaration: ID ASSIGN expr;
access: ID;
extract: access SQUARELBRACKET INT SQUARERBRACKET;

/** ================================================================================= */


// Structural control flow

while_stmt: WHILE WHITESPACE expr WHITESPACE? COLON EOL block;

ifelse_stmt
    : IF WHITESPACE expr WHITESPACE? COLON EOL block
    | IF WHITESPACE expr WHITESPACE? COLON EOL block ELSE COLON EOL block;


/** ================================================================================= */

// Quantum

quantum_register: QUANTUMREGISTER LBRACKET INT RBRACKET;
mode_register: MODEREGISTER LBRACKET INT RBRACKET;

evolve_stmt: EVOLVE targets WITH expr FOR expr;
measure_stmt: MEASURE targets;
init_stmt: INITIALIZE targets;
targets: expr;

/** ================================================================================= */

// Boolean

bool_and_op: AND | AND2;
bool_or_op: OR | OR2;
bool_not_op: NOT | NOT2;
bool_ref: ID;

bool_expr
    : bool_expr bool_or_op bool_expr
    | bool_expr bool_and_op bool_expr
    | bool_not_op bool_expr
    | math_expr EQ math_expr
    | bool_ref
    | LBRACKET bool_expr RBRACKET
    ;

/** ================================================================================= */

// Quantum operator

pauli_op: PAULI_I | PAULI_X | PAULI_Y | PAULI_Z;
ladder_op: CREATION | ANNIHILATION | IDENTITY_OP;
operator_terminal: pauli_op | ladder_op;

operator_expr
    : operator_expr PLUS operator_expr
    | operator_expr MINUS operator_expr
    | operator_expr AT operator_expr
    | operator_expr MULT operator_expr
    | math_expr MULT operator_expr
    | operator_expr MULT math_expr
    | operator_terminal
    | access
    | LBRACKET operator_expr RBRACKET
    ;

/** ================================================================================= */

// Math

math_terminal: INT | FLOAT | MATH_VAR | IMAG | ID;

math_expr
    : math_expr PLUS math_expr
    | math_expr MINUS math_expr
    | math_expr MULT math_expr
    | math_expr DIV math_expr
    | math_expr POWER math_expr
    | (PLUS | MINUS) math_expr
    | math_terminal
    | math_func
    | access
    | LBRACKET math_expr RBRACKET
    ;

math_func_name: ABS | SIN | COS | TAN | EXP | LOG | SINH | COSH | TANH
    | ATAN | ACOS | ASIN | ATANH | ASINH | ACOSH
    | HEAVISIDE | CONJ | REAL | IMAG_FN;

math_func: ATAN2 LBRACKET math_expr COMMA math_expr RBRACKET
    | math_func_name LBRACKET math_expr RBRACKET;
