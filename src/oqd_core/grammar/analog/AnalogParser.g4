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
expr: extract | my_list | aexpr | atom | bool_expr;
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
bool_eq_op: EQ;
bool_not_eq_op: NEQ;
bool_lt_op: LT;
bool_lte_op: LTE;
bool_gt_op: GT;
bool_gte_op: GTE;

bool_expr
    : bool_expr bool_or_op bool_expr
    | bool_expr bool_and_op bool_expr
    | bool_expr bool_eq_op bool_expr
    | bool_expr bool_not_eq_op bool_expr
    | bool_expr bool_lt_op bool_expr
    | bool_expr bool_lte_op bool_expr
    | bool_expr bool_gt_op bool_expr
    | bool_expr bool_gte_op bool_expr
    | bool_not_op bool_expr
    | access
    | LBRACKET bool_expr RBRACKET
    ;

/** ================================================================================= */

// Quantum operator

pauli_op: PAULI_I | PAULI_X | PAULI_Y | PAULI_Z;
ladder_op: CREATION | ANNIHILATION | IDENTITY_OP;
operator_terminal: pauli_op | ladder_op;

/** ================================================================================= */

// Math

math_terminal: INT | FLOAT | MATH_VAR | IMAG | ID | pexpr | fexpr;

math_func_name: ABS | SIN | COS | TAN | EXP | LOG | SINH | COSH | TANH
    | ATAN | ACOS | ASIN | ATANH | ASINH | ACOSH | ATAN2
    | HEAVISIDE | CONJ | REAL | IMAG_FN;

pexpr: LBRACKET aexpr RBRACKET;

fexpr: math_func_name pexpr;

aexpr: mexpr | aexpr WHITESPACE? (PLUS|MINUS|OP_ADD|OP_MINUS) WHITESPACE? mexpr;

mexpr: uexpr | mexpr WHITESPACE? (MULT|DIV|OP_MUL|AT) WHITESPACE? uexpr;

uexpr: eexpr | (PLUS|MINUS) eexpr;

eexpr: atom | atom WHITESPACE? POWER WHITESPACE? uexpr;
