parser grammar AnalogParser;

options { tokenVocab = AnalogLexer; }

/** ================================================================================= */

program: block EOF;

statement
    : declaration
    | evolve_stmt
    | measure_stmt
    | init_stmt
    | while_stmt
    | ifelse_stmt
    | break_stmt
    | continue_stmt
    ;

block: (statement EOL | EOL)* (statement)?;

/** ================================================================================= */

atom: mode_register | quantum_register | operator_terminal | math_terminal | access;
expr: extract | my_list | atom | aexpr | bool_literal;
cond: bool_expr;
my_list: SQUARELBRACKET expr? (COMMA expr)* SQUARERBRACKET;

declaration: ID ASSIGN expr;
access: ID;
extract: access SQUARELBRACKET INT SQUARERBRACKET;

/** ================================================================================= */


// Structural control flow

break_stmt: BREAK;
continue_stmt: CONTINUE;

while_stmt: WHILE LBRACKET cond RBRACKET LBRACE block RBRACE;

ifelse_stmt
    : IF LBRACKET cond RBRACKET LBRACE block RBRACE
    | IF LBRACKET cond RBRACKET LBRACE block RBRACE EOL? ELSE LBRACE block RBRACE;

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
bool_literal: TRUE | FALSE;

bool_expr
    : bool_expr (bool_and_op | bool_or_op | bool_eq_op) bool_expr
    | bool_expr (bool_not_eq_op | bool_lt_op | bool_lte_op | bool_gt_op | bool_gte_op) bool_expr
    | bool_not_op bool_expr
    | bool_literal
    | access
    | atom
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

aexpr: mexpr | aexpr (PLUS|MINUS|OP_ADD|OP_MINUS) mexpr;

mexpr: uexpr | mexpr (MULT|DIV|OP_MUL|AT) uexpr;

uexpr: eexpr | (PLUS|MINUS) eexpr;

eexpr: atom | atom POWER uexpr;
