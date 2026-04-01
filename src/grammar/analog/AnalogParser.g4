parser grammar AnalogParser;

options { tokenVocab = AnalogLexer; }

/** ================================================================================= */

program: block EOF;

statement
    : declaration
    | while_stmt
    | ifelse_stmt
    | break_stmt
    | continue_stmt
    ;

block: (statement EOL | EOL)* (statement)?;

/** ================================================================================= */

atom: mode_register | quantum_register | operator_terminal | math_terminal | bool_literal;
expr
    : aexpr (comparators aexpr)+
    | expr (bool_and_op|bool_or_op) expr
    | bool_not_op expr
    | LBRACKET expr RBRACKET
    | analog_list_extract
    | analog_list
    | atom
    | aexpr
    | evolve_expr
    | measure_expr
    | init_expr;
cond: expr;
analog_list: SQUARELBRACKET expr? (COMMA expr)* SQUARERBRACKET;

declaration: ID ASSIGN expr;
access: ID;
analog_list_extract: access SQUARELBRACKET INT SQUARERBRACKET;

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

evolve_expr: EVOLVE targets WITH expr FOR expr;
measure_expr: MEASURE targets;
init_expr: INITIALIZE targets;
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

comparators
    : bool_eq_op
    | bool_not_eq_op
    | bool_lt_op
    | bool_lte_op
    | bool_gt_op
    | bool_gte_op
    ;

/** ================================================================================= */

// Quantum operator

pauli_op: PAULI_I | PAULI_X | PAULI_Y | PAULI_Z;
ladder_op: CREATION | ANNIHILATION | IDENTITY_OP;
operator_terminal: pauli_op | ladder_op;

/** ================================================================================= */

// Math

math_terminal: INT | FLOAT | MATH_VAR | IMAG | access | pexpr | fexpr;

math_func_name: ABS | SIN | COS | TAN | EXP | LOG | SINH | COSH | TANH
    | ATAN | ACOS | ASIN | ATANH | ASINH | ACOSH | ATAN2
    | HEAVISIDE | CONJ | REAL | IMAG_FN;

pexpr: LBRACKET aexpr RBRACKET;

fexpr: math_func_name pexpr;

aexpr: mexpr | aexpr (PLUS|MINUS|OP_ADD|OP_MINUS) mexpr;

mexpr: uexpr | mexpr (MULT|DIV|OP_MUL|AT) uexpr;

uexpr: eexpr | (PLUS|MINUS) eexpr;

eexpr: atom | atom POWER uexpr;
