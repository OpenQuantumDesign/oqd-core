parser grammar AnalogParser;

options { tokenVocab = AnalogLexer; }

/** ================================================================================= */

program: (statement EOL)* statement? EOF;

statement
    : declaration
    | evolve_stmt
    | measure_stmt
    | init_stmt
    ;

/** ================================================================================= */

declaration: (ID | MATH_VAR) EQ decl_value;
decl_value: bool_expr | atomic_type | operator_expr | math_expr;

/** ================================================================================= */

atomic_type: quantum_register | quantum_bit | my_list | access;
quantum_register: REGISTER INT;
quantum_bit: ID SQUARELBRACKET INT SQUARERBRACKET;
my_list: SQUARELBRACKET atomic_type (COMMA atomic_type)* SQUARERBRACKET;
access: ID;

evolve_stmt: EVOLVE operator_expr FOR math_expr ON targets;
targets: atomic_type;

measure_stmt: MEASURE;
init_stmt: INITIALIZE;

bool_and_op: AND | AND2;
bool_or_op: OR | OR2;
bool_not_op: NOT | NOT2;

bool_expr
    : bool_expr bool_or_op bool_expr
    | bool_expr bool_and_op bool_expr
    | bool_not_op bool_expr
    | bool_ref
    | LBRACKET bool_expr RBRACKET
    ;
bool_ref: ID;

/** ================================================================================= */

pauli_op: PAULI_I | PAULI_X | PAULI_Y | PAULI_Z;
ladder_op: CREATION | A_DAG | ANNIHILATION | IDENTITY_OP;

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
operator_terminal: pauli_op | ladder_op;

/** ================================================================================= */

math_expr
    : math_expr PLUS math_expr
    | math_expr MINUS math_expr
    | math_expr MULT math_expr
    | math_expr DIV math_expr
    | math_expr POWER math_expr
    | (PLUS | MINUS) math_expr
    | math_terminal
    | math_func
    | LBRACKET math_expr RBRACKET
    ;
math_terminal: INT | FLOAT | MATH_VAR | IMAG | ID;

math_func_name: ABS | SIN | COS | TAN | EXP | LOG | SINH | COSH | TANH
    | ATAN | ACOS | ASIN | ATANH | ASINH | ACOSH
    | HEAVISIDE | CONJ | REAL | IMAG_FN;

math_func: ATAN2 LBRACKET math_expr COMMA math_expr RBRACKET
    | math_func_name LBRACKET math_expr RBRACKET;
