parser grammar AtomicParser;

options { tokenVocab = AtomicLexer; }

/** ================================================================================= */

program: block EOF;

statement
    : declaration
    | pulse_stmt
    | parallel_stmt
    | while_stmt
    | ifelse_stmt
    | break_stmt
    | continue_stmt
    ;

block: (statement EOL | EOL)* (statement)?;

/** ================================================================================= */

atom: ion_register | math_terminal | bool_literal;
expr
    : aexpr (comparators aexpr)+
    | expr (bool_and_op|bool_or_op) expr
    | bool_not_op expr
    | LBRACKET expr RBRACKET
    | atomic_list_extract 
    | atomic_list 
    | atom
    | aexpr
    | beam_expr;
cond: expr;
atomic_list: SQUARELBRACKET expr? (COMMA expr)* SQUARERBRACKET;

declaration: ID ASSIGN expr;
access: ID;
atomic_list_extract: access SQUARELBRACKET INT SQUARERBRACKET;

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
ion_register: IONREGISTER LBRACKET INT RBRACKET;

beam_expr: BEAM LBRACKET expr COMMA expr COMMA expr COMMA vec3 COMMA vec3 RBRACKET;
vec3: SQUARELBRACKET expr COMMA expr COMMA expr SQUARERBRACKET;

parallel_stmt: PARALLEL LBRACE block RBRACE;
pulse_stmt: PULSE targets WITH expr FOR expr (COMMA measured)?;
measured: expr;
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

// Math

math_terminal: INT | FLOAT | MATH_VAR | IMAG | access | pexpr | fexpr;

math_func_name: ABS | SIN | COS | TAN | EXP | LOG | SINH | COSH | TANH
    | ATAN | ACOS | ASIN | ATANH | ASINH | ACOSH | ATAN2
    | HEAVISIDE | CONJ | REAL | IMAG_FN;

pexpr: LBRACKET aexpr RBRACKET;

fexpr: math_func_name pexpr;

aexpr: mexpr | aexpr (PLUS|MINUS) mexpr;

mexpr: uexpr | mexpr (MULT|DIV) uexpr;

uexpr: eexpr | (PLUS|MINUS) eexpr;

eexpr: atom | atom POWER uexpr;
