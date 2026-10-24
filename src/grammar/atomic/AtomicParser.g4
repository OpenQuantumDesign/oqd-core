parser grammar AtomicParser;

options { tokenVocab = AtomicLexer; }

/** ================================================================================= */

program: block EOF;

statement
    : declaration
    | parallel_stmt
    | serial_stmt
    | while_stmt
    | ifelse_stmt
    | break_stmt
    | continue_stmt
    | expr
    ;

block: (statement EOL | EOL)* (statement)?;

parallel_stmt: PARALLEL LBRACE block RBRACE;
serial_stmt: SERIAL LBRACE block RBRACE;

/** ================================================================================= */

// Structural control flow

ifelse_stmt
    : IF LBRACKET expr RBRACKET LBRACE block RBRACE (EOL? ELSE LBRACE block RBRACE)?;

while_stmt: WHILE LBRACKET expr RBRACKET LBRACE block RBRACE;

break_stmt: BREAK;
continue_stmt: CONTINUE;

/** ================================================================================= */

// Atom

terminal: math_terminal | bool_literal | atomic_list | func;

/** ================================================================================= */

// Variable

declaration: ID ASSIGN expr;
access: ID;

/** ================================================================================= */

// Boolean

bool_literal: TRUE | FALSE;

/** ================================================================================= */

// Function

arith_func: POSF | NEGF | ADDF | SUBF | MULF | DIVF | POWF | EQF | NEQF
    | LTF | LEQF | GTF | LEQF | ANDF | ORF | XORF | NOTF;

math_func: ABS | SIN | COS | TAN | EXP | LOG | SINH | COSH | TANH
    | ATAN | ACOS | ASIN | ATANH | ASINH | ACOSH | ATAN2 | CONJ
    | HEAVISIDE | REAL | IMAG_FN | ROUND;

quantum_func: IONREGISTER | BEAM | PULSE;

list_func: RANGE | LENGTH | FLATTEN;

misc_func: PRINT;

func_names: math_func | quantum_func | list_func | misc_func | arith_func;

args: expr (COMMA expr)* COMMA?;

func: func_names LBRACKET args? RBRACKET;

/** ================================================================================= */

// List

atomic_list: SQUARELBRACKET expr? (COMMA expr)* COMMA? SQUARERBRACKET;

iexpr: terminal | iexpr SQUARELBRACKET expr SQUARERBRACKET;

/** ================================================================================= */

// Arithmetic

complex: REAL_PART | REAL_PART? IMAG_PART;

math_terminal: INT | FLOAT | MATH_VAR | complex | access | pexpr;

pexpr: LBRACKET expr RBRACKET;

eexpr: iexpr | iexpr POWER uexpr;

uexpr: eexpr | (PLUS|MINUS|NOT) uexpr;

mexpr: uexpr | mexpr (MULT uexpr)+ | mexpr (DIV uexpr)+;

aexpr: mexpr | aexpr (PLUS mexpr)+ | aexpr (MINUS mexpr)+;

cexpr: aexpr | cexpr (LT | LEQ | GT | GEQ) aexpr;

eqexpr: cexpr | eqexpr (EQ | NEQ) cexpr;

andexpr: eqexpr | eqexpr (AND eqexpr)+;

xorexpr: andexpr | andexpr (XOR andexpr)+;

orexpr: xorexpr | xorexpr (OR xorexpr)+;

expr: orexpr;
