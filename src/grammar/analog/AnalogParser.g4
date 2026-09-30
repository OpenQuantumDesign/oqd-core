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

terminal: operator_terminal | math_terminal | bool_literal | analog_list | func;

/** ================================================================================= */

// Variable

declaration: ID ASSIGN expr;
access: ID;

/** ================================================================================= */

// Quantum operator

pauli_op: (PAULI_I | PAULI_X | PAULI_Y | PAULI_Z) (LBRACKET args? RBRACKET)?;
ladder_op: CREATION | ANNIHILATION | IDENTITY_OP;

operator_terminal: pauli_op | ladder_op;

/** ================================================================================= */

// Boolean

bool_literal: TRUE | FALSE;

/** ================================================================================= */

// Function

math_func: ABS | SIN | COS | TAN | EXP | LOG | SINH | COSH | TANH
    | ATAN | ACOS | ASIN | ATANH | ASINH | ACOSH | ATAN2 | CONJ
    | HEAVISIDE | REAL | IMAG_FN | ROUND;


quantum_func: QUANTUMREGISTER | MODEREGISTER | EVOLVE | MEASURE | INITIALIZE;

list_func: RANGE | LENGTH | FLATTEN;

misc_func: PRINT;

func_names: math_func | quantum_func | list_func | misc_func;

args: expr (COMMA expr)* COMMA?;

func: func_names LBRACKET args? RBRACKET;

/** ================================================================================= */

// List

analog_list: SQUARELBRACKET expr? (COMMA expr)* COMMA? SQUARERBRACKET;

iexpr: terminal | iexpr SQUARELBRACKET expr SQUARERBRACKET;

/** ================================================================================= */

// Arithmetic

complex: REAL_PART | REAL_PART? IMAG_PART;

math_terminal: INT | FLOAT | MATH_VAR | complex | access | pexpr;

pexpr: LBRACKET expr RBRACKET;

eexpr: iexpr | eexpr POWER iexpr;

uexpr: eexpr | (PLUS|MINUS|NOT) eexpr;

mexpr: uexpr | mexpr (MULT|DIV|AT) uexpr;

aexpr: mexpr | aexpr (PLUS|MINUS) mexpr;

cexpr: aexpr | cexpr (LT | LEQ | GT | GEQ) aexpr;

eqexpr: cexpr | eqexpr (EQ | NEQ) cexpr;

andexpr: eqexpr | andexpr AND eqexpr;

xorexpr: andexpr | xorexpr XOR andexpr;

orexpr: xorexpr | orexpr OR xorexpr;

expr: orexpr;