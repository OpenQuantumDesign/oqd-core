The math interface is utilized to specify arbitrary profiles for operands in other interfaces.

## Math Expression <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.math.MathExpr] </div>

[MathExpr][oqd_core.interface.math.MathExpr] represents a mathematical expression.

## Primitives

The primitives of the math interface consist of:

/// tab | `MathNum`

//// html | div[style='float: right']
[![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.math.MathNum]
////

[`MathNum`][oqd_core.interface.math.MathNum] represents a number.

<!-- prettier-ignore -->
/// admonition | Note
    type: note
Numbers are considered to belong to the real numbers, i.e. they can be positive or negative floats.
///
///

/// tab | `MathImag`

//// html | div[style='float: right']
[![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.math.MathImag]
////

[`MathImag`][oqd_core.interface.math.MathImag] represents the imaginary unit.
///

/// tab | `MathVar`

//// html | div[style='float: right']
[![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.math.MathVar]
////

[`MathVar`][oqd_core.interface.math.MathVar] represents a protected named variable to be substituted during compile time or runtime.
///

## Operators

The compatible operators for the math interface consist of:

/// tab | `MathAdd`

//// html | div[style='float: right']
[![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.math.MathAdd]
////

[`MathAdd`][oqd_core.interface.math.MathAdd] represents an addition of two expressions.
///

/// tab | `MathMul`

//// html | div[style='float: right']
[![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.math.MathMul]
////

[`MathMul`][oqd_core.interface.math.MathMul] represents an multiplication of two expressions.
///

/// tab | `MathPow`

//// html | div[style='float: right']
[![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.math.MathPow]
////

[`MathPow`][oqd_core.interface.math.MathPow] represents an exponentiation of an expression with the other expression.
///

/// tab | `MathSub`

//// html | div[style='float: right']
[![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.math.MathSub]
////

[`MathSub`][oqd_core.interface.math.MathSub] represents an subtraction of two expressions.
///

/// tab | `MathDiv`

//// html | div[style='float: right']
[![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.math.MathDiv]
////

[`MathDiv`][oqd_core.interface.math.MathDiv] represents an division of two expressions.
///

/// tab | `MathFunc`

//// html | div[style='float: right']
[![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_core.interface.math.MathFunc]
////

[`MathFunc`][oqd_core.interface.math.MathFunc] represents the application of a named function on an expression.

The compatible named functions include:

- trigonometric (`sin`, `cos`, `tan`)
- inverse trigonometric functions (`asin`, `acos`, `atan`, `atan2`)
- hyperbolic trigonometric (`sinh`, `cosh`, `tanh`)
- inverse hyperbolic trigonometric functions (`asinh`, `acosh`, `atanh`)
- exponential (`exp`)
- logarithm (`log`)
- complex number functions (`real`, `imag`, `conj`, `abs`)
- step function (`heaviside`)

///

## Usage

<!-- prettier-ignore -->
/// admonition | Example
    type: example

$$
10 \sin^2(\omega t + \phi)
$$

```py
expr = (
    10
    * MathFunc(
        func="sin", expr=MathVar(name="omega") * MathVar(name="t") + MathVar(name="phi")
    )
    ** 2
)
```

///
