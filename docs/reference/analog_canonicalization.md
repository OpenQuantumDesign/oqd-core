## Operator Canonicalization
<!-- prettier-ignore -->
::: oqd_core.compiler.analog.operator.canonicalize
    options:
        heading_level: 3
        members: [
            "canonicalize_operator_expr",
            "resolve_operator_expr",
        ]

## Rewrite Rules

<!-- prettier-ignore -->
::: oqd_core.compiler.analog.rewrite.canonicalize
    options:
        heading_level: 3
        members: [
            "OperatorDistribute",
            "GatherMathExpr",
            "GatherPauli",
            "PruneIdentity",
            "PauliAlgebra",
            "NormalOrder",
            "ProperOrder",
            "ScaleTerms",
            "SortedOrder",
            "PruneZeros",
        ]

## Verification Rules

<!-- prettier-ignore -->
::: oqd_core.compiler.analog.verify.canonicalize
    options:
        heading_level: 3
        members: [
            "CanVerPauliAlgebra",
            "CanVerGatherMathExpr",
            "CanVerOperatorDistribute",
            "CanVerProperOrder",
            "CanVerPruneIdentity",
            "CanVerGatherPauli",
            "CanVerNormalOrder",
            "CanVerSortedOrder",
            "CanVerScaleTerm",
        ]