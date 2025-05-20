# Copyright 2024-2025 Open Quantum Design

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional
import itertools

from numpy.typing import NDArray
import numpy as np

from oqd_core.compiler.analog.passes import analog_operator_canonicalization
from oqd_core.compiler.math.passes import evaluate_math_expr
from oqd_core.interface.analog import AnalogGate
from oqd_core.interface.analog.operator import OperatorSubtypes
from oqd_core.interface.analog.operator import OperatorBinaryOp
from oqd_core.interface.analog.operator import OperatorTerminal
from oqd_core.interface.analog.operator import OperatorAdd
from oqd_core.interface.analog.operator import OperatorSub
from oqd_core.interface.analog.operator import OperatorKron
from oqd_core.interface.analog.operator import OperatorScalarMul
from oqd_core.interface.analog.operator import Annihilation
from oqd_core.interface.analog.operator import Creation
from oqd_core.interface.analog.operator import Identity
from oqd_core.interface.analog.operator import Pauli
from oqd_core.interface.analog.operator import PauliI
from oqd_core.interface.analog.operator import PauliX
from oqd_core.interface.analog.operator import PauliY
from oqd_core.interface.analog.operator import PauliZ
from oqd_core.interface.math import MathExprSubtypes
from oqd_core.interface.math import MathTerminal
from oqd_core.interface.math import MathVar
from oqd_core.interface.math import MathUnaryOp
from oqd_core.interface.math import MathBinaryOp


__all__ = [
    "isinglike_analysis",
]


@dataclass
class _PauliStringTerm:
    coefficient: np.complex128
    operators: list[Pauli]


@dataclass
class _PauliTermInfo:
    index: int
    term: Pauli

    def index_and_string(self) -> tuple[int, str]:
        return self.index, _pauli_to_char(self.term)

    def __lt__(self, other) -> bool:
        if not isinstance(other, _PauliTermInfo):
            return NotImplemented
        return _pauli_to_char(self.term) < _pauli_to_char(other.term)


@dataclass
class _PauliStringTwoWeightInfo:
    coefficient: np.complex128
    pauli_term0: _PauliTermInfo
    pauli_term1: _PauliTermInfo


def isinglike_analysis(gate: AnalogGate) -> dict[str, NDArray[np.complex128]]:
    """
    Creating the dictionary of coupling matrices for an `AnalogGate` instance that
    implements an Ising-like Hamiltonian.

    Raises a RuntimeError in any of the following cases:
        - a bosonic mode operator is found in Hamiltonian expression
        - a time-dependent parameter is found in Hamiltonian expression
        - at least one Pauli string is not a two-weight Pauli string
        - the analog operator canonicalization process raised an AssertionError
        - the recursive traversal function to collect the Pauli strings found an operator
          that it could not evaluate
        - the Pauli strings are not all the same length
    """
    if _has_bosonic_operator(gate.hamiltonian):
        raise RuntimeError("ERROR: found bosonic mode operator in Hamiltonian")

    try:
        canonicalized: AnalogGate = analog_operator_canonicalization(gate)
    except AssertionError as e:
        raise RuntimeError(f"ERROR: canonicalization function raised exception: {e}\n")

    pauli_strings = _traverse(canonicalized.hamiltonian)
    if len(pauli_strings) == 0:
        return {}

    if not _are_all_pauli_strings_the_same_length(pauli_strings):
        raise RuntimeError("ERROR: the Pauli strings are not all the same length\n")

    two_weights: list[_PauliStringTwoWeightInfo] = []
    for string in pauli_strings:
        two_weight = _get_pauli_string_two_weight_info(string)
        if two_weight is None:
            raise RuntimeError("ERROR: found a Pauli string that isn't two-weight\n")
        else:
            two_weights.append(two_weight)

    n_qubits = len(pauli_strings[0].operators)

    return _build_coupling_matrix(two_weights, n_qubits)


def _build_coupling_matrix(
    two_weights: list[_PauliStringTwoWeightInfo], n_qubits: int
) -> dict[str, NDArray[np.complex128]]:
    """
    Creating the dictionary of coupling matrices from the two-weight information.

    Args:
        two_weights (list[_PauliStringTwoWeightInfo]): the coefficient and position/operator type information about
            the two non-identity Pauli operators
        n_qubits (int): the total number of qubits in the system

    Returns:
        dict[str, NDArray[np.complex128]]: the coupling matrices for the two-weight Pauli strings
    """
    coupling_matrices: dict[str, NDArray[np.complex128]] = {}

    for two_weight in two_weights:
        # this sorting guarantees that (pauli_term_min, pauli_term_max) is one of
        #     (X, Y), (X, Y), (X, Z), (Y, Y), (Y, Z), (Z, Z)
        pauli_term_min, pauli_term_max = sorted(
            [two_weight.pauli_term0, two_weight.pauli_term1]
        )
        i_min, pauli_str_min = pauli_term_min.index_and_string()
        i_max, pauli_str_max = pauli_term_max.index_and_string()

        matrix_key = f"{pauli_str_min}{pauli_str_max}"

        if matrix_key not in coupling_matrices:
            coupling_matrices[matrix_key] = np.zeros(
                (n_qubits, n_qubits), dtype=np.complex128
            )

        if pauli_str_min == pauli_str_max:
            coupling_matrices[matrix_key][i_min, i_max] = two_weight.coefficient
            coupling_matrices[matrix_key][i_max, i_min] = two_weight.coefficient
        else:
            coupling_matrices[matrix_key][i_min, i_max] = two_weight.coefficient

    return coupling_matrices


def _pauli_to_char(pauli: Pauli) -> str:
    """
    Casts a Pauli operator to their respective string representation.
    """
    if isinstance(pauli, PauliX):
        return "X"
    elif isinstance(pauli, PauliY):
        return "Y"
    elif isinstance(pauli, PauliZ):
        return "Z"
    else:
        assert False, "unreachable; only PauliX, PauliY, and PauliZ can be present"


def _traverse(op: OperatorSubtypes) -> list[_PauliStringTerm]:
    """
    Recursively traverse the operator AST and collect all the Pauli strings and their coefficients.

    Args:
        op (OperatorSubtypes): the root of the operator AST

    Raises:
        RuntimeError:
          - if a time-dependent parameter is found in the Hamiltonian expression
          - if the recursive traversal function to collect the Pauli strings found
            an operator that it could not evaluate
          - if a math expression could not be cast to a `np.complex128`

    Returns:
        list[_PauliStringTerm]: a sequence of all the Pauli strings and the coefficients
    """
    if isinstance(op, Pauli):
        return [_PauliStringTerm(np.complex128(1.0, 0.0), [op])]
    elif isinstance(op, OperatorAdd):
        left = _traverse(op.op1)
        right = _traverse(op.op2)
        return left + right
    elif isinstance(op, OperatorSub):
        left = _traverse(op.op1)
        right = _traverse(op.op2)
        right = _rescale_all_coefficients(right, -1)
        return left + right
    elif isinstance(op, OperatorScalarMul):
        if _has_time_dependent_parameter(op.expr):
            raise RuntimeError("ERROR: time-dependent parameter in Hamiltonian.\n")
        value = _evaluate_math_expr_to_complex_float(op.expr)
        products = _traverse(op.op)
        products = _rescale_all_coefficients(products, value)
        return products
    elif isinstance(op, OperatorKron):
        left_products = _traverse(op.op1)
        right_products = _traverse(op.op2)
        products = []
        for left, right in itertools.product(left_products, right_products):
            products.append(
                _PauliStringTerm(
                    coefficient=left.coefficient * right.coefficient,
                    operators=left.operators + right.operators,
                )
            )
        return products
    else:
        raise RuntimeError(f"ERROR: cannot handle the following operator: {type(op)}")


def _are_all_pauli_strings_the_same_length(strings: list[_PauliStringTerm]) -> bool:
    """
    Returns whether or not all Pauli strings in `strings` are the same length.
    """

    if len(strings) == 0:
        assert False, "unreachable; there must be at least one Pauli string"

    if len(strings) == 1:
        return True
    else:
        size = len(strings[0].operators)
        return all([size == len(prod.operators) for prod in strings[1:]])


def _has_bosonic_operator(op: OperatorSubtypes) -> bool:
    """
    Recursively traverses the tree and checks if any of the Annihilation, Creation, or Identity
    operators are present.
    """
    if isinstance(op, OperatorTerminal):
        return isinstance(op, (Annihilation, Creation, Identity))
    elif isinstance(op, OperatorBinaryOp):
        return _has_bosonic_operator(op.op1) or _has_bosonic_operator(op.op2)
    elif isinstance(op, OperatorScalarMul):
        return _has_bosonic_operator(op.op)
    else:
        assert False, "unreachable; all OperatorSubtypes accounted for"


def _has_time_dependent_parameter(expr: MathExprSubtypes) -> bool:
    """
    Recursively traverses the math expression tree and checks if `MathVar` is present.
    """
    if isinstance(expr, MathTerminal):
        return isinstance(expr, MathVar)
    elif isinstance(expr, MathUnaryOp):
        return _has_time_dependent_parameter(expr.expr)
    elif isinstance(expr, MathBinaryOp):
        return _has_time_dependent_parameter(
            expr.expr1
        ) or _has_time_dependent_parameter(expr.expr2)
    else:
        assert False, "unreachable; all MathExprSubtypes accounted for"


def _get_pauli_string_two_weight_info(
    pauli_string: _PauliStringTerm,
) -> Optional[_PauliStringTwoWeightInfo]:
    """
    Extract information about the two non-identity Pauli operator terms in the Pauli string.

    If the Pauli string is a valid two-weight Pauli string, this functions returns an object
    that holds the string's coefficient, and the indices and Pauli operator types of the
    two non-identity Pauli operators.

    If the Pauli string is not a two-weight Pauli string, this function returns None.

    Args:
        string (_PauliStringTerm): the Pauli string to be analyzed
    """
    if any([not isinstance(op, Pauli) for op in pauli_string.operators]):
        assert False, "unreachable; only Pauli operators should be in the Pauli string"

    paulis: list[_PauliTermInfo] = []
    for i, op in enumerate(pauli_string.operators):
        if isinstance(op, PauliI):
            continue

        if len(paulis) >= 2:
            return None

        paulis.append(_PauliTermInfo(i, deepcopy(op)))

    if len(paulis) != 2:
        return None

    return _PauliStringTwoWeightInfo(pauli_string.coefficient, paulis[0], paulis[1])


def _rescale_all_coefficients(
    pauli_strings: list[_PauliStringTerm],
    value: int | float | np.complex128,
) -> list[_PauliStringTerm]:
    """
    Multiply the coefficient of each Pauli string in `pauli_strings` by `value`.

    Args:
        pauli_strings (list[_PauliStringTerm]): the Pauli strings to be rescaled
        value (int | float | np.complex128): the value to rescale them by

    Returns:
        list[_PauliStringTerm]: the rescaled Pauli strings
    """
    output: list[_PauliStringTerm] = deepcopy(pauli_strings)
    for elem in output:
        elem.coefficient *= value

    return output


def _evaluate_math_expr_to_complex_float(expr: MathExprSubtypes) -> np.complex128:
    """
    This function is a wrapper around the `evaluate_math_expr()` function that ensures that
    the output can be converted to a scalar `np.complex128`.
    """
    value = evaluate_math_expr(expr)
    if isinstance(value, complex):
        return np.complex128(value)
    if isinstance(value, int) or isinstance(value, float):
        return np.complex128(value, 0.0)
    else:
        raise RuntimeError(
            "ERROR: unable to cast math expression to a `np.complex128`."
        )
