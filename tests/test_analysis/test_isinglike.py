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

import pytest

from typing import Optional

import numpy as np

from oqd_core.interface.analog import AnalogGate
from oqd_core.interface.analog.operator import OperatorSubtypes
from oqd_core.interface.analog.operator import OperatorAdd
from oqd_core.interface.analog.operator import Annihilation
from oqd_core.interface.analog.operator import Creation
from oqd_core.interface.analog.operator import Identity
from oqd_core.interface.analog.operator import Pauli
from oqd_core.interface.analog.operator import PauliI
from oqd_core.interface.analog.operator import PauliX
from oqd_core.interface.analog.operator import PauliY
from oqd_core.interface.analog.operator import PauliZ
from oqd_core.interface.math import MathExprSubtypes
from oqd_core.interface.math import MathVar
from oqd_core.interface.math import MathAdd
from oqd_core.interface.math import MathNum

from oqd_core.analysis.isinglike import isinglike_analysis
from oqd_core.analysis.isinglike import _ALLOWED_ISINGLIKE_TERMS
from oqd_core.analysis.isinglike import _are_all_pauli_strings_the_same_length
from oqd_core.analysis.isinglike import _build_coupling_matrix
from oqd_core.analysis.isinglike import _get_pauli_string_two_weight_info
from oqd_core.analysis.isinglike import _has_bosonic_operator
from oqd_core.analysis.isinglike import _has_mathvar
from oqd_core.analysis.isinglike import _pauli_to_char
from oqd_core.analysis.isinglike import _PauliStringTerm
from oqd_core.analysis.isinglike import _PauliTermInfo
from oqd_core.analysis.isinglike import _PauliStringTwoWeightInfo
from oqd_core.analysis.isinglike import _rescale_all_coefficients


I = Identity()
A = Annihilation()
C = Creation()
ID = PauliI()
X = PauliX()
Y = PauliY()
Z = PauliZ()
PST = _PauliStringTerm


@pytest.fixture
def default_pauli_strings() -> list[_PauliStringTerm]:
    return [
        _PauliStringTerm(1.0, [X, Y, Z]),
        _PauliStringTerm(2.0, [X, Y, Z]),
        _PauliStringTerm((3.0 + 1.0j), [X, Y, Z]),
    ]


@pytest.mark.parametrize(
    "value, expected",
    [
        (1, ((1.0 + 0.0j), (2.0 + 0.0j), (3.0 + 1.0j))),
        (2, ((2.0 + 0.0j), (4.0 + 0.0j), (6.0 + 2.0j))),
        (3, ((3.0 + 0.0j), (6.0 + 0.0j), (9.0 + 3.0j))),
        (2.0 + 3.0j, ((2.0 + 3.0j), (4.0 + 6.0j), (3.0 + 11.0j))),
    ],
)
def test_rescale_all_coefficients(
    value: int | float | np.complex128,
    expected: tuple[np.complex128],
    default_pauli_strings: list[_PauliStringTerm],
) -> None:
    rescaled = _rescale_all_coefficients(default_pauli_strings, value)

    assert np.allclose(rescaled[0].coefficient, expected[0])
    assert np.allclose(rescaled[1].coefficient, expected[1])
    assert np.allclose(rescaled[2].coefficient, expected[2])


@pytest.mark.parametrize(
    "pauli_string, expected",
    [
        (
            _PauliStringTerm(1.0, [X, Y]),
            _PauliStringTwoWeightInfo(
                np.complex128(1.0, 0.0), _PauliTermInfo(0, X), _PauliTermInfo(1, Y)
            ),
        ),
        (
            _PauliStringTerm(2.0, [X, Y]),
            _PauliStringTwoWeightInfo(
                np.complex128(2.0, 0.0), _PauliTermInfo(0, X), _PauliTermInfo(1, Y)
            ),
        ),
        (
            _PauliStringTerm(1.0, [X, Z]),
            _PauliStringTwoWeightInfo(
                np.complex128(1.0, 0.0), _PauliTermInfo(0, X), _PauliTermInfo(1, Z)
            ),
        ),
        (
            _PauliStringTerm(1.0, [X, ID, Z]),
            _PauliStringTwoWeightInfo(
                np.complex128(1.0, 0.0), _PauliTermInfo(0, X), _PauliTermInfo(2, Z)
            ),
        ),
        (
            _PauliStringTerm(1.0, [ID, ID, Y, ID, X, ID, ID, ID]),
            _PauliStringTwoWeightInfo(
                np.complex128(1.0, 0.0), _PauliTermInfo(2, Y), _PauliTermInfo(4, X)
            ),
        ),
        (_PauliStringTerm(1.0, [X]), None),
        (_PauliStringTerm(1.0, [X, ID]), None),
        (_PauliStringTerm(1.0, [ID, X, ID]), None),
        (_PauliStringTerm(1.0, [X, Y, Z]), None),
        (_PauliStringTerm(1.0, [ID, X, ID, Y, Z]), None),
    ],
)
def test_get_pauli_string_two_weight_info(
    pauli_string: _PauliStringTerm, expected: Optional[_PauliStringTwoWeightInfo]
) -> None:
    actual = _get_pauli_string_two_weight_info(pauli_string)

    if actual is None:
        assert actual is expected
    else:
        assert actual == expected


@pytest.mark.parametrize(
    "expr, expected",
    [
        (MathAdd(expr1=MathNum(value=1), expr2=MathNum(value=2)), False),
        (MathAdd(expr1=MathNum(value=1), expr2=MathVar(name="x")), True),
    ],
)
def test_has_mathvar(expr: MathExprSubtypes, expected: bool) -> None:
    assert _has_mathvar(expr) == expected


@pytest.mark.parametrize(
    "op, expected",
    [
        (OperatorAdd(op1=X, op2=Y), False),
        (OperatorAdd(op1=X, op2=OperatorAdd(op1=Y, op2=C)), True),
        (OperatorAdd(op1=X, op2=OperatorAdd(op1=A, op2=Z)), True),
        (OperatorAdd(op1=I, op2=OperatorAdd(op1=I, op2=C)), True),
    ],
)
def test_bosonic_operator(op: OperatorSubtypes, expected: bool) -> None:
    assert _has_bosonic_operator(op) == expected


@pytest.mark.parametrize(
    "expr, expected",
    [
        ([PST(1, [X, Y, Z]), PST(1, [X, Y, Z]), PST(1, [X, Y, Z])], True),
        ([PST(1, [ID, Y, Z]), PST(1, [X, Y, ID])], True),
        ([PST(1, [ID, Y, Z])], True),
        ([PST(1, [Y, Z]), PST(1, [X, Z]), PST(1, [X, ID])], True),
        ([PST(1, [X, Y, Y, Z]), PST(1, [X, ID, Y, Z]), PST(1, [X, ID, Y, Z])], True),
        ([PST(1, [X, Y]), PST(1, [X, Y, Z]), PST(1, [X, Y, Z])], False),
        ([PST(1, [Z]), PST(1, [Z]), PST(1, [X, Y, ID])], False),
        ([PST(1, [Y, Z]), PST(1, [X, Z]), PST(1, [X, ID, Y])], False),
        ([PST(1, [X, Y, Y, Z]), PST(1, [Y, Z]), PST(1, [X, ID, Y, Z])], False),
    ],
)
def test_are_all_pauli_strings_the_same_length(
    expr: MathExprSubtypes, expected: bool
) -> None:
    assert _are_all_pauli_strings_the_same_length(expr) == expected


@pytest.mark.parametrize(
    "pauli, expected",
    [
        (X, "X"),
        (Y, "Y"),
        (Z, "Z"),
    ],
)
def test_pauli_to_char(pauli: Pauli, expected: str) -> None:
    assert _pauli_to_char(pauli) == expected


def test_build_coupling_matrix_empty() -> None:
    assert _build_coupling_matrix([], 2, set(_ALLOWED_ISINGLIKE_TERMS)) == {}


def test_build_coupling_matrix_same_pauli_one_term() -> None:
    PTI = _PauliTermInfo
    PSTWI = _PauliStringTwoWeightInfo
    two_weights = [PSTWI(np.complex128(1.0, 2.0), PTI(0, X), PTI(1, X))]
    output = _build_coupling_matrix(two_weights, 2, set(_ALLOWED_ISINGLIKE_TERMS))

    assert "XX" in output
    assert np.allclose(
        output["XX"],
        0.5 * np.array([[0.0, 1.0 + 2.0j], [1.0 + 2.0j, 0.0]], dtype=np.complex128),
    )


def test_build_coupling_matrix_same_pauli_three_term() -> None:
    PTI = _PauliTermInfo
    PSTWI = _PauliStringTwoWeightInfo
    two_weights = [
        PSTWI(np.complex128(1.0, 2.0), PTI(0, X), PTI(1, X)),
        PSTWI(np.complex128(3.0, 4.0), PTI(1, X), PTI(2, X)),
        PSTWI(np.complex128(5.0, 6.0), PTI(0, X), PTI(2, X)),
    ]
    output = _build_coupling_matrix(two_weights, 3, set(_ALLOWED_ISINGLIKE_TERMS))

    assert "XX" in output
    assert np.allclose(
        output["XX"],
        0.5
        * np.array(
            [
                [0.0 + 0.0j, 1.0 + 2.0j, 5.0 + 6.0j],
                [1.0 + 2.0j, 0.0 + 0.0j, 3.0 + 4.0j],
                [5.0 + 6.0j, 3.0 + 4.0j, 0.0 + 0.0j],
            ],
            dtype=np.complex128,
        ),
    )


class Test_isinglike_analysis:
    def test_xx_2_qubits(self) -> None:
        gate = AnalogGate(hamiltonian=X @ X)
        output = isinglike_analysis(gate)

        assert "XX" in output
        assert np.allclose(output["XX"], 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]]))

    def test_xx_minus_yy_2_qubits(self) -> None:
        gate = AnalogGate(hamiltonian=X @ X - Y @ Y)
        output = isinglike_analysis(gate)

        assert "XX" in output
        assert np.allclose(output["XX"], 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]]))
        assert "YY" in output
        assert np.allclose(output["YY"], 0.5 * np.array([[0.0, -1.0], [-1.0, 0.0]]))

    def test_xx_3_qubits(self) -> None:
        gate = AnalogGate(hamiltonian=X @ X @ ID)
        output = isinglike_analysis(gate)

        assert "XX" in output
        assert np.allclose(
            output["XX"],
            0.5 * np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        )

    def test_xy_2_qubits(self) -> None:
        gate = AnalogGate(hamiltonian=2.0 * X @ Y)
        output = isinglike_analysis(gate)

        assert "XY" in output
        assert np.allclose(output["XY"], np.array([[0.0, 2.0], [0.0, 0.0]]))

    def test_yx_2_qubits(self) -> None:
        gate = AnalogGate(hamiltonian=2.0 * Y @ X)
        output = isinglike_analysis(gate)

        assert "XY" in output
        assert np.allclose(output["XY"], np.array([[0.0, 0.0], [2.0, 0.0]]))

    def test_yx_zz_2_qubits(self) -> None:
        gate = AnalogGate(hamiltonian=2.0 * Y @ X + 3.0 * Z @ Z)
        output = isinglike_analysis(gate, allowed_terms=["XY", "ZZ"])

        assert "XY" in output
        assert np.allclose(output["XY"], np.array([[0.0, 0.0], [2.0, 0.0]]))

        assert "ZZ" in output
        assert np.allclose(output["ZZ"], 0.5 * np.array([[0.0, 3.0], [3.0, 0.0]]))

    def test_multiterm_4_qubits(self) -> None:
        """
        This test makes sure that it is possible to get all 6 kinds of terms
        in the output, and that large values can be handled.
        """
        # fmt: off
        gate = AnalogGate(
            hamiltonian=1.0 * X @ X @ ID @ ID
                      + 2.0 * X @ ID @ X @ ID
                      + 3.0 * ID @ ID @ X @ X
                      + 4.0 * Y @ ID @ X @ ID
                      - 5.0 * Z @ ID @ X @ ID
                      - 6.0 * Z @ Y @ ID @ ID
                      - 7.0 * Y @ Z @ ID @ ID
                      - 8.0 * ID @ ID @ Y @ Y
                      - 9.0 * ID @ ID @ Z @ Z
        )
        # fmt: on

        output = isinglike_analysis(gate)

        assert "XX" in output
        assert np.allclose(
            output["XX"],
            0.5
            * np.array(
                [
                    [0.0, 1.0, 2.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 3.0],
                    [0.0, 0.0, 3.0, 0.0],
                ]
            ),
        )

        assert "XY" in output
        assert np.allclose(
            output["XY"],
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

        assert "XZ" in output
        assert np.allclose(
            output["XZ"],
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [-5.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

        assert "YY" in output
        assert np.allclose(
            output["YY"],
            0.5
            * np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -8.0],
                    [0.0, 0.0, -8.0, 0.0],
                ]
            ),
        )

        assert "YZ" in output
        assert np.allclose(
            output["YZ"],
            np.array(
                [
                    [0.0, -7.0, 0.0, 0.0],
                    [-6.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

        assert "ZZ" in output
        assert np.allclose(
            output["ZZ"],
            0.5
            * np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -9.0],
                    [0.0, 0.0, -9.0, 0.0],
                ]
            ),
        )

    def test_raises_bosonic_operator(self) -> None:
        gate = AnalogGate(hamiltonian=1.0 * X @ C)

        with pytest.raises(RuntimeError):
            isinglike_analysis(gate)

    def test_raises_time_dependent_operator(self) -> None:
        gate = AnalogGate(hamiltonian=MathVar(name="t") * X @ Y)

        with pytest.raises(RuntimeError):
            isinglike_analysis(gate)

    def test_raises_invalid_allowed_term(self) -> None:
        gate = AnalogGate(hamiltonian=2.0 * X @ Y)

        with pytest.raises(RuntimeError):
            isinglike_analysis(gate, allowed_terms=["ABCD"])

    def test_raises_unallowed_term_found(self) -> None:
        gate = AnalogGate(hamiltonian=2.0 * X @ Y + 1.0 * X @ X)

        with pytest.raises(RuntimeError):
            isinglike_analysis(gate, allowed_terms=["XX"])

    def test_raises_unallowed_term_found(self) -> None:
        gate = AnalogGate(hamiltonian=2.0 * X @ Y + 1.0 * X @ X + 3.0 * Z @ Y)

        with pytest.raises(RuntimeError):
            isinglike_analysis(gate, allowed_terms=["XX", "XY"])

    @pytest.mark.parametrize(
        "op",
        [
            1.0 * X @ ID,
            1.0 * X @ Y + 2.0 * X @ ID,
            1.0 * ID @ X @ Y + 2.0 * X @ ID @ ID,
            1.0 * ID @ X @ Y + 2.0 * X @ Z @ ID + 3.0 * X @ Z @ Y,
        ],
    )
    def test_raises_pauli_string_weight(self, op: OperatorSubtypes) -> None:
        gate = AnalogGate(hamiltonian=op)

        with pytest.raises(RuntimeError):
            isinglike_analysis(gate)

    @pytest.mark.parametrize(
        "op",
        [
            1.0 * X @ Y + 2.0 * X @ Z @ ID,
            1.0 * X @ Y + 1.0 * Z @ Y + 2.0 * X @ Z @ ID,
            1.0 * X @ Y + 1.0 * Z @ ID @ Y + 2.0 * X @ Z @ ID,
            1.0 * X @ Y + 1.0 * Z @ ID @ Y + 2.0 * X @ Z @ ID @ ID @ ID,
        ],
    )
    def test_raises_different_length_paulis(self, op: OperatorSubtypes) -> None:
        gate = AnalogGate(hamiltonian=op)

        with pytest.raises(RuntimeError):
            isinglike_analysis(gate)
