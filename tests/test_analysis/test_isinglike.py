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

from oqd_core.analysis.isinglike import isinglike_analysis
from oqd_core.analysis.isinglike import _get_pauli_string_two_weight_info
from oqd_core.analysis.isinglike import _rescale_all_coefficients
from oqd_core.analysis.isinglike import _PauliStringTerm
from oqd_core.analysis.isinglike import _PauliTermInfo
from oqd_core.analysis.isinglike import _PauliStringTwoWeightInfo


ID = PauliI()
X = PauliX()
Y = PauliY()
Z = PauliZ()


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
        # fmt: off
        (1, (np.complex128(1.0, 0.0), np.complex128(2.0, 0.0), np.complex128(3.0, 1.0))),
        (2, (np.complex128(2.0, 0.0), np.complex128(4.0, 0.0), np.complex128(6.0, 2.0))),
        (3.0, (np.complex128(3.0, 0.0), np.complex128(6.0, 0.0), np.complex128(9.0, 3.0))),
        (np.complex128(2.0, 3.0), (np.complex128(2.0, 3.0), np.complex128(4.0, 6.0), np.complex128(3.0, 11.0))),
        # fmt: on
    ],
)
def test_rescale_all_coefficients(
    value: int | float | np.complex128,
    expected: tuple[np.complex128],
    default_pauli_strings: list[_PauliStringTerm]
) -> None:
    rescaled = _rescale_all_coefficients(default_pauli_strings, value)

    assert np.allclose(rescaled[0].coefficient, expected[0])
    assert np.allclose(rescaled[1].coefficient, expected[1])
    assert np.allclose(rescaled[2].coefficient, expected[2])


@pytest.mark.parametrize(
    "pauli_string, expected",
    [
        # fmt: off
        (
            _PauliStringTerm(1.0, [X, Y]),
            _PauliStringTwoWeightInfo(np.complex128(1.0, 0.0), _PauliTermInfo(0, X), _PauliTermInfo(1, Y))
        ),
        (
            _PauliStringTerm(2.0, [X, Y]),
            _PauliStringTwoWeightInfo(np.complex128(2.0, 0.0), _PauliTermInfo(0, X), _PauliTermInfo(1, Y))
        ),
        (
            _PauliStringTerm(1.0, [X, Z]),
            _PauliStringTwoWeightInfo(np.complex128(1.0, 0.0), _PauliTermInfo(0, X), _PauliTermInfo(1, Z))
        ),
        (
            _PauliStringTerm(1.0, [X, ID, Z]),
            _PauliStringTwoWeightInfo(np.complex128(1.0, 0.0), _PauliTermInfo(0, X), _PauliTermInfo(2, Z))
        ),
        (
            _PauliStringTerm(1.0, [ID, ID, Y, ID, X, ID, ID, ID]),
            _PauliStringTwoWeightInfo(np.complex128(1.0, 0.0), _PauliTermInfo(2, Y), _PauliTermInfo(4, X))
        ),
        (
            _PauliStringTerm(1.0, [X]),
            None
        ),
        (
            _PauliStringTerm(1.0, [X, ID]),
            None
        ),
        (
            _PauliStringTerm(1.0, [ID, X, ID]),
            None
        ),
        (
            _PauliStringTerm(1.0, [X, Y, Z]),
            None
        ),
        (
            _PauliStringTerm(1.0, [ID, X, ID, Y, Z]),
            None
        ),
        # fmt: on
    ],
)
def test_get_pauli_string_two_weight_info(
    pauli_string: _PauliStringTerm,
    expected: Optional[_PauliStringTwoWeightInfo]
) -> None:
    actual = _get_pauli_string_two_weight_info(pauli_string)
    
    if actual is None:
        assert actual is expected
    else:
        assert actual == expected
