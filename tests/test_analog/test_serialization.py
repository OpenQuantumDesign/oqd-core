# Copyright 2024 Open Quantum Design

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

import numpy as np

from oqd_core.interface.analog import (
    Annihilation,
    Creation,
    Identity,
    PauliI,
    PauliX,
    PauliY,
    PauliZ,
    AnalogCircuit,
    AnalogGate,
)

X, Y, Z, PI, A, C, LI = (
    PauliX(),
    PauliY(),
    PauliZ(),
    PauliI(),
    Annihilation(),
    Creation(),
    Identity(),
)


@pytest.mark.parametrize(
    "op",
    [
        X,
        Y @ Y @ X,
        X + Y + Z + PI,
        A + C + LI,
    ],
)
def test_serialize_deserialize_op(op):
    assert op.model_validate(op.model_dump()) == op
    assert op.model_validate_json(op.model_dump_json()) == op


def test_serialize_deserialize_analog_circuit():
    Hx = AnalogGate(hamiltonian=-(np.pi / 4) * X)
    ac = AnalogCircuit()
    ac.evolve(duration=1, gate=Hx)
    ac.evolve(duration=1, gate=Hx)
    ac.evolve(duration=1, gate=Hx)
    ac.measure()

    assert ac.model_validate(ac.model_dump()) == ac
    assert ac.model_validate_json(ac.model_dump_json()) == ac
