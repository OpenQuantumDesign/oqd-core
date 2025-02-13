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

from __future__ import annotations

from typing import List, Tuple, Union

from oqd_compiler_infrastructure import TypeReflectBaseModel
from pydantic import conlist

########################################################################################
from oqd_core.interface.atomic.system import Transition
from oqd_core.interface.math import CastMathExpr

########################################################################################

__all__ = [
    "Beam",
    "Pulse",
    "Protocol",
    "ParallelProtocol",
    "SequentialProtocol",
]

########################################################################################


class Beam(TypeReflectBaseModel):
    """
    Class representing a referenced optical channel/beam for the trapped-ion device.

    Attributes:
        transition: Label for the Transition that the optical channel/beam is referenced to.
        rabi: Rabi frequency of the referenced transition driven by the beam.
        detuning: Detuning away from the referenced transition.
        phase: Phase relative to the ion's clock.
        polarization: Polarization of the beam.
        wavevector: Wavevector of the beam.
        target: Index of the target ion of the beam.
    """

    transition: Union[str, Tuple[str, int], Transition]
    rabi: CastMathExpr
    detuning: CastMathExpr
    phase: CastMathExpr
    polarization: conlist(float, max_length=3, min_length=3)
    wavevector: conlist(float, max_length=3, min_length=3)
    target: int


class Pulse(TypeReflectBaseModel):
    """
    Class representing the application of the beam for some duration.

    Attributes:
        beam: Optical channel/beam to turn on.
        duration: Period of time to turn the optical channel on for.

    """

    beam: Beam
    duration: float


class Protocol(TypeReflectBaseModel):
    """
    Class representing a light-matter interaction protocol/pulse program for the optical channels/beams.
    """

    pass


class ParallelProtocol(Protocol):
    """
    Class representing the parallel composition of a list of pulses or subprotocols.

    Attributes:
        sequence: List of pulses or subprotocols to compose together in a parallel fashion.
    """

    sequence: List[Union[Pulse, ProtocolSubTypes]]


class SequentialProtocol(Protocol):
    """
    Class representing the sequential composition of a list of pulses or subprotocols.

    Attributes:
        sequence: List of pulses or subprotocols to compose together in a sequntial fashion.
    """

    sequence: List[Union[Pulse, ProtocolSubTypes]]


ProtocolSubTypes = Union[SequentialProtocol, ParallelProtocol]
