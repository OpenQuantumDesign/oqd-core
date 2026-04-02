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

from typing import List
from oqd_compiler_infrastructure import TypeReflectBaseModel
from .statement import Statement, ParallelProtocol
from .expr import Pulse, Beam
########################################################################################

__all__ = [
    "AtomicCircuit",
]

########################################################################################


class AtomicCircuit(TypeReflectBaseModel):
    """
    Class representing a trapped-ion experiment in terms of light-matter interactons.

    Attributes:
        statements: The trapped-ion system.
    """

    statements: List[Statement] = []
    
    def beam(self, frequency, rabi, phase, polarization, wavevector):
        self.statements.append(
            Beam(frequency = frequency, rabi = rabi, phase = phase, polarization = polarization, wavevector = wavevector)
        )
        
    def pulse(self, duration, target, beam, measured):
        self.statements.append(
            Pulse(duration=duration, target=target, beam=beam, measured=measured)
        )
    
    def parallel(self, pulses):
        self.statements.append(
            ParallelProtocol(pulses=pulses)
        )
        
