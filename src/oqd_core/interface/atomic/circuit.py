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


from oqd_compiler_infrastructure import TypeReflectBaseModel

########################################################################################

from oqd_core.interface.atomic.system import System
from oqd_core.interface.atomic.protocol import Protocol

########################################################################################

__all__ = [
    "AtomicCircuit",
]

########################################################################################


class AtomicCircuit(TypeReflectBaseModel):
    """
    Class representing a trapped-ion experiment in terms of light-matter interactons.

    Attributes:
        system: The trapped-ion system.
        protocol: Pulse program for the trapped-ion experiment referenced to the trapped-ion system.

    """

    system: System
    protocol: Protocol
