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

from typing import List

from oqd_compiler_infrastructure import TypeReflectBaseModel

from .statement import Evolve, Initialize, Measure, Statement

########################################################################################

__all__ = ["AnalogCircuit"]

########################################################################################


class AnalogCircuit(TypeReflectBaseModel):
    """
    Class representing a quantum information experiment represented in terms of analog operations.

    Attributes:
        statements (List[Union[Measure, Evolve, Initialize]]): List of statements, including initialize, evolve, measure

    """

    statements: List[Statement] = []

    def evolve(self, hamiltonian, duration, targets):
        self.statements.append(
            Evolve(hamiltonian=hamiltonian, duration=duration, targets=targets)
        )

    def initialize(self, targets):
        self.statements.append(Initialize(targets=targets))

    def measure(self, targets):
        self.statements.append(Measure(targets=targets))
