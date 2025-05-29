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

from typing import Dict, List, Optional, Tuple
import numpy as np
from oqd_compiler_infrastructure import TypeReflectBaseModel
from oqd_core.interface.analog.operator import OperatorSubtypes

__all__ = [
    "PauliString",
    "IsingCouplingMatrices", 
    "IsingAnalysisResult",
    "IsingValidationError"
]


class PauliString(TypeReflectBaseModel):
    """
    Represents a Pauli string with its coefficient and qubit positions.
    
    Attributes:
        pauli_ops: List of (qubit_index, pauli_type) tuples
        coefficient: Complex coefficient (must be real and time-independent for Ising)
        is_time_independent: Whether the coefficient is time-independent
    """
    pauli_ops: List[Tuple[int, str]]  # (qubit_index, 'I'|'X'|'Y'|'Z')
    coefficient: complex
    is_time_independent: bool = True

    @property
    def weight(self) -> int:
        """Return the weight (number of non-identity Paulis) of this string."""
        return len([op for op in self.pauli_ops if op[1] != 'I'])

    @property
    def is_two_qubit(self) -> bool:
        """Check if this is a two-qubit Pauli string."""
        return self.weight == 2

    def get_pauli_type_pair(self) -> Optional[str]:
        """Get the Pauli type pair for two-qubit terms (e.g., 'XX', 'XZ')."""
        if not self.is_two_qubit:
            return None
        non_identity = [op[1] for op in self.pauli_ops if op[1] != 'I']
        return ''.join(sorted(non_identity))


class IsingCouplingMatrices(TypeReflectBaseModel):
    """
    Container for Ising coupling matrices between different Pauli operator pairs.
    
    Attributes:
        XX: Coupling matrix for XX interactions
        YY: Coupling matrix for YY interactions  
        ZZ: Coupling matrix for ZZ interactions
        XY: Coupling matrix for XY interactions
        XZ: Coupling matrix for XZ interactions
        YZ: Coupling matrix for YZ interactions
        n_qubits: Number of qubits in the system
    """
    XX: np.ndarray
    YY: np.ndarray
    ZZ: np.ndarray
    XY: np.ndarray
    XZ: np.ndarray
    YZ: np.ndarray
    n_qubits: int

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def zeros(cls, n_qubits: int) -> IsingCouplingMatrices:
        """Create zero coupling matrices for n qubits."""
        zero_matrix = np.zeros((n_qubits, n_qubits), dtype=complex)
        return cls(
            XX=zero_matrix.copy(),
            YY=zero_matrix.copy(),
            ZZ=zero_matrix.copy(),
            XY=zero_matrix.copy(),
            XZ=zero_matrix.copy(),
            YZ=zero_matrix.copy(),
            n_qubits=n_qubits
        )


class IsingValidationError(TypeReflectBaseModel):
    """
    Represents a validation error when checking Ising-like properties.
    
    Attributes:
        error_type: Type of validation error
        message: Human-readable error message
        operator: The problematic operator (optional)
    """
    error_type: str
    message: str
    operator: Optional[OperatorSubtypes] = None


class IsingAnalysisResult(TypeReflectBaseModel):
    """
    Result of Ising analysis on an AnalogGate.
    
    Attributes:
        is_ising_like: Whether the Hamiltonian satisfies Ising-like properties
        coupling_matrices: Coupling matrices for different Pauli pairs (if valid)
        pauli_strings: List of extracted Pauli strings
        errors: List of validation errors (if invalid)
        n_qubits: Number of qubits detected in the Hamiltonian
    """
    is_ising_like: bool
    coupling_matrices: Optional[IsingCouplingMatrices] = None
    pauli_strings: List[PauliString] = []
    errors: List[IsingValidationError] = []
    n_qubits: int = 0
