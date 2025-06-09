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

from typing import Dict, List, Set, Tuple
import numpy as np

from oqd_compiler_infrastructure import ConversionRule, Post
from oqd_core.interface.analog.operation import AnalogGate
from oqd_core.interface.analog.operator import (
    Operator,
    OperatorAdd,
    OperatorSub, 
    OperatorKron,
    OperatorScalarMul,
    PauliI,
    PauliX, 
    PauliY,
    PauliZ,
    Creation,
    Annihilation,
    Identity,
    Ladder
)
from oqd_core.interface.math import MathExpr, MathNum, MathImag, MathMul

from .ising_types import (
    PauliString,
    IsingCouplingMatrices,
    IsingAnalysisResult,
    IsingValidationError
)

__all__ = [
    "IsingAnalysisPass",
    "analyze_ising_gate"
]


class IsingAnalysisPass(ConversionRule):
    """
    Compiler pass to analyze if an AnalogGate implements an Ising-like Hamiltonian.
    
    This pass walks the Hamiltonian operator tree and:
    1. Validates Ising-like properties (qubit-only, weight-2 Pauli strings, time-independent)
    2. Extracts coupling matrices for different Pauli operator pairs
    3. Returns structured analysis results
    """

    def __init__(self):
        super().__init__()
        self.pauli_strings: List[PauliString] = []
        self.errors: List[IsingValidationError] = []
        self.qubit_indices: Set[int] = set()
        self.current_coefficient = 1.0
        self.current_pauli_ops: List[Tuple[int, str]] = []
        self.current_qubit_index = 0

    def map_AnalogGate(self, model: AnalogGate, operands: Dict) -> IsingAnalysisResult:
        """Analyze an AnalogGate for Ising-like properties."""
        # Reset state
        self.pauli_strings = []
        self.errors = []
        self.qubit_indices = set()
        
        # Walk the Hamiltonian
        self._analyze_operator(model.hamiltonian)
        
        # Determine if Ising-like
        is_ising_like = len(self.errors) == 0 and self._validate_ising_properties()
        
        # Extract coupling matrices if valid
        coupling_matrices = None
        n_qubits = max(self.qubit_indices) + 1 if self.qubit_indices else 0
        
        if is_ising_like and n_qubits > 0:
            coupling_matrices = self._extract_coupling_matrices(n_qubits)
        
        return IsingAnalysisResult(
            is_ising_like=is_ising_like,
            coupling_matrices=coupling_matrices,
            pauli_strings=self.pauli_strings,
            errors=self.errors,
            n_qubits=n_qubits
        )

    def _analyze_operator(self, op: Operator, coefficient: complex = 1.0):
        """Recursively analyze an operator."""
        if isinstance(op, OperatorAdd):
            self._analyze_operator(op.op1, coefficient)
            self._analyze_operator(op.op2, coefficient)
            
        elif isinstance(op, OperatorSub):
            self._analyze_operator(op.op1, coefficient)
            self._analyze_operator(op.op2, -coefficient)
            
        elif isinstance(op, OperatorScalarMul):
            # Extract coefficient and check if time-independent
            scalar_coeff = self._extract_coefficient(op.expr)
            if scalar_coeff is None:
                self.errors.append(IsingValidationError(
                    error_type="time_dependent_coefficient",
                    message="Coefficient contains time-dependent terms",
                    operator=op
                ))
                return
            self._analyze_operator(op.op, coefficient * scalar_coeff)
            
        elif isinstance(op, OperatorKron):
            # Analyze Kronecker product (tensor product)
            self._analyze_kronecker_product(op, coefficient)
            
        elif isinstance(op, (PauliI, PauliX, PauliY, PauliZ)):
            # Single Pauli operator
            pauli_type = self._get_pauli_type(op)
            pauli_string = PauliString(
                pauli_ops=[(0, pauli_type)],
                coefficient=coefficient
            )
            self.pauli_strings.append(pauli_string)
            self.qubit_indices.add(0)
            
        elif isinstance(op, (Creation, Annihilation, Identity, Ladder)):
            # Bosonic operators - not allowed in Ising Hamiltonians
            self.errors.append(IsingValidationError(
                error_type="bosonic_operator",
                message=f"Found bosonic operator {type(op).__name__} - not allowed in Ising Hamiltonians",
                operator=op
            ))

    def _analyze_kronecker_product(self, op: OperatorKron, coefficient: complex):
        """Analyze a Kronecker product to extract Pauli string."""
        pauli_ops = []
        qubit_index = 0
        
        # Flatten the Kronecker product and extract Pauli operators
        ops_to_process = [op.op1, op.op2]
        
        for sub_op in ops_to_process:
            if isinstance(sub_op, (PauliI, PauliX, PauliY, PauliZ)):
                pauli_type = self._get_pauli_type(sub_op)
                pauli_ops.append((qubit_index, pauli_type))
                self.qubit_indices.add(qubit_index)
                qubit_index += 1
            elif isinstance(sub_op, OperatorKron):
                # Nested Kronecker product - recursively flatten
                self._flatten_kronecker(sub_op, pauli_ops, qubit_index)
                qubit_index = len(pauli_ops)
            elif isinstance(sub_op, (Creation, Annihilation, Identity, Ladder)):
                self.errors.append(IsingValidationError(
                    error_type="bosonic_operator", 
                    message=f"Found bosonic operator {type(sub_op).__name__} in Kronecker product",
                    operator=sub_op
                ))
                return
            else:
                self.errors.append(IsingValidationError(
                    error_type="unsupported_operator",
                    message=f"Unsupported operator type {type(sub_op).__name__} in Kronecker product",
                    operator=sub_op
                ))
                return
        
        if pauli_ops:
            pauli_string = PauliString(
                pauli_ops=pauli_ops,
                coefficient=coefficient
            )
            self.pauli_strings.append(pauli_string)

    def _flatten_kronecker(self, op: OperatorKron, pauli_ops: List[Tuple[int, str]], start_index: int):
        """Recursively flatten nested Kronecker products."""
        # This is a simplified version - full implementation would handle all nesting levels
        if isinstance(op.op1, (PauliI, PauliX, PauliY, PauliZ)):
            pauli_type = self._get_pauli_type(op.op1)
            pauli_ops.append((start_index, pauli_type))
            self.qubit_indices.add(start_index)
            
        if isinstance(op.op2, (PauliI, PauliX, PauliY, PauliZ)):
            pauli_type = self._get_pauli_type(op.op2)
            pauli_ops.append((start_index + 1, pauli_type))
            self.qubit_indices.add(start_index + 1)

    def _get_pauli_type(self, op) -> str:
        """Get the Pauli type as a string."""
        if isinstance(op, PauliI):
            return 'I'
        elif isinstance(op, PauliX):
            return 'X'
        elif isinstance(op, PauliY):
            return 'Y'
        elif isinstance(op, PauliZ):
            return 'Z'
        else:
            raise ValueError(f"Unknown Pauli operator: {type(op)}")

    def _extract_coefficient(self, expr: MathExpr) -> complex:
        """Extract coefficient from MathExpr and check if time-independent."""
        if isinstance(expr, MathNum):
            return complex(expr.value)
        elif isinstance(expr, MathImag):
            return 1j
        elif isinstance(expr, MathMul):
            # For simplicity, assume multiplication of numbers and imaginary unit
            left = self._extract_coefficient(expr.expr1)
            right = self._extract_coefficient(expr.expr2)
            if left is not None and right is not None:
                return left * right
        
        # If we can't extract a simple coefficient, assume time-dependent
        return None

    def _validate_ising_properties(self) -> bool:
        """Validate that all Pauli strings satisfy Ising properties."""
        for pauli_string in self.pauli_strings:
            # Check weight-2 constraint
            if pauli_string.weight > 2:
                self.errors.append(IsingValidationError(
                    error_type="high_weight_pauli",
                    message=f"Found Pauli string with weight {pauli_string.weight} > 2"
                ))
                return False
                
            # Check time-independence
            if not pauli_string.is_time_independent:
                self.errors.append(IsingValidationError(
                    error_type="time_dependent_coefficient",
                    message="Found time-dependent coefficient"
                ))
                return False
        
        return True

    def _extract_coupling_matrices(self, n_qubits: int) -> IsingCouplingMatrices:
        """Extract coupling matrices from validated Pauli strings."""
        matrices = IsingCouplingMatrices.zeros(n_qubits)
        
        for pauli_string in self.pauli_strings:
            if pauli_string.weight == 2:
                # Get the two non-identity qubits and their Pauli types
                non_identity = [(idx, pauli) for idx, pauli in pauli_string.pauli_ops if pauli != 'I']
                if len(non_identity) == 2:
                    (i, pauli_i), (j, pauli_j) = non_identity
                    coefficient = pauli_string.coefficient.real  # Should be real for Ising
                    
                    # Determine interaction type and update corresponding matrix
                    interaction_type = ''.join(sorted([pauli_i, pauli_j]))
                    
                    if interaction_type == 'XX':
                        matrices.XX[i, j] = matrices.XX[j, i] = coefficient
                    elif interaction_type == 'YY':
                        matrices.YY[i, j] = matrices.YY[j, i] = coefficient
                    elif interaction_type == 'ZZ':
                        matrices.ZZ[i, j] = matrices.ZZ[j, i] = coefficient
                    elif interaction_type == 'XY':
                        matrices.XY[i, j] = matrices.XY[j, i] = coefficient
                    elif interaction_type == 'XZ':
                        matrices.XZ[i, j] = matrices.XZ[j, i] = coefficient
                    elif interaction_type == 'YZ':
                        matrices.YZ[i, j] = matrices.YZ[j, i] = coefficient
        
        return matrices


def analyze_ising_gate(gate: AnalogGate) -> IsingAnalysisResult:
    """
    Convenience function to analyze an AnalogGate for Ising-like properties.
    
    Args:
        gate: AnalogGate to analyze
        
    Returns:
        IsingAnalysisResult containing analysis results and coupling matrices
    """
    analysis_pass = Post(IsingAnalysisPass())
    return analysis_pass(gate)
