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
import numpy as np

from oqd_core.interface.analog import (
    AnalogGate,
    PauliI,
    PauliX,
    PauliY, 
    PauliZ,
    OperatorAdd,
    OperatorKron,
    OperatorScalarMul,
    Creation,
    Annihilation
)
from oqd_core.interface.math import MathNum
from oqd_core.compiler.analog.passes import analyze_ising_gate
from oqd_core.compiler.analog.passes.ising_types import IsingAnalysisResult


class TestIsingAnalysis:
    """Test suite for Ising Hamiltonian analysis."""

    def test_simple_xx_interaction(self):
        """Test analysis of simple XX interaction Hamiltonian."""
        # H = X ⊗ X
        hamiltonian = OperatorKron(op1=PauliX(), op2=PauliX())
        gate = AnalogGate(hamiltonian=hamiltonian)
        
        result = analyze_ising_gate(gate)
        
        assert result.is_ising_like
        assert result.n_qubits == 2
        assert result.coupling_matrices is not None
        
        # Check XX coupling matrix
        expected_xx = np.zeros((2, 2))
        expected_xx[0, 1] = expected_xx[1, 0] = 1.0
        np.testing.assert_array_equal(result.coupling_matrices.XX, expected_xx)
        
        # Other matrices should be zero
        np.testing.assert_array_equal(result.coupling_matrices.YY, np.zeros((2, 2)))
        np.testing.assert_array_equal(result.coupling_matrices.ZZ, np.zeros((2, 2)))

    def test_scaled_xx_interaction(self):
        """Test XX interaction with scalar coefficient."""
        # H = 2.0 * (X ⊗ X)
        hamiltonian = OperatorScalarMul(
            op=OperatorKron(op1=PauliX(), op2=PauliX()),
            expr=MathNum(value=2.0)
        )
        gate = AnalogGate(hamiltonian=hamiltonian)
        
        result = analyze_ising_gate(gate)
        
        assert result.is_ising_like
        assert result.coupling_matrices is not None
        
        expected_xx = np.zeros((2, 2))
        expected_xx[0, 1] = expected_xx[1, 0] = 2.0
        np.testing.assert_array_equal(result.coupling_matrices.XX, expected_xx)

    def test_multiple_interactions(self):
        """Test Hamiltonian with multiple interaction types."""
        # H = X ⊗ X + Y ⊗ Y + Z ⊗ Z
        xx_term = OperatorKron(op1=PauliX(), op2=PauliX())
        yy_term = OperatorKron(op1=PauliY(), op2=PauliY())
        zz_term = OperatorKron(op1=PauliZ(), op2=PauliZ())
        
        hamiltonian = OperatorAdd(
            op1=OperatorAdd(op1=xx_term, op2=yy_term),
            op2=zz_term
        )
        gate = AnalogGate(hamiltonian=hamiltonian)
        
        result = analyze_ising_gate(gate)
        
        assert result.is_ising_like
        assert result.coupling_matrices is not None
        
        # Each interaction type should have coefficient 1.0
        expected_matrix = np.zeros((2, 2))
        expected_matrix[0, 1] = expected_matrix[1, 0] = 1.0
        
        np.testing.assert_array_equal(result.coupling_matrices.XX, expected_matrix)
        np.testing.assert_array_equal(result.coupling_matrices.YY, expected_matrix)
        np.testing.assert_array_equal(result.coupling_matrices.ZZ, expected_matrix)

    def test_mixed_xy_interaction(self):
        """Test mixed XY interaction."""
        # H = X ⊗ Y
        hamiltonian = OperatorKron(op1=PauliX(), op2=PauliY())
        gate = AnalogGate(hamiltonian=hamiltonian)
        
        result = analyze_ising_gate(gate)
        
        assert result.is_ising_like
        assert result.coupling_matrices is not None
        
        expected_xy = np.zeros((2, 2))
        expected_xy[0, 1] = expected_xy[1, 0] = 1.0
        np.testing.assert_array_equal(result.coupling_matrices.XY, expected_xy)

    def test_single_qubit_terms(self):
        """Test Hamiltonian with single-qubit terms (weight-1)."""
        # H = X (single qubit, should be valid)
        hamiltonian = PauliX()
        gate = AnalogGate(hamiltonian=hamiltonian)
        
        result = analyze_ising_gate(gate)
        
        assert result.is_ising_like
        assert len(result.pauli_strings) == 1
        assert result.pauli_strings[0].weight == 1

    def test_identity_terms(self):
        """Test Hamiltonian with identity terms."""
        # H = I ⊗ I (should be weight-0, valid)
        hamiltonian = OperatorKron(op1=PauliI(), op2=PauliI())
        gate = AnalogGate(hamiltonian=hamiltonian)
        
        result = analyze_ising_gate(gate)
        
        assert result.is_ising_like
        assert len(result.pauli_strings) == 1
        assert result.pauli_strings[0].weight == 0

    def test_high_weight_pauli_invalid(self):
        """Test that high-weight Pauli strings are rejected."""
        # H = X ⊗ X ⊗ X (weight-3, should be invalid)
        hamiltonian = OperatorKron(
            op1=OperatorKron(op1=PauliX(), op2=PauliX()),
            op2=PauliX()
        )
        gate = AnalogGate(hamiltonian=hamiltonian)
        
        result = analyze_ising_gate(gate)
        
        assert not result.is_ising_like
        assert len(result.errors) > 0
        assert any(error.error_type == "high_weight_pauli" for error in result.errors)

    def test_bosonic_operators_invalid(self):
        """Test that bosonic operators are rejected."""
        # H = a† (creation operator, should be invalid)
        hamiltonian = Creation()
        gate = AnalogGate(hamiltonian=hamiltonian)
        
        result = analyze_ising_gate(gate)
        
        assert not result.is_ising_like
        assert len(result.errors) > 0
        assert any(error.error_type == "bosonic_operator" for error in result.errors)

    def test_mixed_qubit_bosonic_invalid(self):
        """Test mixing qubit and bosonic operators."""
        # H = X ⊗ a† (mixed qubit-bosonic, should be invalid)
        hamiltonian = OperatorKron(op1=PauliX(), op2=Creation())
        gate = AnalogGate(hamiltonian=hamiltonian)
        
        result = analyze_ising_gate(gate)
        
        assert not result.is_ising_like
        assert len(result.errors) > 0
        assert any(error.error_type == "bosonic_operator" for error in result.errors)

    def test_complex_ising_hamiltonian(self):
        """Test a more complex but valid Ising Hamiltonian."""
        # H = 0.5 * (X ⊗ X) + 0.3 * (Y ⊗ Y) - 0.1 * (Z ⊗ Z) + 0.2 * (X ⊗ Y)
        xx_term = OperatorScalarMul(
            op=OperatorKron(op1=PauliX(), op2=PauliX()),
            expr=MathNum(value=0.5)
        )
        yy_term = OperatorScalarMul(
            op=OperatorKron(op1=PauliY(), op2=PauliY()),
            expr=MathNum(value=0.3)
        )
        zz_term = OperatorScalarMul(
            op=OperatorKron(op1=PauliZ(), op2=PauliZ()),
            expr=MathNum(value=-0.1)
        )
        xy_term = OperatorScalarMul(
            op=OperatorKron(op1=PauliX(), op2=PauliY()),
            expr=MathNum(value=0.2)
        )
        
        hamiltonian = OperatorAdd(
            op1=OperatorAdd(
                op1=OperatorAdd(op1=xx_term, op2=yy_term),
                op2=zz_term
            ),
            op2=xy_term
        )
        gate = AnalogGate(hamiltonian=hamiltonian)
        
        result = analyze_ising_gate(gate)
        
        assert result.is_ising_like
        assert result.coupling_matrices is not None
        
        # Check coupling strengths
        assert np.isclose(result.coupling_matrices.XX[0, 1], 0.5)
        assert np.isclose(result.coupling_matrices.YY[0, 1], 0.3)
        assert np.isclose(result.coupling_matrices.ZZ[0, 1], -0.1)
        assert np.isclose(result.coupling_matrices.XY[0, 1], 0.2)

    def test_empty_hamiltonian(self):
        """Test behavior with an empty/minimal Hamiltonian."""
        # H = I (just identity)
        hamiltonian = PauliI()
        gate = AnalogGate(hamiltonian=hamiltonian)
        
        result = analyze_ising_gate(gate)
        
        assert result.is_ising_like
        assert result.n_qubits == 1
        assert len(result.pauli_strings) == 1
        assert result.pauli_strings[0].weight == 0

    def test_pauli_string_properties(self):
        """Test PauliString property methods."""
        from oqd_core.compiler.analog.passes.ising_types import PauliString
        
        # Test weight calculation
        pauli_string = PauliString(
            pauli_ops=[(0, 'X'), (1, 'Y'), (2, 'I')],
            coefficient=1.0
        )
        assert pauli_string.weight == 2
        assert pauli_string.is_two_qubit
        assert pauli_string.get_pauli_type_pair() == 'XY'
        
        # Test single qubit
        single_qubit = PauliString(
            pauli_ops=[(0, 'Z')],
            coefficient=1.0
        )
        assert single_qubit.weight == 1
        assert not single_qubit.is_two_qubit
        assert single_qubit.get_pauli_type_pair() is None

    def test_coupling_matrices_zeros(self):
        """Test IsingCouplingMatrices.zeros class method."""
        from oqd_core.compiler.analog.passes.ising_types import IsingCouplingMatrices
        
        matrices = IsingCouplingMatrices.zeros(3)
        assert matrices.n_qubits == 3
        assert matrices.XX.shape == (3, 3)
        assert np.allclose(matrices.XX, np.zeros((3, 3)))
        assert np.allclose(matrices.YY, np.zeros((3, 3)))
        assert np.allclose(matrices.ZZ, np.zeros((3, 3)))
