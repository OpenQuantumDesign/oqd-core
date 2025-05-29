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

import numpy as np

from oqd_compiler_infrastructure import RewriteRule, Post
from oqd_core.interface.analog.operation import AnalogGate
from oqd_core.interface.analog.operator import (
    Operator, OperatorAdd, OperatorScalarMul, OperatorKron,
    Pauli, PauliI, PauliX, PauliY, PauliZ,
    Ladder, # Base class for Creation, Annihilation, Identity (bosonic)
)
from oqd_core.interface.math import MathNum


# Helper class to determine number of qubit registers and check for bosonic modes / consistency
class QubitCountRule(RewriteRule):
    def __init__(self):
        super().__init__()
        self.n_qreg = 0
        self.n_qmode = 0
        self.consistent_qreg_size = True
        self._processed_term_ops = set() # Avoid reprocessing shared sub-expressions if they are roots of terms

    def _update_qreg_size(self, size):
        if size <= 0: # qreg size must be positive if qubits are involved
            self.consistent_qreg_size = False
            return
        if self.n_qreg == 0:
            self.n_qreg = size
        elif self.n_qreg != size:
            self.consistent_qreg_size = False

    def _flatten_and_count_ops_in_term(self, op_node):
        q_count = 0
        m_count = 0
        
        elements = []
        # Recursive helper to flatten Kron products and collect Pauli/Ladder ops
        def _recursive_flatten(op):
            nonlocal q_count, m_count # Allow modification of outer scope variables
            if isinstance(op, OperatorKron):
                _recursive_flatten(op.op1)
                _recursive_flatten(op.op2)
            elif isinstance(op, Pauli):
                elements.append(op)
                q_count += 1
            elif isinstance(op, Ladder):
                elements.append(op)
                m_count += 1
            else: # Contains something other than Pauli or Ladder
                self.consistent_qreg_size = False
        
        _recursive_flatten(op_node)
        return q_count, m_count

    def _process_operator_part(self, op_part: Operator):
        if id(op_part) in self._processed_term_ops:
            return
        self._processed_term_ops.add(id(op_part))

        q_count, m_count = self._flatten_and_count_ops_in_term(op_part)
        
        if not self.consistent_qreg_size: return

        if m_count > 0:
            self.n_qmode += m_count 
        
        if q_count > 0:
            self._update_qreg_size(q_count)

    def map_OperatorScalarMul(self, model: OperatorScalarMul):
        if not self.consistent_qreg_size: return model
        self._process_operator_part(model.op)
        self.visit(model.op) # Continue traversal
        return model

    def map_OperatorKron(self, model: OperatorKron): # Handles terms like X@X (implicit coeff 1)
        if not self.consistent_qreg_size: return model
        self._process_operator_part(model)
        # No self.visit needed here as _process_operator_part flattens it
        return model
    
    def map_Pauli(self, model: Pauli): # Handles terms like X (implicit coeff 1)
        if not self.consistent_qreg_size: return model
        # This implies a term with a single Pauli operator.
        # For n_qreg counting, this term involves 1 qubit.
        self._update_qreg_size(1)
        self._processed_term_ops.add(id(model))
        return model

    def map_Ladder(self, model: Ladder): # Handles terms like A (implicit coeff 1)
        if not self.consistent_qreg_size: return model
        self.n_qmode += 1
        self._processed_term_ops.add(id(model))
        return model
    
    def map_OperatorAdd(self, model: OperatorAdd):
        if not self.consistent_qreg_size: return model
        self.visit(model.op1)
        self.visit(model.op2)
        return model

# Main analysis class
class IsingAnalysisRule(RewriteRule):
    def __init__(self, n_qreg, coupling_matrices_dict):
        super().__init__()
        self.n_qreg = n_qreg
        self.coupling_matrices = coupling_matrices_dict
        self.is_ising_like_overall = True

    def _get_pauli_char(self, pauli_op: Pauli):
        if isinstance(pauli_op, PauliX): return 'X'
        if isinstance(pauli_op, PauliY): return 'Y'
        if isinstance(pauli_op, PauliZ): return 'Z'
        if isinstance(pauli_op, PauliI): return 'I'
        return None 

    def _flatten_kron_for_ising_check(self, op_node):
        ordered_paulis = []
        # Recursive helper to flatten Kron products and collect Pauli ops in order
        def _recursive_flatten(op):
            if not self.is_ising_like_overall: return

            if isinstance(op, OperatorKron):
                _recursive_flatten(op.op1)
                _recursive_flatten(op.op2)
            elif isinstance(op, Pauli):
                ordered_paulis.append(op)
            elif isinstance(op, Ladder): # Rule 1: No bosonic
                self.is_ising_like_overall = False
            else: # Non-Pauli/Ladder in Kron
                self.is_ising_like_overall = False
        
        _recursive_flatten(op_node)
        return ordered_paulis

    def _process_term(self, operator_part: Operator, coefficient_val: float):
        if not self.is_ising_like_overall: return

        pauli_ops_in_order = self._flatten_kron_for_ising_check(operator_part)

        if not self.is_ising_like_overall: return 

        if len(pauli_ops_in_order) != self.n_qreg:
            self.is_ising_like_overall = False 
            return

        non_identity_terms = [] # List of (char, index)
        for i, p_op in enumerate(pauli_ops_in_order):
            char = self._get_pauli_char(p_op)
            if char is None: 
                self.is_ising_like_overall = False; return
            if char != 'I':
                non_identity_terms.append((char, i))
        
        if len(non_identity_terms) != 2: # Rule 2: Must be weight-2
            self.is_ising_like_overall = False
            return

        p1_char, idx1 = non_identity_terms[0]
        p2_char, idx2 = non_identity_terms[1]
        
        matrix_key = f"{p1_char}{p2_char}"
        self.coupling_matrices[matrix_key][idx1, idx2] += coefficient_val
        
        # For XX, YY, ZZ terms, make the matrix symmetric by convention J_ij = J_ji
        if p1_char == p2_char and idx1 != idx2: 
            self.coupling_matrices[matrix_key][idx2, idx1] += coefficient_val

    def map_OperatorScalarMul(self, model: OperatorScalarMul):
        if not self.is_ising_like_overall: return model

        if not isinstance(model.expr, MathNum): # Rule 3: Time-independent coeff
            self.is_ising_like_overall = False
            return model
        
        # Optionally check if model.expr.value is real if required by a stricter definition
        # if isinstance(model.expr.value, complex):
        #     self.is_ising_like_overall = False; return model

        self._process_term(model.op, float(model.expr.value))
        self.visit(model.op) 
        return model

    def map_OperatorKron(self, model: OperatorKron): 
        if not self.is_ising_like_overall: return model
        self._process_term(model, 1.0) # Implicit coefficient of 1.0
        return model

    def map_Pauli(self, model: Pauli): # Single Pauli term in a sum
        self.is_ising_like_overall = False # Not weight-2 for Ising interaction term
        return model
    
    def map_Ladder(self, model: Ladder): 
        self.is_ising_like_overall = False # Bosonic
        return model

    def map_OperatorAdd(self, model: OperatorAdd):
        if not self.is_ising_like_overall: return model
        self.visit(model.op1)
        self.visit(model.op2)
        return model

def ising_hamiltonian_analysis(gate: AnalogGate):
    """
    Analyzes an AnalogGate's Hamiltonian to check if it's Ising-like and extracts coupling matrices.

    An Ising-like Hamiltonian must satisfy:
    1. Composed of only qubit registers (no bosonic mode operators).
    2. Each additive term is a weight-2 Pauli string (e.g., c * X_i @ Z_j).
       Exactly two Pauli operators in the tensor product must be non-identity.
    3. Coefficients must be time-independent (represented by MathNum).

    Args:
        gate (AnalogGate): The analog gate to analyze.

    Returns:
        dict: A dictionary of coupling matrices (e.g., 'XX', 'XY', etc.) as numpy arrays
              if the Hamiltonian is Ising-like. For 'XX', 'YY', 'ZZ' matrices, symmetry
              (J_ij = J_ji) is enforced.
        None: If the Hamiltonian is not Ising-like or an error occurs.
    """
    hamiltonian = gate.hamiltonian

    # 1. Determine n_qreg and check for bosonic modes / consistency
    counter_rule = QubitCountRule()
    Post(counter_rule).visit(hamiltonian)

    if not counter_rule.consistent_qreg_size:
        return None 
    if counter_rule.n_qmode > 0:
        return None 

    n_qreg = counter_rule.n_qreg
    if n_qreg == 0: # No qubit operators found, or inconsistent leading to 0
        # If an empty Hamiltonian for 0 qubits is valid, return empty matrices.
        # For now, assume n_qreg > 0 for a meaningful Ising Hamiltonian.
        return None

    # 2. Initialize coupling matrices
    pauli_chars = ['X', 'Y', 'Z']
    coupling_matrices = {f'{p1}{p2}': np.zeros((n_qreg, n_qreg)) 
                         for p1 in pauli_chars for p2 in pauli_chars}

    # 3. Analyze Hamiltonian structure and populate matrices
    analyzer = IsingAnalysisRule(n_qreg, coupling_matrices)
    Post(analyzer).visit(hamiltonian)

    if not analyzer.is_ising_like_overall:
        return None

    return analyzer.coupling_matrices