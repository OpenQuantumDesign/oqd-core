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

from oqd_core.compiler.analog.operator.dim import operator_dim
from oqd_core.analysis.analog.symbol_table import AnalogSymbolTable, target_dim
from oqd_core.analysis.utils.control_flow import ControlFlowGraph
from oqd_core.interface.analog import Evolve, Initialize, Measure
from oqd_core.backend.metric import Expectation
from oqd_core.compiler.analog.error import AnalogCompilerError
from oqd_core.compiler.analog.cfg_passes.walk import iter_stmt_blocks

__all__ = [
    "verify_analog_args_dim",
    "verify_register_access_dim",
    "verify_hamiltonian_target_dim",
]

def verify_analog_args_dim(model, n_qreg, n_qmode):
    """
    This pass checks whether the assigned n_qreg and n_qmode in AnalogCircuit match the n_qreg and n_qmode
    in any Operators (like the Operator inside Expectation) in TaskArgsAnalog

    Args:
        model (TaskArgsAnalog):

    Returns:
        model (TaskArgsAnalog):

    Assumptions:
        All  [`Operator`][oqd_core.interface.analog.expr.OperatorExpr] inside TaskArgsAnalog must be canonicalized
    """
    expected = (n_qreg, n_qmode)
    for metric in model.metrics.values():
        if not isinstance(metric, Expectation):
            continue
        
        dim = operator_dim(metric.operator)
    
        if dim is None or dim != expected:
            raise AnalogCompilerError(f"Inconsistent Hilbert space dimension.")
        
    return model


def verify_register_access_dim(cfg: ControlFlowGraph, symbol_table: AnalogSymbolTable):
    
    for node_id, block in iter_stmt_blocks(cfg):
        stmt = block.stmt
        if not isinstance(stmt, (Evolve, Initialize, Measure)):
            continue
        env = symbol_table.in_env[node_id]
        target_dim(stmt.targets, env)
        
    return cfg

def verify_hamiltonian_target_dim(cfg: ControlFlowGraph, symbol_table: AnalogSymbolTable):
    
    for node_id, block in iter_stmt_blocks(cfg):
        stmt = block.stmt
        if not isinstance(stmt, Evolve):
            continue
        env = symbol_table.in_env[node_id]
        h_dim = operator_dim(stmt.hamiltonian)
        t_dim = target_dim(stmt.targets, env)
        if h_dim != t_dim:
            raise AnalogCompilerError(f"Inconsistent Hilbert space dimension.")
        
    return cfg

