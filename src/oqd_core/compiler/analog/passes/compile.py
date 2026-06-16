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

from oqd_core.backend.metric import Expectation
from oqd_core.compiler.analog.math.passes import canonicalize_math_expr
from oqd_core.compiler.analog.passes.assign import infer_analog_circuit_dim
from oqd_core.compiler.analog.verify.passes import (
    verify_analog_args_dim,
    verify_hamiltonian_target_dim,
    verify_register_access_dim,
)
from oqd_core.compiler.analog.passes.canonicalize import analog_operator_canonicalization
from oqd_core.interface.analog import AnalogCircuit, Declaration, Evolve, IfElse, While
from oqd_core.interface.analog.expr import MathExpr, OperatorExpr
from oqd_core.interface.analog.statement import Statement

########################################################################################

__all__ = [ "compile_analog_circuit" ]

########################################################################################



def canonicalize_math_in_stmt(stmt: Statement):
    if isinstance(stmt, Evolve):
        stmt.duration = canonicalize_math_expr(stmt.duration)
    elif isinstance(stmt, Declaration):
        if isinstance(stmt.value, MathExpr):
            stmt.value = canonicalize_math_expr(stmt.value)
        elif isinstance(stmt.value, OperatorExpr):
            pass
    elif isinstance(stmt, IfElse):
        stmt.then_branch = [canonicalize_math_in_stmt(s) for s in stmt.then_branch]
        stmt.else_branch = [canonicalize_math_in_stmt(s) for s in stmt.else_branch]
    elif isinstance(stmt, While):
        stmt.body = [canonicalize_math_in_stmt(s) for s in stmt.body]
    return stmt


def canonicalize_math_circuit(model: AnalogCircuit):
    for ind, stmt in enumerate(model.statements):
        model.statements[ind] = canonicalize_math_in_stmt(stmt)
    return model


def canonicalize_args_metrics(args):
    for metric in args.metrics.values():
        if isinstance(metric, Expectation):
            metric.operator = analog_operator_canonicalization(metric.operator)


def compile_analog_circuit(model: AnalogCircuit, analysis=None,  args=None):
    
    cfg = analysis.cfg
    symbol_table = analysis.symbol_table
    
    model = analog_operator_canonicalization(model)
    model = canonicalize_math_circuit(model)
    
    verify_register_access_dim(cfg, symbol_table)
    verify_hamiltonian_target_dim(cfg, symbol_table)
    
    n_qreg, n_qmode = infer_analog_circuit_dim(model)
    if args is not None:
        canonicalize_args_metrics(args)
        verify_analog_args_dim(args, n_qreg, n_qmode)
    
    return model, (n_qreg, n_qmode)

