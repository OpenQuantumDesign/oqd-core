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
from oqd_core.compiler.analog.cfg.walk import canonicalize_math_cfg
from oqd_core.compiler.analog.cfg.operator_env import canonicalize_operators_cfg
from oqd_core.compiler.analog.passes.assign import infer_analog_circuit_dim_cfg
from oqd_core.compiler.analog.operator.canonicalize import canonicalize_operator_expr
from oqd_core.compiler.analog.verify.passes import verify_analog_args_dim, verify_hamiltonian_target_dim, verify_register_access_dim
from oqd_core.analysis.analog.analyze import AnalogAnalysisResult
from oqd_core.interface.analog import AnalogCircuit

########################################################################################

__all__ = [ "compile_analog_circuit" ]

########################################################################################


def canonicalize_args_metrics(args):
    for metric in args.metrics.values():
        if isinstance(metric, Expectation):
            metric.operator = canonicalize_operator_expr(metric.operator)


def compile_analog_circuit(model: AnalogCircuit, analysis_result: AnalogAnalysisResult,  args=None):
    
    cfg = analysis_result.cfg
    symbol_table = analysis_result.symbol_table
    canonicalize_operators_cfg(cfg)
    canonicalize_math_cfg(cfg)
    
    verify_register_access_dim(cfg, symbol_table)
    verify_hamiltonian_target_dim(cfg, symbol_table)
    
    n_qreg, n_qmode = infer_analog_circuit_dim_cfg(cfg)
    if args is not None:
        canonicalize_args_metrics(args)
        verify_analog_args_dim(args, n_qreg, n_qmode)
    
    return model, (n_qreg, n_qmode)

