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

import inspect

import ast_comments as ast
from oqd_compiler_infrastructure import (
    CFG,
    CFGBlockAccumulator,
    Chain,
    Post,
    RelabelCFGBlocks,
)

from oqd_core.analysis.analog import (
    AnalogTypeChecker,
    AvailableVariableAnalysis,
    DimensionChecker,
)
from oqd_core.compiler.analog.conversion import (
    AnalogCFGBuilder,
    AnalogCFGtoAST,
    PyASTtoAnalog,
)
from oqd_core.frontend.analog import parse_analog
from oqd_core.interface.analog import AnalogCircuit

########################################################################################

__all__ = ["compile_analog_circuit"]

########################################################################################


def compile_analog_circuit(
    source: str | AnalogCircuit | CFG, *, return_analysis=False
) -> tuple[AnalogCircuit, CFG]:
    cfg_builder = Chain(AnalogCFGBuilder(), CFGBlockAccumulator(), RelabelCFGBlocks())

    match source:
        case str():
            circuit = parse_analog(source)
            cfg = cfg_builder(circuit)
        case AnalogCircuit():
            circuit = source
            cfg = cfg_builder(circuit)
        case CFG():
            cfg = source
            cfg_to_ast = AnalogCFGtoAST()
            circuit = cfg_to_ast(cfg)

    avail_checker = AvailableVariableAnalysis()
    type_checker = AnalogTypeChecker()
    dim_checker = DimensionChecker()

    avail_res = avail_checker.analyze(cfg)
    type_res = type_checker.analyze(cfg)
    dim_res = dim_checker.analyze(cfg)

    if return_analysis:
        return circuit, cfg, (avail_res, type_res, dim_res)

    return circuit, cfg


def analog(func=None, *, return_analysis=False):
    def _decorator(_func):
        source = inspect.getsource(func)
        pyast = ast.parse(source)

        circuit = Post(PyASTtoAnalog())(pyast)

        cfg_builder = Chain(
            AnalogCFGBuilder(), CFGBlockAccumulator(), RelabelCFGBlocks()
        )

        cfg = cfg_builder(circuit)

        avail_checker = AvailableVariableAnalysis()
        type_checker = AnalogTypeChecker()
        dim_checker = DimensionChecker()

        avail_res = avail_checker.analyze(cfg)
        type_res = type_checker.analyze(cfg)
        dim_res = dim_checker.analyze(cfg)

        if return_analysis:
            return circuit, cfg, (avail_res, type_res, dim_res)

        return circuit, cfg

    if func:
        return _decorator(func)

    return _decorator
