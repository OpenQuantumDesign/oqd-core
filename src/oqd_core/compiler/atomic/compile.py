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

from oqd_core.analysis.atomic import (
    AtomicBoundsChecker,
    AtomicTypeChecker,
    AvailableVariableAnalysis,
)
from oqd_core.compiler.atomic.conversion import (
    AtomicCFGBuilder,
    AtomicCFGtoAST,
    PyASTtoAtomic,
)
from oqd_core.frontend.atomic import parse_atomic
from oqd_core.interface.atomic import AtomicCircuit

########################################################################################

__all__ = ["compile_atomic_circuit"]

########################################################################################


def compile_atomic_circuit(
    source: str | AtomicCircuit | CFG, *, return_analysis=False
) -> tuple[AtomicCircuit, CFG]:
    cfg_builder = Chain(AtomicCFGBuilder(), CFGBlockAccumulator(), RelabelCFGBlocks())

    match source:
        case str():
            circuit = parse_atomic(source)
            cfg = cfg_builder(circuit)
        case AtomicCircuit():
            circuit = source
            cfg = cfg_builder(circuit)
        case CFG():
            cfg = source
            cfg_to_ast = AtomicCFGtoAST()
            circuit = cfg_to_ast(cfg)

    avail_checker = AvailableVariableAnalysis()
    type_checker = AtomicTypeChecker()
    bounds_checker = AtomicBoundsChecker()

    avail_res = avail_checker.analyze(cfg)
    type_res = type_checker.analyze(cfg)
    bounds_res = bounds_checker.analyze(cfg)

    if return_analysis:
        return circuit, cfg, (avail_res, type_res, bounds_res)

    return circuit, cfg


def atomic(func=None, *, return_analysis=False):
    def _decorator(_func):
        source = inspect.getsource(func)
        pyast = ast.parse(source)

        circuit = Post(PyASTtoAtomic())(pyast)

        cfg_builder = Chain(
            AtomicCFGBuilder(), CFGBlockAccumulator(), RelabelCFGBlocks()
        )

        cfg = cfg_builder(circuit)

        avail_checker = AvailableVariableAnalysis()
        type_checker = AtomicTypeChecker()
        bounds_checker = AtomicBoundsChecker()

        avail_res = avail_checker.analyze(cfg)
        type_res = type_checker.analyze(cfg)
        bounds_res = bounds_checker.analyze(cfg)

        if return_analysis:
            return circuit, cfg, (avail_res, type_res, bounds_res)

        return circuit, cfg

    if func:
        return _decorator(func)

    return _decorator

