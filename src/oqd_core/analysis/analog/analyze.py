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

from oqd_compiler_infrastructure import DataflowResult
from pydantic import BaseModel, ConfigDict

from oqd_core.analysis.analog.cfg import AnalogCFGBuilder
from oqd_core.analysis.analog.symbol_table import (
    AnalogSymbolTable,
    AnalogSymbolTableBuilder,
)
from oqd_core.analysis.analog.type_checker import AnalogTypeChecker
from oqd_core.analysis.utils.control_flow import ControlFlowGraph


class AnalogAnalysisResult(BaseModel):
    cfg : ControlFlowGraph
    dataflow_result: DataflowResult
    symbol_table: AnalogSymbolTable
    model_config = ConfigDict(arbitrary_types_allowed=True)
    

class Analyze:
    result: AnalogAnalysisResult

    def __init__(self, circuit):
        cfg = AnalogCFGBuilder().run(circuit)
        type_checker = AnalogTypeChecker(cfg)
        dataflow_result = type_checker.dataflow_result
        symbol_analysis = AnalogSymbolTableBuilder(cfg, dataflow_result)
        symbol_table = symbol_analysis.symbol_table
        self.result = AnalogAnalysisResult(
            cfg = cfg, dataflow_result = dataflow_result, symbol_table = symbol_table
        )


