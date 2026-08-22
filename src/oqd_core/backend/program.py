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

from oqd_compiler_infrastructure import VisitableBaseModel

from oqd_core.analysis.analog.symbol_table import AnalogSymbolTable
from oqd_core.analysis.atomic.symbol_table import AtomicSymbolTable
from oqd_core.analysis.utils import ControlFlowGraph
from oqd_core.interface.analog import AnalogCircuit
from oqd_core.interface.atomic import AtomicCircuit

########################################################################################


class AnalogProgram(VisitableBaseModel):
    circuit: AnalogCircuit
    cfg: ControlFlowGraph
    symbol_table: AnalogSymbolTable


class AtomicProgram(VisitableBaseModel):
    circuit: AtomicCircuit
    cfg: ControlFlowGraph
    symbol_table: AtomicSymbolTable
