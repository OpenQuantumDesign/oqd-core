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

from oqd_core.frontend.analog.AnalogCircuitAST import parse_analog
from oqd_core.interface.analog import Declaration, AnalogExpr


def parse_math(source: str | AnalogExpr):
    if isinstance(source, AnalogExpr):
        return source
    circuit = parse_analog(f"x = {source}")
    assert circuit.statements, f"Failed to parse: {source!r}"
    stmt = circuit.statements[0]
    assert isinstance(stmt, Declaration)
    return stmt.value

