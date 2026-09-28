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

########################################################################################


from __future__ import annotations

from typing import Tuple, Union

from oqd_compiler_infrastructure import (
    CFGBlock,
    ForwardDataflowAnalysis,
    Lattice,
    LatticeTop,
    MapLatticeValue,
    Post,
    VisitableBaseModel,
    gen_pass,
    maplattice,
)

from oqd_core.interface.analog import (
    Access,
    AnalogExpr,
    Constant,
    Declaration,
)
from oqd_core.interface.analog.expr import Operator

########################################################################################


class ConstantFoldingError(Exception): ...


########################################################################################


class CLatticeTop(LatticeTop): ...


class CInvalid(CLatticeTop): ...


ConstantFoldingValue = Union[CLatticeTop, Constant, Operator]


class ConstantLattice(Lattice[ConstantFoldingValue]):
    def top(self):
        return CLatticeTop

    def bottom(self):
        return CInvalid

    def leq(self, t1, t2):
        if self.join(t1, t2) == t1:
            return True
        return False

    def join(self, t1, t2):
        if t1 == t2:
            return t1

        if t1 == self.bottom():
            return t2

        if t2 == self.bottom():
            return t1

        return self.top()

    def meet(self, t1, t2):
        if t1 == t2:
            return t1

        if t1 == self.top():
            return t2

        if t2 == self.top():
            return t1

        return self.bottom()


########################################################################################


class ConstantFolding(ForwardDataflowAnalysis[int, CFGBlock, ConstantFoldingValue]):
    lattice = maplattice(ConstantLattice)()

    def initial_state(self, nodes):
        return {
            node: self.lattice.bottom() if node == 0 else self.lattice.top()
            for node in nodes
        }

    def merge(self, states):
        return self.lattice.merge_intersection(states)

    @gen_pass(rule_type="conversion", walk=Post, method=True)
    def _fold(self, model, operands, *, env):
        match model:
            case Constant() | Operator():
                return model

            case Access():
                return env.get(model.name, self.lattice.element_lattice.bottom())

            case Declaration(value=Constant() | Operator() | Access()):
                env[model.name] = operands["value"]

            case Declaration():
                env[model.name] = self.lattice.element_lattice.bottom()

            case _:
                return self.lattice.element_lattice.bottom()

    def transfer(self, graph, node_id, state_in):
        block = graph[node_id]

        state_out = state_in.copy() if isinstance(state_in, dict) else {}

        self._fold(block, env=state_out)
        return state_out
