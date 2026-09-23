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

from typing import List, Union

from oqd_compiler_infrastructure import (
    CFGBlock,
    ForwardDataflowAnalysis,
    LatticeBase,
    LatticeTop,
    maplattice,
)

from oqd_core.interface.analog import (
    Access,
    AnalogList,
    Annihilation,
    Break,
    Continue,
    Creation,
    Declaration,
    Evolve,
    Extract,
    Identity,
    MathMul,
    ModeRegister,
    OperatorAdd,
    OperatorKron,
    OperatorMul,
    OperatorSub,
    PauliI,
    PauliX,
    PauliY,
    PauliZ,
    QuantumRegister,
)

########################################################################################


class DAny(LatticeTop): ...


class DInvalid(DAny): ...


DQuantum = List[Union[int, "DQuantum"]]

DLatticeValue = Union[DAny, DInvalid, DQuantum]


class DimensionLattice(LatticeBase[DLatticeValue]):
    """Quantum dimension lattice for analog layer."""

    def top(self):
        return DAny

    def bottom(self):
        return DInvalid

    def leq(self, t1: DLatticeValue, t2: DLatticeValue) -> bool:
        if self.join(t1, t2) == t1:
            return True
        return False

    def join(self, t1: DLatticeValue, t2: DLatticeValue) -> DLatticeValue:
        if t1 == t2:
            return t1

        if t1 == self.bottom():
            return t2

        if t2 == self.bottom():
            return t1

        return self.top()

    def meet(self, t1: DLatticeValue, t2: DLatticeValue) -> DLatticeValue:
        if t1 == t2:
            return t1

        if t1 == self.top():
            return t2

        if t2 == self.top():
            return t1

        return self.bottom()


########################################################################################


class DimensionError(Exception): ...


########################################################################################


class DimensionChecker(ForwardDataflowAnalysis[int, CFGBlock, DLatticeValue]):
    lattice = maplattice(DimensionLattice)()

    def merge(self, states):
        return self.lattice.merge_meet(states)

    def _infer_dim(self, expr, *, env):
        match expr:
            case Access():
                return env[expr.name]

            case PauliX() | PauliY() | PauliZ() | PauliI():
                return [2]

            case Annihilation() | Creation() | Identity():
                return [-1]

            case QuantumRegister():
                return [[2]] * expr.size

            case ModeRegister():
                return [[-1]] * expr.size

            case AnalogList():
                return [self._infer_dim(element, env=env) for element in expr.values]

            case Extract():
                value = self._infer_dim(expr.access, env=env)

                return value if value == DInvalid else value[expr.index]

            case OperatorAdd() | OperatorSub():
                args = (
                    self._infer_dim(expr.op1, env=env),
                    self._infer_dim(expr.op2, env=env),
                )

                if not self.lattice.element_lattice.equal(args[0], args[1]):
                    raise DimensionError()

                return args[0]

            case OperatorKron():
                args = (
                    self._infer_dim(expr.op1, env=env),
                    self._infer_dim(expr.op2, env=env),
                )

                return args[0] + args[1]

            case OperatorMul():
                args = (
                    self._infer_dim(expr.op1, env=env),
                    self._infer_dim(expr.op2, env=env),
                )

                if not self.lattice.element_lattice.equal(args[0], args[1]):
                    raise DimensionError()

                return args[0]

            case MathMul():
                args = (
                    self._infer_dim(expr.expr1, env=env),
                    self._infer_dim(expr.expr2, env=env),
                )

                return self.lattice.element_lattice.join(args)

            case Evolve():
                args = (
                    self._infer_dim(expr.hamiltonian, env=env),
                    self._infer_dim(expr.targets, env=env),
                )

                if self.lattice.element_lattice.equal(args[0], args[1]):
                    return DInvalid

                if all(
                    map(lambda x: isinstance(x, list), args[1])
                ) and self.lattice.element_lattice.equal(
                    args[0], [a[0] for a in args[1]]
                ):
                    return DInvalid

                raise DimensionError(
                    f"Got Hamiltonian dimensions ({args[0]}) and target dimensions ({args[1]}), expected target dimensions to be one of:\n"
                    f"  {args[0]}\n"
                    f"  {[[a] for a in args[0]]}"
                )

            case _:
                return DInvalid

    def transfer(self, graph, node_id, state_in):
        block = graph[node_id]

        state_out = {} if state_in == self.lattice.top() else state_in.copy()

        for stmt in block.stmts:
            match stmt:
                case _ if block.edge_labels:
                    continue
                case Continue() | Break():
                    continue
                case Declaration():
                    state_out[stmt.name] = self._infer_dim(stmt.value, env=state_out)
                case _:
                    self._infer_dim(stmt, env=state_out)

        return state_out
