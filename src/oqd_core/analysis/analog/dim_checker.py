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

from functools import reduce
from typing import List, Tuple, Type, Union
from warnings import warn

from oqd_compiler_infrastructure import (
    CFGBlock,
    ForwardDataflowAnalysis,
    Lattice,
    LatticeTop,
    MapLatticeValue,
    maplattice,
)

from oqd_core.analysis.utils import AnalysisError, AnalysisWarning
from oqd_core.frontend.analog import serialize_analog
from oqd_core.interface.analog import (
    Access,
    Add,
    AnalogList,
    Annihilation,
    BuiltinCall,
    Constant,
    Creation,
    Declaration,
    Div,
    Evolve,
    Extract,
    Identity,
    Kron,
    ModeRegister,
    Mul,
    Neg,
    PauliI,
    PauliX,
    PauliY,
    PauliZ,
    Pos,
    Pow,
    QuantumRegister,
    Sub,
)

########################################################################################


class DAny(LatticeTop): ...


class DInvalid(DAny): ...


DQuantum = List[Union[int, "DQuantum"]]

DLatticeValue = Union[DQuantum, Type[DAny]]

########################################################################################


class DimensionLattice(Lattice[DLatticeValue]):
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


class DimensionError(AnalysisError):
    """Dimension error class for Analog."""


class DimensionWarning(AnalysisWarning): ...


########################################################################################


DimEnv = MapLatticeValue[DLatticeValue]


class DimensionChecker(ForwardDataflowAnalysis[int, CFGBlock, DimEnv]):
    """Forward dataflow dimension checker over the Control Flow Graph."""

    lattice = maplattice(DimensionLattice, default_mode="top")()

    def _is_integer(self, expr):
        return isinstance(expr, Constant) and type(expr.value) is int

    def _infer_dim(self, expr, *, env: DimEnv, loc: Tuple[int, int]):
        match expr:
            case Access():
                return (
                    self.lattice.element_lattice.top()
                    if env is self.lattice.top()
                    else env.get(expr.name, self.lattice.element_lattice.top())
                )

            case (
                PauliX(dims=dims)
                | PauliY(dims=dims)
                | PauliZ(dims=dims)
                | PauliI(dims=dims)
                | Annihilation(dims=dims)
                | Creation(dims=dims)
                | Identity(dims=dims)
            ) if self._is_integer(dims):
                return [dims.value]

            case (
                PauliX(dims=dims)
                | PauliY(dims=dims)
                | PauliZ(dims=dims)
                | PauliI(dims=dims)
                | Annihilation(dims=dims)
                | Creation(dims=dims)
                | Identity(dims=dims)
            ) if isinstance(dims, AnalogList) and all(
                [self._is_integer(dim) for dim in dims.values]
            ):
                return [dim.value for dim in dims.values]

            case (
                PauliX(dims=dims)
                | PauliY(dims=dims)
                | PauliZ(dims=dims)
                | PauliI(dims=dims)
                | Annihilation(dims=dims)
                | Creation(dims=dims)
                | Identity(dims=dims)
            ):
                warn(
                    f"(block={loc[0]}, stmt={loc[1]}) DimensionChecker only checks dimensions of an operator of constant dims,"
                    f" got `{serialize_analog(expr)}`",
                    DimensionWarning,
                )
                return self.lattice.element_lattice.top()

            case QuantumRegister() if self._is_integer(expr.size) and self._is_integer(
                expr.dim
            ):
                return [[expr.dim.value]] * expr.size.value

            case ModeRegister() if self._is_integer(expr.size):
                return [[-1]] * expr.size.value

            case QuantumRegister() | ModeRegister():
                warn(
                    f"(block={loc[0]}, stmt={loc[1]}) DimensionChecker only checks dimensions of a quantum register of constant size,"
                    f" got `{serialize_analog(expr)}`",
                    DimensionWarning,
                )
                return self.lattice.element_lattice.top()

            case AnalogList():
                return [
                    self._infer_dim(element, env=env, loc=loc)
                    for element in expr.values
                ]

            case Extract() if self._is_integer(expr.index):
                value = self._infer_dim(expr.value, env=env, loc=loc)

                if value is self.lattice.element_lattice.top():
                    return value

                if value is self.lattice.element_lattice.bottom():
                    return value

                return value[expr.index.value]

            case Neg() | Pos():
                arg = self._infer_dim(expr.expr, env=env, loc=loc)
                return arg

            case Add() | Sub() | Mul():
                binop = {Add: "add", Sub: "subtract", Mul: "multiply"}[expr.__class__]

                args = [self._infer_dim(e, env=env, loc=loc) for e in expr.exprs]
                op_args = list(
                    filter(
                        lambda x: x is not self.lattice.element_lattice.bottom(),
                        args,
                    )
                )

                dim = reduce(
                    self.lattice.element_lattice.meet,
                    op_args,
                    self.lattice.element_lattice.top(),
                )

                if op_args and dim is self.lattice.element_lattice.bottom():
                    raise DimensionError(
                        f"attempted to {binop} operators of different dimensions {tuple(args)}",
                        loc=loc,
                    )

                return dim

            case Div() | Pow():
                arg0 = self._infer_dim(expr.exprs[0], env=env, loc=loc)

                return arg0

            case Kron():
                args = [self._infer_dim(e, env=env, loc=loc) for e in expr.exprs]

                return reduce(lambda x, y: x + y, args, [])

            case Evolve():
                hamiltonian_dim = self._infer_dim(expr.hamiltonian, env=env, loc=loc)
                jumps_dim = [
                    self._infer_dim(L, env=env, loc=loc) for L in expr.jumps.values
                ]
                targets_dim = self._infer_dim(expr.targets, env=env, loc=loc)

                if isinstance(targets_dim, list) and isinstance(targets_dim[0], list):
                    targets_dim = [d for target in targets_dim for d in target]

                args = (hamiltonian_dim, *jumps_dim, targets_dim)

                if (
                    self.lattice.element_lattice.merge_meet(args)
                    is not self.lattice.element_lattice.bottom()
                ):
                    return self.lattice.element_lattice.bottom()

                raise DimensionError(
                    f"evolve got inconsistent Hamiltonian dimensions ({args[0]}), jump operators dimensions {args[1:-1]}"
                    f" and target dimensions ({args[-1]})",
                    loc=loc,
                )

            case BuiltinCall(
                func="sin" | "cos" | "tan" | "exp" | "log" | "sinh" | "cosh" | "tanh",
                args=args,
            ):
                return self._infer_dim(args[0], env=env, loc=loc)

            case _:
                return self.lattice.element_lattice.bottom()

    def merge(self, states):
        return self.lattice.merge_meet(states)

    def transfer(self, graph, node_id, state_in):
        block = graph[node_id]

        state_out = {} if state_in == self.lattice.top() else state_in.copy()
        for n, stmt in enumerate(block.stmts):
            match stmt:
                case Declaration():
                    state_out[stmt.name] = self._infer_dim(
                        stmt.value, env=state_out, loc=(node_id, n)
                    )
                case _:
                    self._infer_dim(stmt, env=state_out, loc=(node_id, n))

        return state_out
