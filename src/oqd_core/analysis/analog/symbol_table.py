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

from __future__ import annotations

from typing import Iterable, Union

from oqd_compiler_infrastructure.dataflow import (
    DataflowResult,
    ForwardDataflowAnalysis,
)
from oqd_compiler_infrastructure.lattice import (
    Lattice,
    LatticeBottom,
    LatticeTop,
    maplattice,
)
from pydantic import BaseModel, ConfigDict

from oqd_core.analysis.analog.type_checker import AnalogTypeChecker
from oqd_core.analysis.analog.types import (
    TLatticeValue,
    TList,
    TMRef,
    TMReg,
    TQRef,
    TQReg,
    TTargetRef,
    TypeEnv,
)
from oqd_core.analysis.utils.control_flow import CFGStart, CFGStop, ControlFlowGraph
from oqd_core.interface.analog import (
    Access,
    AnalogList,
    Declaration,
    Extract,
    ModeRegister,
    QuantumRegister,
)

########################################################################################


class AnalogSymbolError(TypeError):
    """Symbol table error class for Analog."""
    pass


class SymbolBinding(BaseModel):
    lattice_type: TLatticeValue
    target_dim: tuple[int, int]
    list_elem: SymbolBinding | None = None
    model_config = ConfigDict(frozen=True)

RegisterEnv = dict[str, SymbolBinding]

class AnalogSymbolTable(BaseModel):
    in_env: dict[int, RegisterEnv]
    stmt_index: dict[int, int]


class SymbolBindingLattice(Lattice[Union[SymbolBinding, type[LatticeTop]]]):
    def top(self) -> type[LatticeTop]:
        return LatticeTop
    
    def bottom(self) -> type[LatticeTop]:
        return LatticeBottom
    
    def leq(self, t1, t2) -> bool:
        if t1 is LatticeBottom:
            return True
        if t2 is LatticeTop:
            return True
        if t1 is LatticeTop:
            return t2 is LatticeTop
        if t2 is LatticeBottom:
            return False
        return t1 == t2
    
    def join(self, t1, t2):
        if t1 is LatticeTop or t2 is LatticeTop:
            return LatticeTop
        if t1 is LatticeBottom:
            return t2
        if t2 is LatticeBottom:
            return t1
        if t1 == t2:
            return t1
        return LatticeTop
    
    def meet(self, t1, t2):
        if t1 is LatticeBottom or t2 is LatticeBottom:
            return LatticeBottom
        if t1 is LatticeTop:
            return t2
        if t2 is LatticeTop:
            return t1
        if t1 == t2:
            return t1
        return LatticeBottom


def is_target_lattice_type(t: TLatticeValue) -> bool:
    if t in (TQReg, TMReg, TQRef, TMRef, TTargetRef):
        return True
    if isinstance(t, TList):
        return is_target_lattice_type(t.elem)
    return False


def bind_target_value(expr, t: TLatticeValue, env: RegisterEnv) -> SymbolBinding:
    
    if isinstance(expr, Access):
        if expr.name not in env:
            raise AnalogSymbolError(f"Undefined variable: {expr.name}")
        return env[expr.name]
    
    if t is TQReg:
        return SymbolBinding(lattice_type=TQReg, target_dim=(expr.size, 0))
    
    if t is TMReg:
        return SymbolBinding(lattice_type=TMReg, target_dim=(0, expr.size))
    
    if isinstance(expr, Extract):
        if expr.access.name not in env:
            raise AnalogSymbolError(f"Undefined variable: {expr.access.name}")
        base = env[expr.access.name]
        n_qreg, n_qmode = base.target_dim
        if n_qreg > 0:
            if expr.index >= n_qreg:
                raise AnalogSymbolError("Extract index out of range")
            return SymbolBinding(lattice_type=TQRef, target_dim=(1, 0))
        if n_qmode > 0:
            if expr.index >= n_qmode:
                raise AnalogSymbolError("Extract index out of range")
            return SymbolBinding(lattice_type=TMRef, target_dim=(0, 1))
        raise AnalogSymbolError("Extract index out of range")
    
    if isinstance(expr, AnalogList):
        if not isinstance(t, TList) or not is_target_lattice_type(t):
            raise AnalogSymbolError("target list expected")
        elem_bindings = [bind_target_value(v, t.elem, env) for v in expr.values]
        target_dim = (0, 0)
        for binding in elem_bindings:
            target_dim = (
                target_dim[0] + binding.target_dim[0],
                target_dim[1] + binding.target_dim[1],
            )
        return SymbolBinding(
            lattice_type=t,
            target_dim=target_dim,
            list_elem=elem_bindings[0],
        )
    raise AnalogSymbolError(f"Unsupported target expression: {type(expr).__name__}")


def target_dim(expr, env: RegisterEnv):
    if isinstance(expr, Access):
        if expr.name not in env:
            raise AnalogSymbolError(f"Undefined Variable: {expr.name}")
        return env[expr.name].target_dim
    
    if isinstance(expr, Extract):
        if expr.access.name not in env:
            raise AnalogSymbolError(f"Undefined Variable: {expr.access.name}")
        base = env[expr.access.name]
        n_qreg, n_qmode = base.target_dim
        if n_qreg > 0:
            if expr.index >= n_qreg:
                raise AnalogSymbolError("Extract index out of range")
            return (1, 0)
        if n_qmode > 0:
            if expr.index >= n_qmode:
                raise AnalogSymbolError("Extract index out of range")
            return (0,1)
        raise AnalogSymbolError("Extract index out of range")
    
    if isinstance(expr, AnalogList):
        dim = (0, 0)
        for value in expr.values:
            new = target_dim(value, env)
            dim = (dim[0] + new[0], dim[1] + new[1])
        return dim
    
    if isinstance(expr, QuantumRegister):
        return (expr.size, 0)
        
    if isinstance(expr, ModeRegister):
        return (0, expr.size)
    
    raise AnalogSymbolError(f"Invalid target: {type(expr).__name__} ")


class AnalogSymbolTableBuilder(ForwardDataflowAnalysis[int, RegisterEnv]):
    """Forward dataflow symbol table for register / target dimension checking."""
    def __init__(self, graph: ControlFlowGraph, type_result: DataflowResult[int, TypeEnv] | None = None,) -> None:
        if type_result is None:
            type_result = AnalogTypeChecker(graph).dataflow_result
        self.type_out_states = type_result.out_states
        
        self.lattice = maplattice(SymbolBindingLattice)()
        self.blocks = graph.blocks
        
        self.dataflow_result = self.analyze(graph, self.merge_symbol_env)

        self.symbol_table = AnalogSymbolTable(
            in_env={
                node_id: {} if state is LatticeBottom else dict(state)
                for node_id, state in self.dataflow_result.in_states.items()
            },
            stmt_index={
                id(block.stmt): node_id
                for node_id, block in graph.blocks.items()
                if not isinstance(block.stmt, (CFGStart, CFGStop))
            },
        )
    
    def merge_symbol_env(self, states: Iterable[RegisterEnv]) -> RegisterEnv:
        states_list = list(states)
        if not states_list:
            return self.lattice.bottom()
        merged = {} if states_list[0] is LatticeBottom else dict(states_list[0])
        for state in states_list[1:]:
            if state is LatticeBottom:
                continue
            for name in set(merged).union(state):
                b1 = merged.get(name)
                b2 = state.get(name)
                if b1 is None:
                    merged[name] = b2
                elif b2 is None:
                    continue
                elif b1 != b2:
                    raise AnalogSymbolError(f"Incompatible register bindings for {name}")
        return merged
    
    def transfer(self, node_id: int, state_in: RegisterEnv) -> RegisterEnv:
        env = {} if state_in is LatticeBottom else dict(state_in)
        stmt = self.blocks[node_id].stmt
        if isinstance(stmt, Declaration):
            t = self.type_out_states[node_id].get(stmt.name)
            if t is not None and is_target_lattice_type(t):
                state_out = dict(env)
                state_out[stmt.name] = bind_target_value(stmt.value, t, env)
                return state_out
        return env

