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

from oqd_compiler_infrastructure.dataflow import DataflowResult, ForwardDataflowAnalysis
from oqd_compiler_infrastructure.lattice import Lattice, LatticeBottom, LatticeTop, maplattice

from oqd_core.compiler.atomic.cfg_passes.walk import canonicalize_beam, canonicalize_expr
from oqd_core.analysis.atomic.types import TBeam, TBool, TScalar, TLatticeValue, TypeEnv, TPulse
from oqd_core.analysis.utils.control_flow import ControlFlowGraph
from oqd_core.compiler.atomic.error import AtomicCompilerError
from oqd_core.compiler.atomic.math.passes import canonicalize_math_expr
from oqd_core.interface.atomic import (
    Access,
    Bool,
    BoolAnd,
    BoolEq,
    BoolGreaterThan,
    BoolGreaterThanEq,
    BoolLessThan,
    BoolLessThanEq,
    BoolNot,
    BoolNotEq,
    BoolOr,
    Declaration,
    MathAdd,
    MathDiv,
    MathFunc,
    MathImag,
    MathMul,
    MathNum,
    MathPow,
    MathSub,
    MathVar,
    Pulse,
    Beam,
    AtomicList,
    ParallelProtocol,
    SerialProtocol,
)

ScalarEnv = dict[str, object]

MATH_BOOL_TYPES = (MathAdd, MathSub, MathMul, MathDiv, MathPow, 
                   BoolAnd, BoolOr, BoolEq, BoolNotEq, BoolLessThan, 
                   BoolLessThanEq, BoolGreaterThan, BoolGreaterThanEq)

class ScalarExprLattice(Lattice[Union[object, type[LatticeTop]]]):
    def top(self):
        return LatticeTop
    
    def bottom(self):
        return LatticeBottom
    
    def leq(self, t1, t2) -> bool:
        if t1 is LatticeBottom or t2 is LatticeTop:
            return True
        if t1 is LatticeTop or t2 is LatticeBottom:
            return t2 is LatticeTop
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


def resolve_scalar_expr(expr, env: ScalarEnv):
    if isinstance(expr, Access):
        if expr.name not in env:
            raise AtomicCompilerError(f"Undefined variable: {expr.name}")
        bound = env[expr.name]
        if bound is LatticeTop:
            return expr
        if isinstance(bound, Access):
            return resolve_scalar_expr(bound, env)
        return bound
    
    if isinstance(expr, MATH_BOOL_TYPES):
        return expr.__class__(
            expr1=resolve_scalar_expr(expr.expr1, env),
            expr2=resolve_scalar_expr(expr.expr2, env),
        )
        
    if isinstance(expr, MathFunc):
        if isinstance(expr.expr, list):
            return MathFunc(
                func=expr.func,
                expr=[resolve_scalar_expr(e, env) for e in expr.expr],
            )
        return MathFunc(func=expr.func, expr=resolve_scalar_expr(expr.expr, env))
    
    if isinstance(expr, BoolNot):
        return BoolNot(expr=resolve_scalar_expr(expr.expr, env))
    
    if isinstance(expr, (MathNum, MathVar, MathImag, Bool)):
        return expr

    raise AtomicCompilerError(f"Cannot resolve scalar expression: {type(expr).__name__}")


def resolve_beam_expr(beam: Beam, env: ScalarEnv) -> Beam:
    pol, wv = beam.polarization, beam.wavevector
    return Beam(
        frequency=resolve_scalar_expr(beam.frequency, env),
        rabi=resolve_scalar_expr(beam.rabi, env),
        phase=resolve_scalar_expr(beam.phase, env),
        polarization=AtomicList(values=[resolve_scalar_expr(v, env) for v in pol.values])
        if isinstance(pol, AtomicList) else resolve_scalar_expr(pol, env),
        wavevector=AtomicList(values=[resolve_scalar_expr(v, env) for v in wv.values])
        if isinstance(wv, AtomicList) else resolve_scalar_expr(wv, env),
    )

def resolve_beam_ref(expr, env: ScalarEnv):
    if isinstance(expr, Access):
        if expr.name not in env:
            raise AtomicCompilerError(f"Undefined variable: {expr.name}")
        bound = env[expr.name]
        if isinstance(bound, Access):
            return resolve_beam_ref(bound, env)
        if not isinstance(bound, Beam):
            raise AtomicCompilerError(f"Access {expr.name} is not a beam")
        return bound.model_copy(deep=True)
    if isinstance(expr, Beam):
        return resolve_beam_expr(expr, env)
    raise AtomicCompilerError(f"Cannot resolve beam expression: {type(expr).__name__}")


def resolve_pulse_expr(pulse: Pulse, env: ScalarEnv) -> Pulse:
    return Pulse(
        beam=canonicalize_beam(resolve_beam_ref(pulse.beam, env)),
        duration=canonicalize_scalar_expr(resolve_scalar_expr(pulse.duration, env)),
        target=canonicalize_expr(pulse.target),
        measured=resolve_scalar_expr(pulse.measured, env),
    )


def resolve_pulse_ref(expr, env: ScalarEnv):
    if isinstance(expr, Access):
        if expr.name not in env:
            raise AtomicCompilerError(f"Undefined variable: {expr.name}")
        bound = env[expr.name]
        if isinstance(bound, Access):
            return resolve_pulse_ref(bound, env)
        if not isinstance(bound, Pulse):
            raise AtomicCompilerError(f"Access {expr.name} is not a pulse")
        return bound.model_copy(deep=True)
    
    if isinstance(expr, Pulse):
        return resolve_pulse_expr(expr, env)
    raise AtomicCompilerError(f"Cannot resolve pulse expression: {type(expr).__name__}")


def resolve_protocol_pulses(pulses, env: ScalarEnv):
    resolved = []
    for child in pulses:
        if isinstance(child, Access):
            child = resolve_pulse_ref(child, env)
        elif isinstance(child, Pulse):
            child = resolve_pulse_expr(child, env)
        elif isinstance(child, (ParallelProtocol, SerialProtocol)):
            child = child.__class__(
                pulses=resolve_protocol_pulses(child.pulses, env)
            )
        resolved.append(child)
    return resolved


def canonicalize_scalar_expr(expr):
    if isinstance(expr, Bool):
        return expr
    return canonicalize_math_expr(expr)


class ScalarEnvBuilder(ForwardDataflowAnalysis[int, ScalarEnv]):
    def __init__(
        self,
        graph: ControlFlowGraph,
        type_result: DataflowResult[int, TypeEnv],
    ) -> None:
        self.type_out_states = type_result.out_states
        self.lattice = maplattice(ScalarExprLattice)()
        self.blocks = graph.blocks
        self.analyze(graph, self.merge_scalar_env)
        
    def merge_scalar_env(self, states: Iterable[ScalarEnv]) -> ScalarEnv:
        states_list = list(states)
        if not states_list:
            return self.lattice.bottom()
        
        merged = {} if states_list[0] is LatticeBottom else dict(states_list[0])
        for state in states_list[1:]:
            if state is LatticeBottom:
                continue
            for name in set(merged).union(state):
                v1, v2 = merged.get(name), state.get(name)
                if v1 is None:
                    merged[name] = v2
                elif v2 is None:
                    continue
                elif v1 is LatticeTop or v2 is LatticeTop or v1 != v2:
                    merged[name] = LatticeTop
        return merged
        
    def transfer(self, node_id: int, state_in: ScalarEnv) -> ScalarEnv:
        env = {} if state_in is LatticeBottom else dict(state_in)
        stmt = self.blocks[node_id].stmt
        block = self.blocks[node_id]

        if block.kind == "branch":
            block.stmt = resolve_scalar_expr(stmt, env)
            return env
        
        if isinstance(stmt, Declaration):
            t: TLatticeValue | None = self.type_out_states[node_id].get(stmt.name)
            if t is TScalar:
                bound = canonicalize_scalar_expr(
                    resolve_scalar_expr(stmt.value, env)
                )
                out = dict(env)
                out[stmt.name] = bound
                return out
            if t is TBool:
                bound = resolve_scalar_expr(stmt.value, env)
                out = dict(env)
                out[stmt.name] = bound
                return out
            if t is TBeam:
                beam = canonicalize_beam(resolve_beam_expr(stmt.value, env))
                stmt.value = beam
                out = dict(env)
                out[stmt.name] = beam
                return out
            if t is TPulse:
                pulse = resolve_pulse_expr(stmt.value, env)
                stmt.value = pulse
                out = dict(env)
                out[stmt.name] = pulse
                return out
            return env
            
        if isinstance(stmt, Pulse):
            resolved = resolve_pulse_expr(stmt, env)
            stmt.beam = resolved.beam
            stmt.duration = resolved.duration
            stmt.target = resolved.target
            stmt.measured = resolved.measured
            return env
        
        if isinstance(stmt, Access):
            block.stmt = resolve_pulse_ref(stmt, env)
            return env
        
        if isinstance(stmt, (ParallelProtocol, SerialProtocol)):
            stmt.pulses = resolve_protocol_pulses(stmt.pulses, env)
            return env
        
        return env


def canonicalize_scalars_cfg(
    cfg: ControlFlowGraph,
    type_result: DataflowResult[int, TypeEnv],
) -> ControlFlowGraph:
    ScalarEnvBuilder(cfg, type_result)
    return cfg

