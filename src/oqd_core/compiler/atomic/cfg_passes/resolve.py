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

from oqd_compiler_infrastructure.lattice import LatticeTop

from oqd_core.compiler.atomic.cfg_passes.walk import canonicalize_beam, canonicalize_expr, canonicalize_scalar_expr
from oqd_core.compiler.atomic.error import AtomicCompilerError
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

