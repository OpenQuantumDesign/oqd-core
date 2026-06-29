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

from oqd_compiler_infrastructure.lattice import LatticeBottom

from oqd_core.analysis.atomic.types import (
    BIN_SIG_TABLE,
    AtomicTypeError,
    AtomicTypeLattice,
    TBeam,
    TBool,
    TIonRef,
    TIonReg,
    TLatticeValue,
    TList,
    TPulse,
    TScalar,
    TTarget,
    TTargetRef,
    TypeEnv,
    type_name,
)
from oqd_core.analysis.utils.control_flow import alias_types
from oqd_core.interface.atomic import (
    Access,
    AtomicExprSubtypes,
    AtomicList,
    Beam,
    Bool,
    BoolEq,
    BoolNot,
    BoolNotEq,
    Extract,
    IonRegister,
    MathFunc,
    MathImag,
    MathNum,
    MathVar,
    Pulse,
)
from oqd_core.interface.atomic.expr import Terminal

########################################################################################

EXPR_NODE_TYPES = alias_types(AtomicExprSubtypes)
TERMINAL_NODE_TYPES = alias_types(Terminal)

MATH_FUNCS =  {
    "abs", "sin", "cos", "tan", "exp", "log",
    "sinh", "cosh", "tanh", "atan", "acos", "asin",
    "atanh", "asinh", "acosh", "heaviside", "conj", "real", "imag",
}

class AtomicSemantics:
    def __init__(self, value_lattice: AtomicTypeLattice):
        self.value_lattice = value_lattice

    def infer_type(self, expr: type, env: TypeEnv) -> TLatticeValue:
        if not isinstance(expr, EXPR_NODE_TYPES):
            raise AtomicTypeError(f"Unsupported expression node: {type(expr).__name__}")

        if isinstance(expr, TERMINAL_NODE_TYPES):
            if isinstance(expr, (MathNum, MathVar, MathImag)):
                return TScalar
            if isinstance(expr, Bool):
                return TBool
            if isinstance(expr, IonRegister):
                return TIonReg
            if isinstance(expr, Access):
                if expr.name not in env:
                    raise AtomicTypeError(f"Undefined variable: {expr.name}")
                return env[expr.name]
    
        if isinstance(expr, AtomicList):
            if not expr.values:
                return TList(elem=LatticeBottom)
            
            t = self.infer_type(expr.values[0], env)
            for v in expr.values[1:]:
                t = self.value_lattice.join(t, self.infer_type(v, env))
            return TList(elem=t)
        
        if isinstance(expr, Extract):
            if expr.access.name not in env:
                raise AtomicTypeError(f"Undefined variable: {expr.access.name}")
            base = env[expr.access.name]
            if base is TIonReg:
                return TIonRef
            if isinstance(base, TList):
                return base.elem
            raise AtomicTypeError(f"Cannot index into {type_name(base)}")
        
        sig = BIN_SIG_TABLE.get(type(expr))
        if sig is not None:
            (lreq, rreq), out = sig
            t1 = self.infer_type(expr.expr1, env)
            t2 = self.infer_type(expr.expr2, env)
            if not self.value_lattice.leq(t1, lreq) or not self.value_lattice.leq(t2, rreq):
                raise AtomicTypeError(f"{type(expr).__name__} got {type_name(t1)}, {type_name(t2)} expected {type_name(lreq)}, {type_name(rreq)}")
            return out
        
        if isinstance(expr, MathFunc):
            if expr.func in MATH_FUNCS:
                arg = expr.expr
                t = self.infer_type(arg, env)
                if not self.value_lattice.leq(t, TScalar):
                    raise AtomicTypeError(f"{expr.func} expects scalar, got {type_name(t)}")
                return TScalar
            
            if expr.func == "atan2":
                arg = expr.expr
                if len(arg) != 2:
                    raise AtomicTypeError("atan2 expects exactly 2 arguments")
                t1 = self.infer_type(arg[0], env)
                t2 = self.infer_type(arg[1], env)
                if not self.value_lattice.leq(t1, TScalar) or not self.value_lattice.leq(t2, TScalar):
                    raise AtomicTypeError(f"{expr.func} expects scalar, got {type_name(t1)}, {type_name(t2)}")
                return TScalar
            
            raise AtomicTypeError(f"Unsupported math function: {expr.func}")
        
        
        if isinstance(expr, (BoolEq, BoolNotEq)):
            t1 = self.infer_type(expr.expr1, env)
            t2 = self.infer_type(expr.expr2, env)
            if t1 not in (TBool, TScalar) or t2 not in (TBool, TScalar):
                raise AtomicTypeError(f"{type(expr).__name__} expects bool or scalar, got {type_name(t1)}, {type_name(t2)}")
            if t1 is not t2:
                raise AtomicTypeError(f"{type(expr).__name__}: got {type_name(t1)} vs {type_name(t2)}")
            return TBool
        
        if isinstance(expr, BoolNot):
            t = self.infer_type(expr.expr, env)
            if not self.value_lattice.leq(t, TBool):
                raise AtomicTypeError(f"{type(expr).__name__} expects bool, got {type_name(t)}")
            return TBool
        
        
        if isinstance(expr, Beam):
            if isinstance(expr, Beam):
                freq_t = self.infer_type(expr.frequency, env)
                if not self.value_lattice.leq(freq_t, TScalar):
                    raise AtomicTypeError(f"{type(expr).__name__} expects scalar frequency, got {type_name(freq_t)}")
                
                rabi_t = self.infer_type(expr.rabi, env)
                if not self.value_lattice.leq(rabi_t, TScalar):
                    raise AtomicTypeError(f"{type(expr).__name__} expects scalar rabi, got {type_name(rabi_t)}")
                
                phase_t = self.infer_type(expr.phase, env)
                if not self.value_lattice.leq(phase_t, TScalar):
                    raise AtomicTypeError(f"{type(expr).__name__} expects scalar phase, got {type_name(phase_t)}")
                
                pol_t = self.infer_type(expr.polarization, env)
                if isinstance(pol_t, TList):
                    if not self.value_lattice.leq(pol_t.elem, TScalar):
                        raise AtomicTypeError(f"{type(expr).__name__} expects scalar polarization components, got {type_name(pol_t)}")
                elif not self.value_lattice.leq(pol_t, TScalar):
                    raise AtomicTypeError(f"{type(expr).__name__} expects scalar polarization, got {type_name(pol_t)}")
                
                wave_t = self.infer_type(expr.wavevector, env)
                if isinstance(wave_t, TList):
                    if not self.value_lattice.leq(wave_t.elem, TScalar):
                        raise AtomicTypeError(f"{type(expr).__name__} expects scalar wavevector components, got {type_name(wave_t)}")
                elif not self.value_lattice.leq(wave_t, TScalar):
                    raise AtomicTypeError(f"{type(expr).__name__} expects scalar wavevector, got {type_name(wave_t)}")
                    
            return TBeam
        
        
        if isinstance(expr, Pulse):
            duration_t = self.infer_type(expr.duration, env)
            if not self.value_lattice.leq(duration_t, TScalar):
                raise AtomicTypeError(f"{type(expr).__name__} expects scalar duration, got {type_name(duration_t)}")
            target_t = self.infer_type(expr.target, env)
            if isinstance(target_t, TList):
                if not self.value_lattice.leq(target_t.elem, TTargetRef):
                    raise AtomicTypeError(f"{type(expr).__name__} expects ion targets, got {type_name(target_t)}")
            elif not self.value_lattice.leq(target_t, TTarget):
                raise AtomicTypeError(f"{type(expr).__name__} expects ion targets, got {type_name(target_t)}")
            beam_t = self.infer_type(expr.beam, env)
            if not self.value_lattice.leq(beam_t, TBeam):
                raise AtomicTypeError(f"{type(expr).__name__} expects beam expression, got {type_name(beam_t)}")
            measured_t = self.infer_type(expr.measured, env)
            if not self.value_lattice.leq(measured_t, TBool):
                raise AtomicTypeError(f"{type(expr).__name__} expects bool measured flag, got {type_name(measured_t)}")

            return TPulse
        
        raise AtomicTypeError(f"Unsupported expression node: {type(expr).__name__}")
        


