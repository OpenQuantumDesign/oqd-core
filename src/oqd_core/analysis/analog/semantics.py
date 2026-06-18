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

from oqd_core.analysis.analog.types import (
    BIN_SIG_TABLE,
    OP_TABLE,
    OPMUL_ALLOWED,
    AnalogTypeError,
    AnalogTypeLattice,
    TAnalog,
    TBool,
    TLatticeValue,
    TList,
    TMRef,
    TMReg,
    TOp,
    TQRef,
    TQReg,
    TScalar,
    TTarget,
    TTargetRef,
    TypeEnv,
    type_name,
)
from oqd_core.analysis.utils.control_flow import alias_types
from oqd_core.interface.analog import (
    Access,
    AnalogExprSubtypes,
    AnalogList,
    Bool,
    BoolEq,
    BoolNot,
    BoolNotEq,
    Evolve,
    Extract,
    Initialize,
    MathFunc,
    MathImag,
    MathNum,
    MathVar,
    Measure,
    ModeRegister,
    OperatorMul,
    PauliI,
    PauliX,
    PauliY,
    PauliZ,
    QuantumRegister,
)
from oqd_core.interface.analog.expr import Annihilation, Creation, Identity, Terminal

########################################################################################

EXPR_NODE_TYPES = alias_types(AnalogExprSubtypes)
TERMINAL_NODE_TYPES = alias_types(Terminal)

MATH_FUNCS =  {
    "abs", "sin", "cos", "tan", "exp", "log",
    "sinh", "cosh", "tanh", "atan", "acos", "asin",
    "atanh", "asinh", "acosh", "heaviside", "conj", "real", "imag",
}

class AnalogSemantics:
    def __init__(self, value_lattice: AnalogTypeLattice):
        self.value_lattice = value_lattice

    def infer_type(self, expr, env: TypeEnv) -> TLatticeValue:
        if not isinstance(expr, EXPR_NODE_TYPES):
            raise AnalogTypeError(f"Unsupported expression node: {type(expr).__name__}")

        if isinstance(expr, TERMINAL_NODE_TYPES):
            if isinstance(expr, (MathNum, MathVar, MathImag)):
                return TScalar
            if isinstance(expr, Bool):
                return TBool
            if isinstance(expr, (PauliI, PauliX, PauliY, PauliZ, Creation, Annihilation, Identity)):
                return TOp
            if isinstance(expr, QuantumRegister):
                return TQReg
            if isinstance(expr, ModeRegister):
                return TMReg
            if isinstance(expr, Access):
                if expr.name not in env:
                    raise AnalogTypeError(f"Undefined variable: {expr.name}")
                return env[expr.name]
    
        if isinstance(expr, AnalogList):
            if not expr.values:
                return TList(elem=LatticeBottom)
            
            t = self.infer_type(expr.values[0], env)
            for v in expr.values[1:]:
                t = self.value_lattice.join(t, self.infer_type(v, env))
            return TList(elem=t)
        
        if isinstance(expr, Extract):
            if expr.access.name not in env:
                raise AnalogTypeError(f"Undefined variable: {expr.access.name}")
            base = env[expr.access.name]
            if base is TQReg:
                return TQRef
            if base is TMReg:
                return TMRef
            if isinstance(base, TList):
                return base.elem
            raise AnalogTypeError(f"Cannot index into {type_name(base)}")
        
        sig = BIN_SIG_TABLE.get(type(expr))
        if sig is not None:
            (lreq, rreq), out = sig
            t1 = self.infer_type(expr.expr1, env)
            t2 = self.infer_type(expr.expr2, env)
            if not self.value_lattice.leq(t1, lreq) or not self.value_lattice.leq(t2, rreq):
                raise AnalogTypeError(f"{type(expr).__name__} got {type_name(t1)}, {type_name(t2)} expected {type_name(lreq)}, {type_name(rreq)}")
            return out
        
        sig = OP_TABLE.get(type(expr))
        if sig is not None:
            (lreq, rreq), out = sig
            t1 = self.infer_type(expr.op1, env)
            t2 = self.infer_type(expr.op2, env)
            if not self.value_lattice.leq(t1, lreq) or not self.value_lattice.leq(t2, rreq):
                raise AnalogTypeError(f"{type(expr).__name__} got {type_name(t1)}, {type_name(t2)} expected {type_name(lreq)}, {type_name(rreq)}")
            return out
        
        if isinstance(expr, MathFunc):
            if expr.func in MATH_FUNCS:
                arg = expr.expr
                t = self.infer_type(arg, env)
                if not self.value_lattice.leq(t, TScalar):
                    raise AnalogTypeError(f"{expr.func} expects scalar, got {type_name(t)}")
                return TScalar
            
            if expr.func == "atan2":
                arg = expr.expr
                if len(arg) != 2:
                    raise AnalogTypeError("atan2 expects exactly 2 arguments")
                t1 = self.infer_type(arg[0], env)
                t2 = self.infer_type(arg[1], env)
                if not self.value_lattice.leq(t1, TScalar) or not self.value_lattice.leq(t2, TScalar):
                    raise AnalogTypeError(f"{expr.func} expects scalar, got {type_name(t1)}, {type_name(t2)}")
                return TScalar
            
            raise AnalogTypeError(f"Unsupported math function: {expr.func}")
        
        if isinstance(expr, OperatorMul):
            t1 = self.infer_type(expr.op1, env)
            t2 = self.infer_type(expr.op2, env)
            out = OPMUL_ALLOWED.get((t1, t2))
            if out is None:
                raise AnalogTypeError(f"{type(expr).__name__} expects operator or scalar, got {type_name(t1)}, {type_name(t2)}")
            return out
        
        if isinstance(expr, (BoolEq, BoolNotEq)):
            t1 = self.infer_type(expr.expr1, env)
            t2 = self.infer_type(expr.expr2, env)
            if t1 not in (TBool, TScalar) or t2 not in (TBool, TScalar):
                raise AnalogTypeError(f"{type(expr).__name__} expects bool or scalar, got {type_name(t1)}, {type_name(t2)}")
            if t1 is not t2:
                raise AnalogTypeError(f"{type(expr).__name__}: got {type_name(t1)} vs {type_name(t2)}")
            return TBool
        
        if isinstance(expr, BoolNot):
            t = self.infer_type(expr.expr, env)
            if not self.value_lattice.leq(t, TBool):
                raise AnalogTypeError(f"{type(expr).__name__} expects bool, got {type_name(t)}")
            return TBool
        
        if isinstance(expr, (Initialize, Measure)):
            t = self.infer_type(expr.targets, env)
            if isinstance(t, TList):
                if not self.value_lattice.leq(t.elem, TTargetRef):
                    raise AnalogTypeError(f"{type(expr).__name__} expects Quantum targets, got {type_name(t)}")
            elif not self.value_lattice.leq(t, TTarget):
                raise AnalogTypeError(f"{type(expr).__name__} expects Quantum targets, got {type_name(t)}")
            return TAnalog
        
        if isinstance(expr, Evolve):
            target_t = self.infer_type(expr.targets, env)
            if isinstance(target_t, TList):
                if not self.value_lattice.leq(target_t.elem, TTargetRef):
                    raise AnalogTypeError(f"{type(expr).__name__} expects Quantum targets, got {type_name(target_t)}")
            elif not self.value_lattice.leq(target_t, TTarget):
                raise AnalogTypeError(f"{type(expr).__name__} expects Quantum targets, got {type_name(target_t)}")
            
            duration_t = self.infer_type(expr.duration, env)
            if not self.value_lattice.leq(duration_t, TScalar):
                raise AnalogTypeError(f"{type(expr).__name__} expects scalar duration, got {type_name(duration_t)}")
            
            hamiltonian_t = self.infer_type(expr.hamiltonian, env)
            if not self.value_lattice.leq(hamiltonian_t, TOp):
                raise AnalogTypeError(f"{type(expr).__name__} expects operator hamiltonian, got {type_name(hamiltonian_t)}")
            
            return TAnalog
        
        raise AnalogTypeError(f"Unsupported expression node: {type(expr).__name__}")
        
