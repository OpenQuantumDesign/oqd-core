# Copyright 2024-2025 Open Quantum Design

# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from dataclasses import dataclass
from types import UnionType
from typing import Annotated, Iterable, Union, get_args, get_origin

from oqd_compiler_infrastructure.lattice import LatticeBase, LatticeBottom, LatticeTop

from oqd_core.analysis.utils import CFGNode
from oqd_core.interface.atomic import (
    Access,
    AtomicExprSubtypes,
    AtomicList,
    Beam,
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
    Break,
    Continue,
    Declaration,
    Extract,
    IonRegister,
    MathAdd,
    MathDiv,
    MathFunc,
    MathImag,
    MathMul,
    MathNum,
    MathPow,
    MathSub,
    MathVar,
    ParallelProtocol,
    Pulse,
    SerialProtocol,
    Terminal,
)

from .dataflow import (
    DataflowResult,
    MapForwardDataflowAnalysis,
)

########################################################################################

class AtomicTypeError(TypeError):
    """Type Error class for Atomic."""
    pass

def alias_types(alias):
    """Flatten `Annotated`/`Union` aliases into a tuple of concrete Python types."""
    origin = get_origin(alias)
    if origin is Annotated:
        return alias_types(get_args(alias)[0])
    
    if origin in (Union, UnionType):
        out: list[type] = []
        for arg in get_args(alias):
            out.extend(alias_types(arg))
        return tuple(dict.fromkeys(out))
    
    if isinstance(alias, type):
        return (alias,)
    return ()

EXPR_NODE_TYPES = alias_types(AtomicExprSubtypes)
TERMINAL_NODE_TYPES = alias_types(Terminal)

########################################################################################

@dataclass
class TList(LatticeTop):
    """Lattice value representing a list."""
    elem: LatticeValue

LatticeValue = Union[TList, type[LatticeTop]]

def type_name(t: LatticeValue) -> str:
    """Format a lattice value into a readable type name for error messages."""
    if isinstance(t, TList):
        return f"TList[{type_name(t.elem)}]"
    if isinstance(t, type) and issubclass(t, LatticeTop):
        return t.__name__
    return str(t)


class TAtomic(LatticeTop):
    pass

class TScalar(TAtomic):
    pass

class TBool(TAtomic):
    pass

class TBeam(TAtomic):
    pass

class TPulse(TAtomic):
    pass

class TTarget(TAtomic):
    pass

class TTargetRef(TTarget):
    pass

class TIonReg(TTarget):
    pass


class TIonRef(TTargetRef):
    pass


class AtomicTypeLattice(LatticeBase[LatticeValue]):
    """Type lattice for atomic expressions."""
    def __init__(self):
        super().__init__()
        self.add_node(TAtomic, LatticeTop)
        self.add_node(TScalar, TAtomic)
        self.add_node(TBool, TAtomic)
        self.add_node(TBeam, TAtomic)
        self.add_node(TPulse, TAtomic)
        self.add_node(TTarget, TAtomic)
        self.add_node(TTargetRef, TTarget)
        self.add_node(TIonReg, TTarget)
        self.add_node(TIonRef, TTargetRef)
    
    def leq(self, t1: LatticeValue, t2: LatticeValue) -> bool:
        if t1 is LatticeBottom:
            return True
        if isinstance(t1, TList) and isinstance(t2, TList):
            return self.leq(t1.elem, t2.elem)
        if isinstance(t1, TList) or isinstance(t2, TList):
            return False
        return super().leq(t1, t2)
    
    def join(self, t1: LatticeValue, t2: LatticeValue) -> LatticeValue:
        if self.leq(t1, t2):
            return t2
        if self.leq(t2, t1):
            return t1
        if isinstance(t1, TList) and isinstance(t2, TList):
            return TList(elem=self.lattice.join(t1.elem, t2.elem))
        if isinstance(t1, TList) or isinstance(t2, TList):
            return TAtomic
        return super().join(t1, t2)
    
    def meet(self, t1: LatticeValue, t2: LatticeValue) -> LatticeValue:
        if self.leq(t1, t2):
            return t1
        if self.leq(t2, t1):
            return t2
        if isinstance(t1, TList) and isinstance(t2, TList):
            return TList(elem=self.lattice.meet(t1.elem, t2.elem))
        return super().meet(t1, t2)


########################################################################################


# Binary expression signature table: node -> ((left_type, right_type), output_type)
BIN_SIG_TABLE = {
    MathAdd: ((TScalar, TScalar), TScalar),
    MathSub: ((TScalar, TScalar), TScalar),
    MathMul: ((TScalar, TScalar), TScalar),
    MathDiv: ((TScalar, TScalar), TScalar),
    MathPow: ((TScalar, TScalar), TScalar),

    BoolAnd: ((TBool, TBool), TBool),
    BoolOr: ((TBool, TBool), TBool),

    BoolLessThan: ((TScalar, TScalar), TBool),
    BoolLessThanEq: ((TScalar, TScalar), TBool),
    BoolGreaterThan: ((TScalar, TScalar), TBool),
    BoolGreaterThanEq: ((TScalar, TScalar), TBool),
}


########################################################################################


class AtomicCFG:
    def __init__(self, cfg: CFGNode):
        self.cfg = cfg
    def nodes(self) -> Iterable[int]:
        return self.cfg.keys()
    def predecessors(self, node: int) -> Iterable[int]:
        return (pred.register_id for pred in self.cfg[node].preds)
    def successors(self, node: int) -> Iterable[int]:
        return (succ.register_id for succ in self.cfg[node].succs)


########################################################################################


class AtomicTypeChecker(MapForwardDataflowAnalysis[int, LatticeValue]):
    """Forward dataflow type checker over the CFG."""
    def __init__(self):
        self.lattice = AtomicTypeLattice()
        super().__init__(self.lattice)
        self.cfg: CFGNode | None = None
        
    def leq(self, t1: LatticeValue, t2: LatticeValue) -> bool:
        return self.lattice.leq(t1, t2)
        
    
    def transfer(self, node_id: int, state_in: dict[str, LatticeValue]) -> dict[str, LatticeValue]:
        cfg_node = self.cfg[node_id]
        stmt = cfg_node.stmt
        if isinstance(stmt, str):
            return dict(state_in)
        if cfg_node.kind == "branch":
            condition_t = self.infer_expr(stmt, state_in)
            if condition_t is not TBool:
                raise AtomicTypeError("branch condition must be bool")
            return dict(state_in)
        if isinstance(stmt, Declaration):
            state_out = dict(state_in)
            state_out[stmt.name] = self.infer_expr(stmt.value, state_in)
            return state_out
        if isinstance(stmt, (Break, Continue)):
            return dict(state_in)
        if isinstance(stmt, (ParallelProtocol, SerialProtocol)):
            stack = [stmt]
            while stack:
                curr = stack.pop()
                if isinstance(curr, (ParallelProtocol, SerialProtocol)):
                    stack.extend(reversed(curr.pulses))
                    continue
                t = self.infer_expr(curr, state_in)
                if not self.leq(t, TPulse):
                    raise AtomicTypeError(
                        f"Parallel/Serial blocks expect only Pulse statements, got {type_name(t)}"
                    )
            return dict(state_in)
        self.infer_expr(stmt, state_in)
        return dict(state_in)
    
    
    def analyze_dataflow(self, cfg: CFGNode) -> DataflowResult[int, dict[str, LatticeValue]]:
        self.cfg = cfg
        try:
            return self.analyze(AtomicCFG(cfg))
        except Exception as e:
            raise AtomicTypeError(f"Type checking failed during CFG / dataflow analysis: {e}")
        
    
    def infer_expr(self, expr: type, env: dict[str, LatticeValue]) -> dict[str, LatticeValue]:
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
            
            t = self.infer_expr(expr.values[0], env)
            for v in expr.values[1:]:
                t = self.lattice.join(t, self.infer_expr(v, env))
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
            t1 = self.infer_expr(expr.expr1, env)
            t2 = self.infer_expr(expr.expr2, env)
            if not self.leq(t1, lreq) or not self.leq(t2, rreq):
                raise AtomicTypeError(f"{type(expr).__name__} got {type_name(t1)}, {type_name(t2)} expected {type_name(lreq)}, {type_name(rreq)}")
            return out
        
        if isinstance(expr, MathFunc):
            math_funcs =  {
                "abs", "sin", "cos", "tan", "exp", "log",
                "sinh", "cosh", "tanh", "atan", "acos", "asin",
                "atanh", "asinh", "acosh", "heaviside", "conj", "real", "imag",
            }
            if expr.func in math_funcs:
                arg = expr.expr
                t = self.infer_expr(arg, env)
                if not self.leq(t, TScalar):
                    raise AtomicTypeError(f"{expr.func} expects scalar, got {type_name(t)}")
                return TScalar
            
            if expr.func == "atan2":
                arg = expr.expr
                if len(arg) != 2:
                    raise AtomicTypeError("atan2 expects exactly 2 arguments")
                t1 = self.infer_expr(arg[0], env)
                t2 = self.infer_expr(arg[1], env)
                if not self.leq(t1, TScalar) or not self.leq(t2, TScalar):
                    raise AtomicTypeError(f"{expr.func} expects scalar, got {type_name(t1)}, {type_name(t2)}")
                return TScalar
            
            raise AtomicTypeError(f"Unsupported math function: {expr.func}")
        
        
        if isinstance(expr, (BoolEq, BoolNotEq)):
            t1 = self.infer_expr(expr.expr1, env)
            t2 = self.infer_expr(expr.expr2, env)
            if t1 not in (TBool, TScalar) or t2 not in (TBool, TScalar):
                raise AtomicTypeError(f"{type(expr).__name__} expects bool or scalar, got {type_name(t1)}, {type_name(t2)}")
            if t1 is not t2:
                raise AtomicTypeError(f"{type(expr).__name__}: got {type_name(t1)} vs {type_name(t2)}")
            return TBool
        
        if isinstance(expr, BoolNot):
            t = self.infer_expr(expr.expr, env)
            if not self.leq(t, TBool):
                raise AtomicTypeError(f"{type(expr).__name__} expects bool, got {type_name(t)}")
            return TBool
        
        
        if isinstance(expr, Beam):
            if isinstance(expr, Beam):
                freq_t = self.infer_expr(expr.frequency, env)
                if not self.leq(freq_t, TScalar):
                    raise AtomicTypeError(f"{type(expr).__name__} expects scalar frequency, got {type_name(freq_t)}")
                
                rabi_t = self.infer_expr(expr.rabi, env)
                if not self.leq(rabi_t, TScalar):
                    raise AtomicTypeError(f"{type(expr).__name__} expects scalar rabi, got {type_name(rabi_t)}")
                
                phase_t = self.infer_expr(expr.phase, env)
                if not self.leq(phase_t, TScalar):
                    raise AtomicTypeError(f"{type(expr).__name__} expects scalar phase, got {type_name(phase_t)}")
                
                pol_t = self.infer_expr(expr.polarization, env)
                if isinstance(pol_t, TList):
                    if not self.leq(pol_t.elem, TScalar):
                        raise AtomicTypeError(f"{type(expr).__name__} expects scalar polarization components, got {type_name(pol_t)}")
                elif not self.leq(pol_t, TScalar):
                    raise AtomicTypeError(f"{type(expr).__name__} expects scalar polarization, got {type_name(pol_t)}")
                
                wave_t = self.infer_expr(expr.wavevector, env)
                if isinstance(wave_t, TList):
                    if not self.leq(wave_t.elem, TScalar):
                        raise AtomicTypeError(f"{type(expr).__name__} expects scalar wavevector components, got {type_name(wave_t)}")
                elif not self.leq(wave_t, TScalar):
                    raise AtomicTypeError(f"{type(expr).__name__} expects scalar wavevector, got {type_name(wave_t)}")
                    
            return TBeam
        
        
        if isinstance(expr, Pulse):
            duration_t = self.infer_expr(expr.duration, env)
            if not self.leq(duration_t, TScalar):
                raise AtomicTypeError(f"{type(expr).__name__} expects scalar duration, got {type_name(duration_t)}")
            target_t = self.infer_expr(expr.target, env)
            if isinstance(target_t, TList):
                if not self.leq(target_t.elem, TTargetRef):
                    raise AtomicTypeError(f"{type(expr).__name__} expects ion targets, got {type_name(target_t)}")
            elif not self.leq(target_t, TTarget):
                raise AtomicTypeError(f"{type(expr).__name__} expects ion targets, got {type_name(target_t)}")
            beam_t = self.infer_expr(expr.beam, env)
            if not self.leq(beam_t, TBeam):
                raise AtomicTypeError(f"{type(expr).__name__} expects beam expression, got {type_name(beam_t)}")
            measured_t = self.infer_expr(expr.measured, env)
            if not self.leq(measured_t, TBool):
                raise AtomicTypeError(f"{type(expr).__name__} expects bool measured flag, got {type_name(measured_t)}")

            return TPulse
        





