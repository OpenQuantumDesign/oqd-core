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
from typing import Union

from oqd_compiler_infrastructure.dataflow import (
    DataflowResult,
    MapForwardDataflowAnalysis,
)
from oqd_compiler_infrastructure.lattice import LatticeBase, LatticeBottom, LatticeTop

from oqd_core.analysis.utils import ControlFlowGraph, alias_types
from oqd_core.interface.analog import (
    Access,
    AnalogExprSubtypes,
    AnalogList,
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
    Evolve,
    Extract,
    Initialize,
    MathAdd,
    MathDiv,
    MathFunc,
    MathImag,
    MathMul,
    MathNum,
    MathPow,
    MathSub,
    MathVar,
    Measure,
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
from oqd_core.interface.analog.expr import Annihilation, Creation, Identity, Terminal

########################################################################################

class AnalogTypeError(TypeError):
    """Type Error class for Analog."""
    pass

EXPR_NODE_TYPES = alias_types(AnalogExprSubtypes)
TERMINAL_NODE_TYPES = alias_types(Terminal)

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


class TAnalog(LatticeTop):
    pass

class TScalar(TAnalog):
    pass

class TBool(TAnalog):
    pass

class TOp(TAnalog):
    pass

class TTarget(TAnalog):
    pass

class TTargetRef(TTarget):
    pass

class TQReg(TTarget):
    pass

class TMReg(TTarget):
    pass

class TQRef(TTargetRef):
    pass

class TMRef(TTargetRef):
    pass


class AnalogTypeLattice(LatticeBase[LatticeValue]):
    """Type lattice for analog expressions."""
    def __init__(self):
        super().__init__()
        self.add_node(TAnalog, LatticeTop)
        self.add_node(TScalar, TAnalog)
        self.add_node(TBool, TAnalog)
        self.add_node(TOp, TAnalog)
        self.add_node(TTarget, TAnalog)
        self.add_node(TTargetRef, TTarget)
        self.add_node(TQReg, TTarget)
        self.add_node(TMReg, TTarget)
        self.add_node(TQRef, TTargetRef)
        self.add_node(TMRef, TTargetRef)
    
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
            return TList(elem=self.join(t1.elem, t2.elem))
        if isinstance(t1, TList) or isinstance(t2, TList):
            return TAnalog
        return super().join(t1, t2)
    
    def meet(self, t1: LatticeValue, t2: LatticeValue) -> LatticeValue:
        if self.leq(t1, t2):
            return t1
        if self.leq(t2, t1):
            return t2
        if isinstance(t1, TList) and isinstance(t2, TList):
            return TList(elem=self.meet(t1.elem, t2.elem))
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


# Operator expression signatures
OP_TABLE = {
    OperatorAdd: ((TOp, TOp), TOp),
    OperatorSub: ((TOp, TOp), TOp),
    OperatorKron: ((TOp, TOp), TOp),
}


# Allowed type pairs for OperatorMul
OPMUL_ALLOWED = {
    (TOp, TOp): TOp,
    (TOp, TScalar): TOp,
    (TScalar, TOp): TOp,
}


########################################################################################
  

class AnalogTypeChecker(MapForwardDataflowAnalysis[int, LatticeValue]):
    """Forward dataflow type checker over the analog CFG."""
    def __init__(self, graph: ControlFlowGraph) -> None:
        self.lattice = AnalogTypeLattice()
        super().__init__(self.lattice)
        self.cfg_nodes = graph.cfg_nodes
        self.result : DataflowResult[int, dict[str, LatticeValue]] | None = None
        try:
            self.result = self.analyze(graph)
        except Exception as e:
            raise AnalogTypeError(f"Type checking failed during CFG / dataflow analysis: {e}")
    
    
    def leq(self, t1: LatticeValue, t2: LatticeValue) -> bool:
        return self.lattice.leq(t1, t2)
    
    
    def transfer(self, node_id: int, state_in: dict[str, LatticeValue]) -> dict[str, LatticeValue]:
        cfg_node = self.cfg_nodes[node_id]
        stmt = cfg_node.stmt
        if isinstance(stmt, str):
            return dict(state_in)
        if cfg_node.kind == "branch":
            condition_t = self.infer_expr(stmt, state_in)
            if condition_t is not TBool:
                raise AnalogTypeError("branch condition must be bool")
            return dict(state_in)
        if isinstance(stmt, Declaration):
            state_out = dict(state_in)
            state_out[stmt.name] = self.infer_expr(stmt.value, state_in)
            return state_out
        if isinstance(stmt, (Break, Continue)):
            return dict(state_in)
        self.infer_expr(stmt, state_in)
        return dict(state_in)

        
    def infer_expr(self, expr: type, env: dict[str, LatticeValue]) -> dict[str, LatticeValue]:
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
            
            t = self.infer_expr(expr.values[0], env)
            for v in expr.values[1:]:
                t = self.lattice.join(t, self.infer_expr(v, env))
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
            t1 = self.infer_expr(expr.expr1, env)
            t2 = self.infer_expr(expr.expr2, env)
            if not self.leq(t1, lreq) or not self.leq(t2, rreq):
                raise AnalogTypeError(f"{type(expr).__name__} got {type_name(t1)}, {type_name(t2)} expected {type_name(lreq)}, {type_name(rreq)}")
            return out
        
        sig = OP_TABLE.get(type(expr))
        if sig is not None:
            (lreq, rreq), out = sig
            t1 = self.infer_expr(expr.op1, env)
            t2 = self.infer_expr(expr.op2, env)
            if not self.leq(t1, lreq) or not self.leq(t2, rreq):
                raise AnalogTypeError(f"{type(expr).__name__} got {type_name(t1)}, {type_name(t2)} expected {type_name(lreq)}, {type_name(rreq)}")
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
                    raise AnalogTypeError(f"{expr.func} expects scalar, got {type_name(t)}")
                return TScalar
            
            if expr.func == "atan2":
                arg = expr.expr
                if len(arg) != 2:
                    raise AnalogTypeError("atan2 expects exactly 2 arguments")
                t1 = self.infer_expr(arg[0], env)
                t2 = self.infer_expr(arg[1], env)
                if not self.leq(t1, TScalar) or not self.leq(t2, TScalar):
                    raise AnalogTypeError(f"{expr.func} expects scalar, got {type_name(t1)}, {type_name(t2)}")
                return TScalar
            
            raise AnalogTypeError(f"Unsupported math function: {expr.func}")
        
        if isinstance(expr, OperatorMul):
            t1 = self.infer_expr(expr.op1, env)
            t2 = self.infer_expr(expr.op2, env)
            out = OPMUL_ALLOWED.get((t1, t2))
            if out is None:
                raise AnalogTypeError(f"{type(expr).__name__} expects operator or scalar, got {type_name(t1)}, {type_name(t2)}")
            return out
        
        if isinstance(expr, (BoolEq, BoolNotEq)):
            t1 = self.infer_expr(expr.expr1, env)
            t2 = self.infer_expr(expr.expr2, env)
            if t1 not in (TBool, TScalar) or t2 not in (TBool, TScalar):
                raise AnalogTypeError(f"{type(expr).__name__} expects bool or scalar, got {type_name(t1)}, {type_name(t2)}")
            if t1 is not t2:
                raise AnalogTypeError(f"{type(expr).__name__}: got {type_name(t1)} vs {type_name(t2)}")
            return TBool
        
        if isinstance(expr, BoolNot):
            t = self.infer_expr(expr.expr, env)
            if not self.leq(t, TBool):
                raise AnalogTypeError(f"{type(expr).__name__} expects bool, got {type_name(t)}")
            return TBool
        
        if isinstance(expr, (Initialize, Measure)):
            t = self.infer_expr(expr.targets, env)
            if isinstance(t, TList):
                if not self.leq(t.elem, TTargetRef):
                    raise AnalogTypeError(f"{type(expr).__name__} expects Quantum targets, got {type_name(t)}")
            elif not self.leq(t, TTarget):
                raise AnalogTypeError(f"{type(expr).__name__} expects Quantum targets, got {type_name(t)}")
            return TAnalog
        
        if isinstance(expr, Evolve):
            target_t = self.infer_expr(expr.targets, env)
            if isinstance(target_t, TList):
                if not self.leq(target_t.elem, TTargetRef):
                    raise AnalogTypeError(f"{type(expr).__name__} expects Quantum targets, got {type_name(target_t)}")
            elif not self.leq(target_t, TTarget):
                raise AnalogTypeError(f"{type(expr).__name__} expects Quantum targets, got {type_name(target_t)}")
            
            duration_t = self.infer_expr(expr.duration, env)
            if not self.leq(duration_t, TScalar):
                raise AnalogTypeError(f"{type(expr).__name__} expects scalar duration, got {type_name(duration_t)}")
            
            hamiltonian_t = self.infer_expr(expr.hamiltonian, env)
            if not self.leq(hamiltonian_t, TOp):
                raise AnalogTypeError(f"{type(expr).__name__} expects operator hamiltonian, got {type_name(hamiltonian_t)}")
            
            return TAnalog
        
