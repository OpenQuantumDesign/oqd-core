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

from dataclasses import dataclass, field
from types import UnionType
from typing import Annotated, Dict, Union, get_args, get_origin

from oqd_core.frontend.analog.cfg import gen_cfg
from oqd_core.interface.analog import (
    Access,
    AnalogCircuit,
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
    IfElse,
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
    While,
)
from oqd_core.interface.analog.expr import Annihilation, Creation, Identity, Terminal
from oqd_core.interface.analog.statement import Statement

########################################################################################

class AnalogTypeError(TypeError):
    pass

def alias_types(alias):
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

EXPR_NODE_TYPES = alias_types(AnalogExprSubtypes)
STATEMENT_NODE_TYPES = alias_types(Statement)
TERMINAL_NODE_TYPES = alias_types(Terminal)

########################################################################################

class TAnalog:
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

class TBottom(TAnalog):
    pass

@dataclass
class TList(TAnalog):
    elem: Union[TList, type[TAnalog]]

def type_name(t):
    if isinstance(t, TList):
        return f"TList[{type_name(t.elem)}]"
    if isinstance(t, type) and issubclass(t, TAnalog):
        return t.__name__
    return str(t)

PARENTS = {
    TBottom: [],
    TScalar: (TAnalog,),
    TBool: (TAnalog,),
    TOp: (TAnalog,),
    TTarget: (TAnalog,),
    TTargetRef: (TTarget,),
    TQReg: (TTarget,),
    TMReg: (TTarget,),
    TQRef: (TTargetRef,),
    TMRef: (TTargetRef,),
    TAnalog: (),
}

########################################################################################

@dataclass
class Scope:
    parent: Union[Scope, None] = None
    symbols: Dict[str, Union[type[TAnalog], TList]] = field(default_factory=dict)
    children: list[Scope] = field(default_factory=list)
    
    def lookup(self, name):
        scope = self
        while scope is not None:
            if name in scope.symbols:
                return scope.symbols[name]
            scope = scope.parent
        raise AnalogTypeError(f"Undefined variable: {name}")
            
    def declare(self, name: str, datatype: Union[type[TAnalog], TList]):
        self.symbols[name] = datatype
        
    def to_dict(self):
        if not self.symbols and not self.children:
            return None
        return {
            "symbols": {k: type_name(v) for k, v in self.symbols.items()},
            "children": [child.to_dict() for child in self.children],
        }


class AnalogTypeChecker:
    def __init__(self):
        self.root = Scope()
        self.scope = self.root
        
    
    def push_scope(self):
        child = Scope(parent=self.scope)
        self.scope.children.append(child)
        self.scope = child

    
    def pop_scope(self):
        assert self.scope.parent is not None
        self.scope = self.scope.parent
        
    
    def atomic_ancestors(self, t):
        if not issubclass(t, TAnalog):
            raise AnalogTypeError(f"Expected analog type class, got {t}")
        out = {t}
        stack = [t]
        while stack:
            curr = stack.pop()
            for p in PARENTS.get(curr, ()):
                if p not in out:
                    out.add(p)
                    stack.append(p)
        return out
    
    
    def leq(self, t1, t2):
        if isinstance(t1, TList) and isinstance(t2, TList):
            return self.leq(t1.elem, t2.elem)
        if isinstance(t1, TList) or isinstance(t2, TList):
            return False
        if not issubclass(t1, TAnalog) or not issubclass(t2, TAnalog):
            return False
        if t1 is TBottom or t1 is t2:
            return True
        return t2 in self.atomic_ancestors(t1)
    
    
    def join(self, t1, t2):
        if self.leq(t1, t2):
            return t2
        if self.leq(t2, t1):
            return t1
        if isinstance(t1, TList) and isinstance(t2, TList):
            return TList(elem=self.join(t1.elem, t2.elem))
        if isinstance(t1, TList) or isinstance(t2, TList):
            return TAnalog
        if not issubclass(t1, TAnalog) or not issubclass(t2, TAnalog):
            return TAnalog
        common_ancestors = self.atomic_ancestors(t1).intersection(self.atomic_ancestors(t2))
        if not common_ancestors:
            return TAnalog
        
        minimal_ancestors = set()
        for candidate in common_ancestors:
            smaller = any(
                other is not candidate and self.leq(other, candidate)
                for other in common_ancestors
            )
            if not smaller:
                minimal_ancestors.add(candidate)
        if len(minimal_ancestors) != 1:
            return TAnalog
        return next(iter(minimal_ancestors))
    
    
    def meet(self, t1, t2):
        if self.leq(t1, t2):
            return t1
        if self.leq(t2, t1):
            return t2
        if isinstance(t1, TList) and isinstance(t2, TList):
            return TList(elem=self.meet(t1.elem, t2.elem))
        return TBottom
    

    def merge_envs(self, pred_envs):
        if not pred_envs:
            return {}
        
        common_keys = set(pred_envs[0].keys())
        for env in pred_envs[1:]:
            common_keys &= set(env.keys())
            
        merged = {}
        for name in common_keys:
            t = pred_envs[0][name]
            for env in pred_envs[1:]:
                t = self.join(t, env[name])
            merged[name] = t
        
        return merged
        
    
    def transfer_node(self, node, in_env):
        stmt = node.stmt
        if isinstance(stmt, str):
            return dict(in_env)
        old_scope = self.scope
        self.scope = Scope(parent=None, symbols=dict(in_env))
        try:
            if node.kind == "branch":
                condition_t = self.infer_expr(stmt)
                if condition_t is not TBool:
                    raise AnalogTypeError("branch condition must be bool")
                return dict(in_env)
            if isinstance(stmt, Declaration):
                out_env = dict(in_env)
                out_env[stmt.name] = self.infer_expr(stmt.value)
                return out_env
            if isinstance(stmt, (Break, Continue)):
                return dict(in_env)
            self.infer_expr(stmt)
            return dict(in_env)
        finally:
            self.scope = old_scope
    
    
    def analyze_dataflow(self, circuit: AnalogCircuit):
        cfg = gen_cfg(circuit)
        in_state = {nid: None for nid in cfg}
        out_state = {nid: None for nid in cfg}
        
        worklist = list(cfg.keys())
        
        while worklist:
            nid = worklist.pop(0)
            node = cfg[nid]
            
            if node.kind == "start":
                new_in = {}
            else:
                pred_outs = [
                    out_state[p.register_id] 
                    for p in node.preds
                    if out_state[p.register_id] is not None
                ]
                new_in = self.merge_envs(pred_outs) if pred_outs else {}
                
            new_out = self.transfer_node(node, new_in)
            
            if in_state[nid] != new_in or out_state[nid] != new_out:
                in_state[nid] = new_in
                out_state[nid] = new_out
                
                for succ in node.succs:
                    sid = succ.register_id
                    if sid not in worklist:
                        worklist.append(sid)
        
        return {
            "cfg": {nid: node.to_dict() for nid, node in cfg.items()},
            "in": {
                nid: {name: type_name(t) for name, t in env.items()}
                for nid, env in in_state.items()
            },
            "out": {
                nid: {name: type_name(t) for name, t in env.items()}
                for nid, env in out_state.items()
            },
        }
        
    
    def infer_expr(self, expr):
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
                return self.scope.lookup(expr.name)
    
        if isinstance(expr, AnalogList):
            if not expr.values:
                return TList(elem=TBottom)
            
            head = self.infer_expr(expr.values[0])
            for v in expr.values[1:]:
                t = self.infer_expr(v)
                if t != head:
                    raise AnalogTypeError(f"Analog list: {type_name(head)} vs {type_name(t)}")
            return TList(elem=head)
        
        if isinstance(expr, Extract):
            base = self.scope.lookup(expr.access.name)
            if base is TQReg:
                return TQRef
            if base is TMReg:
                return TMRef
            if isinstance(base, TList):
                return base.elem
            raise AnalogTypeError(f"Cannot index into {type_name(base)}")
        
        if isinstance(expr, (MathAdd, MathSub, MathDiv, MathPow, MathMul)):
            t1 = self.infer_expr(expr.expr1)
            t2 = self.infer_expr(expr.expr2)
            if t1 is not TScalar or t2 is not TScalar:
                raise AnalogTypeError(f"{type(expr).__name__} expects scalar, got {type_name(t1)}, {type_name(t2)}")
            return TScalar
        
        if isinstance(expr, MathFunc):
            math_funcs =  {
                "abs", "sin", "cos", "tan", "exp", "log",
                "sinh", "cosh", "tanh", "atan", "acos", "asin",
                "atanh", "asinh", "acosh", "heaviside", "conj", "real", "imag",
            }
            if expr.func in math_funcs:
                arg = expr.expr
                t = self.infer_expr(arg)
                if t is not TScalar:
                    raise AnalogTypeError(f"{expr.func} expects scalar, got {type_name(t)}")
                return TScalar
            
            if expr.func == "atan2":
                arg = expr.expr
                if len(arg) != 2:
                    raise AnalogTypeError("atan2 expects exactly 2 arguments")
                t1 = self.infer_expr(arg[0])
                t2 = self.infer_expr(arg[1])
                if t1 is not TScalar or t2 is not TScalar:
                    raise AnalogTypeError(f"{expr.func} expects scalar, got {type_name(t1)}, {type_name(t2)}")
                return TScalar
            
            raise AnalogTypeError(f"Unsupported math function: {expr.func}")
            
            
        
        if isinstance(expr, (OperatorAdd, OperatorSub, OperatorKron)):
            op1 = self.infer_expr(expr.op1)
            op2 = self.infer_expr(expr.op2)
            if op1 is not TOp or op2 is not TOp:
                raise AnalogTypeError(f"{type(expr).__name__} expects operator, got {type_name(op1)}, {type_name(op2)}")
            return TOp
        
        if isinstance(expr, OperatorMul):
            op1 = self.infer_expr(expr.op1)
            op2 = self.infer_expr(expr.op2)
            allowed = (
                (op1 is TOp and op2 is TOp) or
                (op1 is TOp and op2 is TScalar) or
                (op1 is TScalar and op2 is TOp)
            )
            if not allowed:
                raise AnalogTypeError(f"{type(expr).__name__} expects operator or scalar, got {type_name(op1)}, {type_name(op2)}")
            return TOp
        
        if isinstance(expr, (BoolAnd, BoolOr)):
            t1 = self.infer_expr(expr.expr1)
            t2 = self.infer_expr(expr.expr2)
            if t1 is not TBool or t2 is not TBool:
                raise AnalogTypeError(f"{type(expr).__name__} expects bool, got {type_name(t1)}, {type_name(t2)}")
            return TBool
        
        if isinstance(expr, (BoolEq, BoolNotEq)):
            t1 = self.infer_expr(expr.expr1)
            t2 = self.infer_expr(expr.expr2)
            if t1 not in (TBool, TScalar) or t2 not in (TBool, TScalar):
                raise AnalogTypeError(f"{type(expr).__name__} expects bool or scalar, got {type_name(t1)}, {type_name(t2)}")
            if t1 is not t2:
                raise AnalogTypeError(f"{type(expr).__name__}: got {type_name(t1)} vs {type_name(t2)}")
            return TBool
        
        
        if isinstance(expr, (BoolLessThan, BoolLessThanEq, BoolGreaterThan, BoolGreaterThanEq)):
            t1 = self.infer_expr(expr.expr1)
            t2 = self.infer_expr(expr.expr2)
            if t1 is not TScalar or t2 is not TScalar:
                raise AnalogTypeError(f"{type(expr).__name__} expects bool, got {type_name(t1)}, {type_name(t2)}")
            return TBool
        
        if isinstance(expr, BoolNot):
            t = self.infer_expr(expr.expr)
            if t is not TBool:
                raise AnalogTypeError(f"{type(expr).__name__} expects bool, got {type_name(t)}")
            return TBool
        
        
        if isinstance(expr, (Initialize, Measure)):
            t = self.infer_expr(expr.targets)
            if isinstance(t, TList):
                if t.elem not in (TQRef, TMRef):
                    raise AnalogTypeError(f"{type(expr).__name__} expects Quantum targets, got {type_name(t)}")
            elif t not in (TQReg, TMReg, TQRef, TMRef):
                raise AnalogTypeError(f"{type(expr).__name__} expects Quantum targets, got {type_name(t)}")
            return TAnalog
        
        
        if isinstance(expr, Evolve):
            tt = self.infer_expr(expr.targets)
            if isinstance(tt, TList):
                if tt.elem not in (TQRef, TMRef):
                    raise AnalogTypeError(f"{type(expr).__name__} expects Quantum targets, got {type_name(tt)}")
            elif tt not in (TQReg, TMReg, TQRef, TMRef):
                raise AnalogTypeError(f"{type(expr).__name__} expects Quantum targets, got {type_name(tt)}")
            
            td = self.infer_expr(expr.duration)
            if td is not TScalar:
                raise AnalogTypeError(f"{type(expr).__name__} expects scalar duration, got {type_name(td)}")
            
            th = self.infer_expr(expr.hamiltonian)
            if th is not TOp:
                raise AnalogTypeError(f"{type(expr).__name__} expects operator hamiltonian, got {type_name(th)}")
            
            return TAnalog
        
        

    
    def check_stmt(self, stmt):
        
        if not isinstance(stmt, STATEMENT_NODE_TYPES):
            raise AnalogTypeError(f"Unsupported statement node: {type(stmt).__name__}")
    
        if isinstance(stmt, Declaration):
            datatype = self.infer_expr(stmt.value)
            self.scope.declare(stmt.name, datatype)
            return
        
        if isinstance(stmt, IfElse):
            condition = self.infer_expr(stmt.condition)
            if condition is not TBool:
                raise AnalogTypeError("if condition must be bool")
            self.push_scope()
            for s in stmt.then_branch:
                self.check_stmt(s)
            self.pop_scope()
            self.push_scope()
            for s in stmt.else_branch:
                self.check_stmt(s)
            self.pop_scope()
            return
        
        if isinstance(stmt, While):
            condition = self.infer_expr(stmt.condition)
            if condition is not TBool:
                raise AnalogTypeError("while condition must be bool")
            self.push_scope()
            for s in stmt.body:
                self.check_stmt(s)
            self.pop_scope()
            return

        if isinstance(stmt, Break):
            return
        
        if isinstance(stmt, Continue):
            return
        
        self.infer_expr(stmt)
    

def type_check_analog(circuit: AnalogCircuit) -> None:
    checker = AnalogTypeChecker()
    checker.analyze_dataflow(circuit)
    return

