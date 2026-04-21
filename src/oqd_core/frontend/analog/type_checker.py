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

@dataclass
class TList(TAnalog):
    elem: Union[TList, TAnalog]


########################################################################################

@dataclass
class Scope:
    parent: Union[Scope, None] = None
    symbols: Dict[str, Union[TAnalog, TList]] = field(default_factory=dict)
    children: list[Scope] = field(default_factory=list)
    
    def lookup(self, name):
        scope = self
        while scope is not None:
            if name in scope.symbols:
                return scope.symbols[name]
            scope = scope.parent
        raise AnalogTypeError(f"Undefined variable: {name}")
            
    def declare(self, name: str, datatype: Union[TAnalog, TList]):
        self.symbols[name] = datatype
        
    def to_dict(self):
        if not self.symbols and not self.children:
            return None
        return {
            "symbols": {k: type(v).__name__ for k, v in self.symbols.items()},
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
    
    
    def infer_expr(self, expr):
        if not isinstance(expr, EXPR_NODE_TYPES):
            raise AnalogTypeError(f"Unsupported expression node: {type(expr).__name__}")

        if isinstance(expr, TERMINAL_NODE_TYPES):
            if isinstance(expr, (MathNum, MathVar, MathImag)):
                return TScalar()
            if isinstance(expr, Bool):
                return TBool()
            if isinstance(expr, (PauliI, PauliX, PauliY, PauliZ, Creation, Annihilation, Identity)):
                return TOp()
            if isinstance(expr, QuantumRegister):
                return TQReg()
            if isinstance(expr, ModeRegister):
                return TMReg()
            if isinstance(expr, Access):
                return self.scope.lookup(expr.name)
    
        if isinstance(expr, AnalogList):
            if not expr.values:
                return TList(elem=None)
            
            head = self.infer_expr(expr.values[0])
            for v in expr.values[1:]:
                t = self.infer_expr(v)
                if type(t) is not type(head):
                    raise AnalogTypeError(f"Analog list: {type(head).__name__} vs {type(t).__name__}")
            return TList(elem=head)
        
        if isinstance(expr, Extract):
            base = self.scope.lookup(expr.access.name)
            if isinstance(base, TQReg):
                return TQRef()
            if isinstance(base, TMReg):
                return TMRef()
            if isinstance(base, TList):
                return base.elem
            raise AnalogTypeError(f"Cannot index into {type(base).__name__}")
        
        if isinstance(expr, (MathAdd, MathSub, MathDiv, MathPow, MathMul)):
            t1 = self.infer_expr(expr.expr1)
            t2 = self.infer_expr(expr.expr2)
            if not isinstance(t1, TScalar) or not isinstance(t2, TScalar):
                raise AnalogTypeError(f"{type(expr).__name__} expects scalar, got {type(t1).__name__}, {type(t2).__name__}")
            return TScalar()
        
        if isinstance(expr, MathFunc):
            math_funcs =  {
                "abs", "sin", "cos", "tan", "exp", "log",
                "sinh", "cosh", "tanh", "atan", "acos", "asin",
                "atanh", "asinh", "acosh", "heaviside", "conj", "real", "imag",
            }
            if expr.func in math_funcs:
                arg = expr.expr
                t = self.infer_expr(arg)
                if not isinstance(t, TScalar):
                    raise AnalogTypeError(f"{expr.func} expects scalar, got {type(t).__name__}")
                return TScalar()
            
            if expr.func == "atan2":
                arg = expr.expr
                if len(arg) != 2:
                    raise AnalogTypeError("atan2 expects exactly 2 arguments")
                t1 = self.infer_expr(arg[0])
                t2 = self.infer_expr(arg[1])
                if not isinstance(t1, TScalar) or not isinstance(t2, TScalar):
                    raise AnalogTypeError(f"{expr.func} expects scalar, got {type(t1).__name__}, {type(t2).__name__}")
                return TScalar()
            
            raise AnalogTypeError(f"Unsupported math function: {expr.func}")
            
            
        
        if isinstance(expr, (OperatorAdd, OperatorSub, OperatorKron)):
            op1 = self.infer_expr(expr.op1)
            op2 = self.infer_expr(expr.op2)
            if not isinstance(op1, TOp) or not isinstance(op2, TOp):
                raise AnalogTypeError(f"{type(expr).__name__} expects operator, got {type(op1).__name__}, {type(op2).__name__}")
            return TOp()
        
        if isinstance(expr, OperatorMul):
            op1 = self.infer_expr(expr.op1)
            op2 = self.infer_expr(expr.op2)
            allowed = (
                (isinstance(op1, TOp) and isinstance(op2, TOp)) or
                (isinstance(op1, TOp) and isinstance(op2, TScalar)) or
                (isinstance(op1, TScalar) and isinstance(op2, TOp))
            )
            if not allowed:
                raise AnalogTypeError(f"{type(expr).__name__} expects operator or scalar, got {type(op1).__name__}, {type(op2).__name__}")
            return TOp()
        
        if isinstance(expr, (BoolAnd, BoolOr)):
            t1 = self.infer_expr(expr.expr1)
            t2 = self.infer_expr(expr.expr2)
            if not isinstance(t1, TBool) or not isinstance(t2, TBool):
                raise AnalogTypeError(f"{type(expr).__name__} expects bool, got {type(t1).__name__}, {type(t2).__name__}")
            return TBool()
        
        if isinstance(expr, (BoolEq, BoolNotEq)):
            t1 = self.infer_expr(expr.expr1)
            t2 = self.infer_expr(expr.expr2)
            if not isinstance(t1, (TBool, TScalar)) or not isinstance(t2, (TBool, TScalar)):
                raise AnalogTypeError(f"{type(expr).__name__} expects bool or scalar, got {type(t1).__name__}, {type(t2).__name__}")
            if type(t1) is not type(t2):
                raise AnalogTypeError(f"{type(expr).__name__}: got {type(t1).__name__} vs {type(t2).__name__}")
            return TBool()
        
        
        if isinstance(expr, (BoolLessThan, BoolLessThanEq, BoolGreaterThan, BoolGreaterThanEq)):
            t1 = self.infer_expr(expr.expr1)
            t2 = self.infer_expr(expr.expr2)
            if not isinstance(t1, TScalar) or not isinstance(t2, TScalar):
                raise AnalogTypeError(f"{type(expr).__name__} expects bool, got {type(t1).__name__}, {type(t2).__name__}")
            return TBool()
        
        if isinstance(expr, BoolNot):
            t = self.infer_expr(expr.expr)
            if not isinstance(t, TBool):
                raise AnalogTypeError(f"{type(expr).__name__} expects bool, got {type(t).__name__}")
            return TBool()
        
        
        if isinstance(expr, (Initialize, Measure)):
            t = self.infer_expr(expr.targets)
            if isinstance(t, TList):
                if not isinstance(t.elem, (TQRef, TMRef)):
                    raise AnalogTypeError(f"{type(expr).__name__} expects Quantum targets, got {type(t).__name__}")
            elif not isinstance(t, (TQReg, TMReg, TQRef, TMRef)):
                raise AnalogTypeError(f"{type(expr).__name__} expects Quantum targets, got {type(t).__name__}")
            return TAnalog()
        
        
        if isinstance(expr, Evolve):
            tt = self.infer_expr(expr.targets)
            if isinstance(tt, TList):
                if not isinstance(tt.elem, (TQRef, TMRef)):
                    raise AnalogTypeError(f"{type(expr).__name__} expects Quantum targets, got {type(tt).__name__}")
            elif not isinstance(tt, (TQReg, TMReg, TQRef, TMRef)):
                raise AnalogTypeError(f"{type(expr).__name__} expects Quantum targets, got {type(tt).__name__}")
            
            td = self.infer_expr(expr.duration)
            if not isinstance(td, TScalar):
                raise AnalogTypeError(f"{type(expr).__name__} expects scalar duration, got {type(td).__name__}")
            
            th = self.infer_expr(expr.hamiltonian)
            if not isinstance(th, TOp):
                raise AnalogTypeError(f"{type(expr).__name__} expects operator hamiltonian, got {type(th).__name__}")
            
            return TAnalog()
        
        

    
    def check_stmt(self, stmt):
        
        if not isinstance(stmt, STATEMENT_NODE_TYPES):
            raise AnalogTypeError(f"Unsupported statement node: {type(stmt).__name__}")
    
        if isinstance(stmt, Declaration):
            datatype = self.infer_expr(stmt.value)
            self.scope.declare(stmt.name, datatype)
            return
        
        if isinstance(stmt, IfElse):
            condition = self.infer_expr(stmt.condition)
            if not isinstance(condition, TBool):
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
            if not isinstance(condition, TBool):
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
    for stmt in circuit.statements:
        checker.check_stmt(stmt)
    
    return
    return checker.root.to_dict()

