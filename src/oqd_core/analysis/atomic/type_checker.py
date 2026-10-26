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

from functools import reduce
from typing import TypeVar

from oqd_compiler_infrastructure import (
    CFG,
    CFGBlock,
    ForwardDataflowAnalysis,
    maplattice,
)

from oqd_core.analysis.atomic.types import (
    ATOMIC_SUPPORTED_FUNC_SIGNATURES,
    AtomicTypeError,
    AtomicTypeLattice,
    TBeam,
    TBool,
    TComplex,
    TFloat,
    TInt,
    TList,
    TPulse,
    TypeEnv,
    get_type_name,
)
from oqd_core.interface.atomic import (
    Access,
    AtomicList,
    Beam,
    Break,
    BuiltinCall,
    Complex,
    Constant,
    Continue,
    Declaration,
    Extract,
    IonRegister,
    Neg,
    Not,
    ParallelProtocol,
    Pos,
    Pulse,
    RuntimeVar,
    SerialProtocol,
)
from oqd_core.interface.atomic.expr import BinaryOp

########################################################################################


class AtomicTypeChecker(ForwardDataflowAnalysis[int, CFGBlock, TypeEnv]):
    """Forward dataflow type checker over the Control Flow Graph."""
    
    lattice = maplattice(AtomicTypeLattice, default_mode="top")()

    def __init__(self, runtime_var_types={}, **kwargs):
        super().__init__(**kwargs)
        self.runtime_var_types = runtime_var_types
    
    def _compare_variable_type(
        self, signature, args, subsignature, subargs, variable_type_mapping={}
    ):
        try:
            for n, _ in enumerate(subsignature):
                if getattr(subsignature[n], "__origin__", None) is TList:
                    self._compare_variable_type(
                        signature,
                        args,
                        subsignature[n].__args__,
                        subargs[n].__args__,
                        variable_type_mapping=variable_type_mapping,
                    )
                    continue

                if not isinstance(subsignature[n], TypeVar):
                    continue

                if variable_type_mapping.get(subsignature[n], None) is None:
                    variable_type_mapping[subsignature[n]] = subargs[n]
                    continue

                if variable_type_mapping[subsignature[n]] != subargs[n]:
                    raise AtomicTypeError(
                        f"Got signature {self._print_function_signature(args)} inconsistent with {signature}"
                    )

            return True, variable_type_mapping
        except Exception:
            return False, _
    
    def _check_protocol(self, protocol, env: TypeEnv) -> TypeEnv:
        if isinstance(protocol, SerialProtocol):
            curr_env = env
            for child in protocol.pulses:
                curr_env = self._check_protocol_stmt(child, curr_env)
            return curr_env

        if isinstance(protocol, ParallelProtocol):
            for child in protocol.pulses:
                self._check_protocol_stmt(child, env)
            return env

        raise AtomicTypeError(f"Unsupported protocol node: {type(protocol).__name__}")
    
    def _replace_variable_type(self, variable_type_mapping, original_type):
        if getattr(original_type, "__origin__", None) is TList:
            return TList[
                tuple(
                    [
                        self._replace_variable_type(variable_type_mapping, arg)
                        for arg in original_type.__args__
                    ]
                )
            ]

        if variable_type_mapping.get(original_type, None):
            return variable_type_mapping[original_type]

        return original_type
    
    def _match_single_function_signature(self, signature, *args):
        sig_args_types, sig_return_type = signature

        if len(args) != len(sig_args_types):
            return False, None

        success, variable_type_mapping = self._compare_variable_type(
            signature, args, sig_args_types, args, {}
        )

        if success and all(
            [
                self.lattice.element_lattice.leq(
                    self._replace_variable_type(variable_type_mapping, sig_arg_type),
                    arg_type,
                )
                for arg_type, sig_arg_type in zip(args, sig_args_types)
            ]
        ):
            return True, self._replace_variable_type(
                variable_type_mapping, sig_return_type
            )

        return False, None

    def _print_function_signature(self, signature):
        sig_args_types, sig_return_type = signature
        
        if sig_return_type:
            return (
                f"({', '.join([get_type_name(x) for x in sig_args_types])})"
                " -> "
                f"{get_type_name(sig_return_type)}"
            )

        return f"({', '.join([get_type_name(x) for x in sig_args_types])})"
    
    def _match_function_signature(self, func, *args):
        signature = (args, None)
        supported_signatures = ATOMIC_SUPPORTED_FUNC_SIGNATURES[func]

        if any(
            [arg_type is self.lattice.element_lattice.bottom() for arg_type in args]
        ):
            raise AtomicTypeError(
                f"Got signature {self._print_function_signature(signature)} containing TLatticeBottom for {func}, "
                "these arguments type is inconsistent"
            )

        for sig in supported_signatures:
            _match, return_type = self._match_single_function_signature(sig, *args)

            if _match:
                return return_type

        raise AtomicTypeError(
            f"Got signature {self._print_function_signature(signature)} for {func}, "
            "but signature must be one of:\n  "
            + "\n  ".join(
                [self._print_function_signature(sig) for sig in supported_signatures]
            )
        )
    
    def _infer_function_signature(self, expr, *, env: TypeEnv):
        match expr:
            case Not() | Neg() | Pos():
                name = expr.__class__.__name__
                args = [expr.expr]

            case BuiltinCall() if expr.func.startswith("_"):
                name = expr.func.removeprefix("_").capitalize()
                args = expr.args

            case BuiltinCall():
                name = expr.func
                args = expr.args

            case Beam():
                name = expr.__class__.__name__
                args = [expr.frequency, expr.rabi, expr.phase, expr.polarization, expr.wavevector]

            case Pulse():
                name = expr.__class__.__name__
                args = [expr.beam, expr.duration, expr.target, expr.measured]

            case IonRegister():
                name = expr.__class__.__name__
                args = [expr.size]
            
            case Extract():
                name = expr.__class__.__name__
                args = [expr.value, expr.index]
            
            case _:
                raise AtomicTypeError(f"unable to infer type information from {expr}")

        return self._match_function_signature(
            name, *[self._infer_type(a, env=env) for a in args]
        )
    
    def _infer_binop_signature(self, name, args, *, env: TypeEnv):
        return reduce(
            lambda x, y: self._match_function_signature(name, x, y),
            [self._infer_type(a, env=env) for a in args],
        )
    
    def _infer_type(self, expr, *, env: TypeEnv):
        match expr:
            case Access():
                return (
                    self.lattice.element_lattice.top()
                    if env is self.lattice.top()
                    else env.get(expr.name, self.lattice.element_lattice.top())
                )
            case RuntimeVar():
                return getattr(self.runtime_var_types, expr.name, TFloat)
            case Constant() if type(expr.value) is bool:
                return TBool
            case Constant() if type(expr.value) is int:
                return TInt
            case Constant() if isinstance(expr.value, float):
                return TFloat
            case Constant() if isinstance(expr.value, Complex):
                return TComplex
            case Constant() if isinstance(expr.value, Beam):
                return TBeam
            case Constant() if isinstance(expr.value, Pulse):
                return TPulse
            case AtomicList() if len(expr.values) == 0:
                return TList[self.lattice.element_lattice.top()]
            case AtomicList():
                elem_types = [self._infer_type(e, env=env) for e in expr.values]

                combined_elem_type = reduce(
                    self.lattice.element_lattice.meet, elem_types
                )

                if combined_elem_type is self.lattice.element_lattice.bottom():
                    raise AtomicTypeError(
                        f"list elements must all have consistent type but got [{', '.join([get_type_name(e) for e in elem_types])}]"
                    )

                return TList[combined_elem_type]
            case BinaryOp():
                return self._infer_binop_signature(
                    expr.__class__.__name__, expr.exprs, env=env
                )
            case BuiltinCall() if expr.func in [
                "_add",
                "_sub",
                "_mul",
                "_div",
                "_pow",
                "_eq",
                "_neq",
                "_lt",
                "_leq",
                "_gt",
                "_geq",
                "_and",
                "_or",
                "_xor",
            ]:
                return self._infer_binop_signature(
                    expr.func.removeprefix("_").capitalize(), expr.args, env=env
                )
            case _:
                return self._infer_function_signature(expr, env=env)

    def merge(self, states):
        out = self.lattice.merge_meet(states)

        invalids = (
            tuple(
                k for k, v in out.items() if v is self.lattice.element_lattice.bottom()
            )
            if isinstance(out, dict)
            else tuple()
        )
        if invalids:
            raise AtomicTypeError(
                f"declaration of {invalids} inconsistent types across different branches"
            )

        return out

    def _check_protocol_stmt(self, stmt, env: TypeEnv) -> TypeEnv:
        # TODO: Verify that every stmt inside Parallel/Serial block is:
        # {Parallel, Serial, Declaration, Pulse}. If stmt is Declaration, stmt.value must be Pulse.

        if isinstance(stmt, (ParallelProtocol, SerialProtocol)):
            return self._check_protocol(stmt, env)

        if isinstance(stmt, Declaration):
            declare_type = self._infer_type(
                stmt.value, env=env
            )
            env[stmt.name] = self.lattice.element_lattice.meet(
                declare_type,
                env.get(
                    stmt.name,
                    self.lattice.element_lattice.top(),
                ),
            )
            # t = self._infer_type(stmt.value, env=env)
            if not self.lattice.element_lattice.leq(declare_type, TPulse):
                raise AtomicTypeError(
                    f"Parallel/Serial blocks expect only Pulse statements, got {get_type_name(declare_type)}"
                )
            return env

        t = self._infer_type(stmt, env=env)
        if not self.lattice.element_lattice.leq(t, TPulse):
            raise AtomicTypeError(
                f"Parallel/Serial blocks expect only Pulse statements, got {get_type_name(t)}"
            )
        return env

    def transfer(self, graph: CFG, node_id: int, state_in: TypeEnv) -> TypeEnv:
        block = graph[node_id]

        block_types = {}
        for stmt in block.stmts:
            if block.edge_labels:
                cond_type = self._infer_type(
                    stmt, env=self.lattice.meet(state_in, block_types)
                )
                if not self.lattice.element_lattice.equal(cond_type, TBool):
                    raise AtomicTypeError(
                        f"branch condition must be TBool got ({get_type_name(cond_type)})"
                    )
                continue

            if isinstance(stmt, (Break, Continue)):
                continue

            if isinstance(stmt, (ParallelProtocol, SerialProtocol)):
                self._check_protocol(stmt, self.lattice.meet(state_in, block_types))
                continue

            if isinstance(stmt, Declaration):
                declare_type = self._infer_type(
                    stmt.value, env=self.lattice.meet(state_in, block_types)
                )

                block_types[stmt.name] = self.lattice.element_lattice.meet(
                    declare_type,
                    block_types.get(
                        stmt.name,
                        self.lattice.element_lattice.top(),
                    ),
                )

                if (
                    self.lattice.meet(state_in, block_types)[stmt.name]
                    is self.lattice.element_lattice.bottom()
                ):
                    raise AtomicTypeError(
                        f"attempted to redeclare '{stmt.name}' to a different type {get_type_name(declare_type)}"
                    )

                continue

            self._infer_type(stmt, env=self.lattice.meet(state_in, block_types))

        state_out = self.lattice.meet(state_in, block_types)
        return state_out

