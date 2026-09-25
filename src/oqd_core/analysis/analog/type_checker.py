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

########################################################################################


from __future__ import annotations

from functools import reduce
from typing import TypeVar

from oqd_compiler_infrastructure import (
    CFG,
    CFGBlock,
    ForwardDataflowAnalysis,
    LatticeTop,
    maplattice,
)

from oqd_core.analysis.analog.types import (
    ANALOG_SUPPORTED_FUNC_SIGNATURES,
    AnalogTypeError,
    AnalogTypeLattice,
    TAnalog,
    TBool,
    TComplex,
    TFloat,
    TInt,
    TList,
    TOp,
    TypeEnv,
    get_type_name,
)
from oqd_core.interface.analog import (
    Access,
    AnalogList,
    Break,
    BuiltinCall,
    Complex,
    Constant,
    Continue,
    Declaration,
    Evolve,
    Extract,
    Initialize,
    Measure,
    ModeRegister,
    Neg,
    Not,
    Pos,
    QuantumRegister,
    RuntimeVar,
)
from oqd_core.interface.analog.expr import BinaryOp, Operator

########################################################################################


class AnalogTypeChecker(ForwardDataflowAnalysis[int, CFGBlock, TypeEnv]):
    """Forward dataflow type checker over the Control Flow Graph."""

    lattice = maplattice(AnalogTypeLattice)()

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
                    raise AnalogTypeError(
                        f"Got signature {self._print_function_signature(args)} inconsistent with {signature}"
                    )

            return True, variable_type_mapping
        except Exception:
            return False, _

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
        supported_signatures = ANALOG_SUPPORTED_FUNC_SIGNATURES[func]

        if any(
            [arg_type is self.lattice.element_lattice.bottom() for arg_type in args]
        ):
            raise AnalogTypeError(
                f"Got signature {self._print_function_signature(signature)} containing TLatticeBottom for {func}, "
                "these arguments type is inconsistent"
            )

        for sig in supported_signatures:
            _match, return_type = self._match_single_function_signature(sig, *args)

            if _match:
                return return_type

        raise AnalogTypeError(
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

            case BuiltinCall():
                name = expr.func
                args = expr.args

            case Evolve():
                name = expr.__class__.__name__
                args = [expr.hamiltonian, expr.duration, expr.targets]

            case Initialize() | Measure():
                name = expr.__class__.__name__
                args = [expr.targets]

            case QuantumRegister() | ModeRegister():
                name = expr.__class__.__name__
                args = [expr.size]
            case Extract():
                name = expr.__class__.__name__
                args = [expr.access, expr.index]
            case _:
                raise AnalogTypeError(f"Unable to infer type information from {expr}")

        return self._match_function_signature(
            name, *[self._infer_type(a, env=env) for a in args]
        )

    def _infer_binop_signature(self, expr, *, env: TypeEnv):
        return reduce(
            lambda x, y: self._match_function_signature(expr.__class__.__name__, x, y),
            [self._infer_type(a, env=env) for a in expr.exprs],
        )

    def _infer_type(self, expr, *, env: TypeEnv):
        match expr:
            case Access():
                return TAnalog if env is LatticeTop else env[expr.name]
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
            case AnalogList() if len(expr.values) == 0:
                return TList[TAnalog]
            case AnalogList():
                elem_types = [self._infer_type(e, env=env) for e in expr.values]

                combined_elem_type = reduce(
                    self.lattice.element_lattice.join, elem_types
                )

                if self.lattice.element_lattice.leq(TAnalog, combined_elem_type):
                    raise AnalogTypeError(
                        f"List elements must all be compatible but got [{', '.join([get_type_name(e) for e in elem_types])}]"
                    )

                return TList[combined_elem_type]
            case Operator():
                return TOp
            case BinaryOp():
                return self._infer_binop_signature(expr, env=env)
            case _:
                return self._infer_function_signature(expr, env=env)

    def merge(self, states):
        return self.lattice.merge_meet(states)

    def transfer(self, graph: CFG, node_id: int, state_in: TypeEnv) -> TypeEnv:
        block = graph[node_id]

        state_out = {} if state_in == self.lattice.top() else state_in.copy()

        for stmt in block.stmts:
            if block.edge_labels:
                cond_type = self._infer_type(stmt, env=state_out)
                if not self.lattice.element_lattice.equal(cond_type, TBool):
                    raise AnalogTypeError(
                        f"branch condition must be TBool got ({get_type_name(cond_type)})"
                    )
                continue

            if isinstance(stmt, (Break, Continue)):
                continue

            if isinstance(stmt, Declaration):
                state_out[stmt.name] = self._infer_type(stmt.value, env=state_out)

                continue

            self._infer_type(stmt, env=state_out)

        return state_out
