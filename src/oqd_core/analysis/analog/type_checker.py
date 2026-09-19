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
    TQReg,
    TQRegElem,
    TypeEnv,
    get_type_name,
)
from oqd_core.interface.analog import (
    Access,
    AnalogList,
    Annihilation,
    Bool,
    BoolEq,
    BoolGreaterThan,
    BoolGreaterThanEq,
    BoolLessThan,
    BoolLessThanEq,
    BoolNot,
    BoolNotEq,
    Break,
    Continue,
    Creation,
    Declaration,
    Evolve,
    Extract,
    Identity,
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
    PauliI,
    PauliX,
    PauliY,
    PauliZ,
    QuantumRegister,
)

########################################################################################


class AnalogTypeChecker(ForwardDataflowAnalysis[int, CFGBlock, TypeEnv]):
    """Forward dataflow type checker over the Control Flow Graph."""

    lattice = maplattice(AnalogTypeLattice)()

    def __init__(self, runtime_var_types={}, **kwargs):
        super().__init__(**kwargs)
        self.runtime_var_types = runtime_var_types

    def _match_single_function_signature(self, signature, func, *args, env: TypeEnv):
        sig_args_types, sig_return_type = signature

        if len(args) != len(sig_args_types):
            return False, None

        if all(
            [
                self.lattice.element_lattice.leq(sig_arg_type, arg_type)
                for arg_type, sig_arg_type in zip(args, sig_args_types)
            ]
        ):
            return True, sig_return_type

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

    def _match_function_signature(self, func, *args, env: TypeEnv):
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
            _match, return_type = self._match_single_function_signature(
                sig, func, *args, env=env
            )

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
            case (
                MathAdd()
                | MathSub()
                | MathMul()
                | MathDiv()
                | MathPow()
                | BoolEq()
                | BoolNotEq()
                | BoolGreaterThan()
                | BoolGreaterThanEq()
                | BoolLessThan()
                | BoolLessThanEq()
            ):
                name = expr.__class__.__name__
                args = [expr.expr1, expr.expr2]

            case BoolNot():
                name = expr.__class__.__name__
                args = [expr.expr]

            case MathFunc():
                name = expr.func
                args = expr.expr if isinstance(expr.expr, list) else [expr.expr]

            case OperatorAdd() | OperatorKron() | OperatorMul():
                name = expr.__class__.__name__
                args = [expr.op1, expr.op2]

            case Evolve():
                name = expr.__class__.__name__
                args = [expr.hamiltonian, expr.duration, expr.targets]

            case Initialize() | Measure():
                name = expr.__class__.__name__
                args = [expr.targets]

            case _:
                raise AnalogTypeError(f"Unable to infer type information from {expr}")

        return self._match_function_signature(
            name, *[self._infer_type(a, env=env) for a in args], env=env
        )

    def _infer_type(self, expr, *, env: TypeEnv):
        match expr:
            case Access():
                return TAnalog if env is LatticeTop else env[expr.name]
            case MathVar():
                return getattr(self.runtime_var_types, expr.name, TFloat)
            case MathImag():
                return TComplex
            case MathNum():
                return TInt if isinstance(expr.value, int) else TFloat
            case Bool():
                return TBool
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
            case QuantumRegister() | ModeRegister():
                return TQReg
            case Extract() if self.lattice.element_lattice.equal(
                env[expr.access.name], TQReg
            ):
                return TQRegElem
            case Extract() if env[expr.access.name].__origin__ == TList:
                return env[expr.access.name].__args__[0]
            case (
                PauliI()
                | PauliX()
                | PauliY()
                | PauliZ()
                | Annihilation()
                | Creation()
                | Identity()
            ):
                return TOp
            case _:
                return self._infer_function_signature(expr, env=env)

    def merge(self, states):
        return self.merge_intersection(states)

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
