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

import ast_comments as ast
from oqd_compiler_infrastructure import CFG, CFGBlock, ConversionRule, RewriteRule

from oqd_core.interface.analog import (
    Access,
    Add,
    AnalogCircuit,
    AnalogList,
    And,
    Annihilation,
    Break,
    BuiltinCall,
    Complex,
    Constant,
    Continue,
    Creation,
    Declaration,
    Div,
    Eq,
    Evolve,
    Extract,
    Geq,
    Gt,
    Identity,
    IfElse,
    Initialize,
    Kron,
    Leq,
    Lt,
    Measure,
    ModeRegister,
    Mul,
    Neg,
    Neq,
    Not,
    Or,
    PauliI,
    PauliX,
    PauliY,
    PauliZ,
    Pos,
    Pow,
    QuantumRegister,
    RuntimeVar,
    Sub,
    While,
    Xor,
)

########################################################################################

__all__ = ["PyASTtoAnalog", "AnalogCFGBuilder", "AnalogCFGtoAST"]

########################################################################################


class AnalogCFGBuilder(RewriteRule):
    def new_node(self, preds, stmt, tags=None):
        node = CFGBlock(
            register_id=self.index,
            stmts=[stmt] if stmt else [],
            preds=preds,
            tags=tags if tags else {},
        )
        self.blocks[node.register_id] = node
        self.index += 1

        explicit_labels = self.edge_labels or {}
        self.edge_labels = None

        for pred in node.preds:
            label = explicit_labels.get(pred)
            if label is None:
                label = self.fallthrough_labels.pop(pred, None)

            self.blocks[pred].add_succ(node.register_id, label=label)

        return node.register_id

    def walk_stmt(self, stmt, preds, edge_labels=None):
        old = self.preds
        old_labels = self.edge_labels
        self.preds = preds
        self.edge_labels = edge_labels
        result = self(stmt)
        self.preds = old
        self.edge_labels = old_labels
        return result

    def walk_block(self, statements, preds, entry_label=None):
        edge_labels = {}

        if entry_label:
            edge_labels = {preds[0]: entry_label}

        for stmt in statements:
            preds = self.walk_stmt(stmt, preds, edge_labels=edge_labels)
            edge_labels = None
        return preds

    def map_AnalogCircuit(self, model: AnalogCircuit) -> CFG:
        self.index = 0
        self.blocks = {}
        self.loop_stack = []
        self.preds = []
        self.edge_labels = None
        self.fallthrough_labels = {}
        node = self.new_node([], [])
        node = self.walk_block(model.statements, [node])
        node = self.new_node(node, [])
        return CFG(blocks=self.blocks)

    def map_IfElse(self, model: IfElse):
        node = self.new_node(
            self.preds,
            model.condition,
            tags={"__scf__": model.__class__.__qualname__},
        )

        then_branch = (
            self.walk_block(model.then_branch, [node], entry_label="true")
            if model.then_branch
            else [node]
        )
        else_branch = (
            self.walk_block(model.else_branch, [node], entry_label="false")
            if model.else_branch
            else [node]
        )

        fallthrough_labels = []
        if model.then_branch == []:
            fallthrough_labels.append("true")
        if model.else_branch == []:
            fallthrough_labels.append("false")

        self.fallthrough_labels[node] = fallthrough_labels

        return then_branch + else_branch

    def map_While(self, model: While):
        node = self.new_node(
            self.preds,
            model.condition,
            tags={"__scf__": model.__class__.__qualname__},
        )

        if model.body:
            self.loop_stack.append(node)
            body = self.walk_block(model.body, [node], entry_label="true")
            self.loop_stack.pop()

            self.blocks[node].add_preds(body)
            for loop_back in body:
                label = self.fallthrough_labels.pop(loop_back, None)
                self.blocks[loop_back].add_succ(node, label=label)
        else:
            self.blocks[node].add_succ(node, label="true")

        self.fallthrough_labels[node] = "false"

        return list(self.blocks[node].exit_nodes) + [node]

    def map_Break(self, model: Break):
        if not self.loop_stack:
            raise TypeError("break statement used outside loop")
        break_node = self.new_node(self.preds, model)
        self.blocks[self.loop_stack[-1]].exit_nodes.add(break_node)
        return []

    def map_Continue(self, model: Continue):
        if not self.loop_stack:
            raise TypeError("continue statement used outside loop")
        continue_node = self.new_node(self.preds, model)
        self.blocks[self.loop_stack[-1]].add_pred(continue_node)
        self.blocks[continue_node].add_succ(self.loop_stack[-1])
        return []

    def generic_map(self, model):
        return [self.new_node(self.preds, model)]


########################################################################################


class AnalogCFGtoAST(RewriteRule):
    def __init__(self, pdom_result):
        super().__init__()
        self._pdom_result = pdom_result

    @property
    def pdom(self):
        return self._pdom_result.out_states

    @property
    def dataflow_analysis(self):
        return self._pdom_result.dataflow_analysis

    @property
    def lattice(self):
        return self._pdom_result.dataflow_analysis.lattice

    def _consume_ifelse(self, current_block, blocks):
        ifelse_until = reduce(
            self.lattice.meet,
            [self.pdom[succ] for succ in current_block.succs],
        )

        then_block, then_succ = self._consume(
            blocks,
            start=current_block.edge_labels["true"],
            until=ifelse_until,
        )

        else_block, else_succ = self._consume(
            blocks,
            start=current_block.edge_labels["false"],
            until=ifelse_until,
        )

        return IfElse(
            condition=current_block.stmts[0],
            then_branch=then_block,
            else_branch=else_block,
        ), else_succ

    def _get_while_dead_blocks(self, blocks, end):
        return [
            b
            for b in blocks.keys()
            if (len(blocks[b].preds) == 0 and end in self.pdom[b])
        ]

    def _consume_while(self, current_block, blocks):
        while_until = reduce(
            self.lattice.meet,
            [self.pdom[succ] for succ in current_block.succs],
        )

        loop_block, loop_succ = self._consume(
            blocks,
            start=current_block.edge_labels["true"],
            until=while_until,
        )

        while_dead_blocks = self._get_while_dead_blocks(
            blocks, current_block.edge_labels["false"]
        )

        for b in sorted(while_dead_blocks):
            loop_block.extend(
                self._consume(blocks, start=b, until={current_block.register_id})[0]
            )

        return While(
            condition=current_block.stmts[0], body=loop_block
        ), current_block.edge_labels["false"]

    def _consume(self, blocks, start=0, until=None):
        succ = start
        statements = []
        while succ in blocks.keys():
            if until and succ in until:
                break

            current_block = blocks.pop(succ)

            if len(current_block.succs) == 0:
                statements.extend(current_block.stmts)
                break

            match current_block.tags.get("__scf__", None):
                case "IfElse":
                    ifelse_statement, ifelse_succ = self._consume_ifelse(
                        current_block, blocks
                    )
                    statements.append(ifelse_statement)
                    succ = ifelse_succ

                case "While":
                    while_statement, while_succ = self._consume_while(
                        current_block, blocks
                    )
                    statements.append(while_statement)
                    succ = while_succ

                case _:
                    statements.extend(current_block.stmts)
                    succ = list(current_block.succs)[0]

        return statements, succ

    def map_CFG(self, model):
        circuit = AnalogCircuit()

        statements, _ = self._consume(model.blocks.copy())
        circuit.statements.extend(statements)

        return circuit


########################################################################################


class PyASTtoAnalog(ConversionRule):
    def generic_map(self, model, operands):
        return tuple()

    def map_Module(self, model, operands):
        return AnalogCircuit(statements=operands["body"][0])

    def _filter_block(self, statements):
        return list(filter(lambda x: x != tuple(), statements))

    def map_FunctionDef(self, model, operands):
        return self._filter_block(operands["body"])

    def map_Assign(self, model, operands):
        if len(model.targets) != 1:
            raise ValueError()
        if model.targets[0].id.startswith("__"):
            raise ValueError()
        return Declaration(name=model.targets[0].id, value=operands["value"])

    def map_Constant(self, model, operands):
        if isinstance(model.value, complex):
            return Constant(value=Complex(real=model.value.real, imag=model.value.imag))
        return Constant(value=model.value)

    def map_Return(self, model, operands):
        return operands["value"]

    def map_Call(self, model, operands):
        args = operands["args"]

        if not isinstance(model.func, ast.Name):
            raise ValueError

        name = model.func.id

        match name:
            case (
                "sin"
                | "cos"
                | "heaviside"
                | "abs"
                | "sin"
                | "cos"
                | "tan"
                | "exp"
                | "log"
                | "sinh"
                | "cosh"
                | "tanh"
                | "atan"
                | "acos"
                | "asin"
                | "atanh"
                | "asinh"
                | "acosh"
                | "heaviside"
                | "conj"
                | "real"
                | "imag"
                | "atan2"
                | "round"
                | "range"
                | "flatten"
                | "print"
                | "len"
            ):
                return BuiltinCall(func=name, args=args)

            case "evolve":
                if len(args) not in [3, 4]:
                    raise ValueError()

                if len(args) == 3:
                    return Evolve(
                        hamiltonian=args[0], duration=args[1], targets=args[2]
                    )

                return Evolve(
                    hamiltonian=args[0],
                    jumps=args[1],
                    duration=args[2],
                    targets=args[3],
                )

            case "initialize" | "measure":
                if len(args) != 1:
                    raise ValueError()
                return dict(initialize=Initialize, measure=Measure)[name](
                    targets=args[0]
                )

            case "qmode":
                if len(args) != 1:
                    raise ValueError()
                return ModeRegister(size=args[0])

            case "qreg":
                if len(args) not in [1, 2]:
                    raise ValueError()

                if len(args) == 1:
                    return QuantumRegister(size=args[0])

                return QuantumRegister(size=args[0], dim=args[1])

            case "X" | "Y" | "Z" | "I":
                if len(args) not in [0, 2, 3]:
                    raise ValueError()
                if len(args) == 0:
                    return dict(X=PauliX, Y=PauliY, Z=PauliZ, I=PauliI)[name]()
                if len(args) == 2:
                    return dict(X=PauliX, Y=PauliY, Z=PauliZ, I=PauliI)[name](
                        level1=args[0], level2=args[1]
                    )
                return dict(X=PauliX, Y=PauliY, Z=PauliZ, I=PauliI)[name](
                    level1=args[0], level2=args[1], dim=args[2]
                )
            case "A" | "C" | "J":
                return dict(A=Annihilation, C=Creation, J=Identity)[name]()

            case _:
                raise ValueError("Unsupported builtin function call")

    def map_List(self, model, operands):
        return AnalogList(values=operands["elts"])

    def map_Subscript(self, model, operands):
        return Extract(value=operands["value"], index=operands["slice"])

    def map_UnaryOp(self, model, operands):
        operand = operands["operand"]
        match model.op:
            case ast.USub():
                return Neg(expr=operand)
            case ast.UAdd():
                return Pos(expr=operand)
            case ast.Not():
                return Not(expr=operand)

    def map_BinOp(self, model, operands):
        left, right = operands["left"], operands["right"]
        match model.op:
            case ast.Add():
                return Add(exprs=[left, right])
            case ast.Sub():
                return Sub(exprs=[left, right])
            case ast.Mult():
                return Mul(exprs=[left, right])
            case ast.Div():
                return Div(exprs=[left, right])
            case ast.Pow():
                return Pow(exprs=[left, right])
            case ast.MatMult():
                return Kron(exprs=[left, right])
            case ast.BitAnd():
                return And(exprs=[left, right])
            case ast.BitOr():
                return Or(exprs=[left, right])
            case ast.BitXor():
                return Xor(exprs=[left, right])

    def map_BoolOp(self, model, operands):
        args = operands["values"]
        match model.op:
            case ast.And():
                return And(exprs=args)
            case ast.Or():
                return Or(exprs=args)
            case ast.Xor():
                return Xor(exprs=args)

    def _compare_helper(self, op, left, right):
        match op:
            case ast.Eq():
                return Eq(exprs=[left, right])
            case ast.NotEq():
                return Neq(exprs=[left, right])
            case ast.Lt():
                return Lt(exprs=[left, right])
            case ast.LtE():
                return Leq(exprs=[left, right])
            case ast.Gt():
                return Gt(exprs=[left, right])
            case ast.GtE():
                return Geq(exprs=[left, right])

    def map_Compare(self, model, operands):
        left = operands["left"]
        args = operands["comparators"]
        ops = model.ops

        current = left

        while ops:
            current = self._compare_helper(ops.pop(0), current, args.pop(0))

        return current

    def map_While(self, model, operands):
        return While(
            condition=operands["test"], body=self._filter_block(operands["body"])
        )

    def map_Name(self, model, operands):
        if isinstance(model.ctx, ast.Load):
            return (
                RuntimeVar(name="#" + model.id.removeprefix("__"))
                if model.id.startswith("__")
                else Access(name=model.id)
            )

    def map_If(self, model, operands):
        if model.orelse:
            return IfElse(
                condition=operands["test"],
                then_branch=self._filter_block(operands["body"]),
                else_branch=self._filter_block(operands["orelse"]),
            )

        return IfElse(
            condition=operands["test"], then_branch=self._filter_block(operands["body"])
        )

    def map_Expr(self, model, operands):
        return operands["value"]
