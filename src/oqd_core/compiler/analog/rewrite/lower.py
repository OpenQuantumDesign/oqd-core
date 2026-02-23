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

from typing import Dict, List, Optional, Tuple
from oqd_compiler_infrastructure import Post, ConversionRule

########################################################################################
from oqd_core.interface.analog import (
    AnalogCircuit,
    AnalogCircuitSSA,
    Block,
    Branch,
    CondBranch,
    Evolve,
    Exit,
    IfElse,
    Initialize,
    Measure,
    SSADefBool,
    SSADefMath,
    SSAValBool,
    While,
)
from oqd_core.interface.analog.bool import BoolRef
from oqd_core.interface.analog.ssa import BlockBodyItem
from oqd_core.interface.analog.declaration import BoolDeclaration, MathExprDeclaration
from oqd_core.interface.math import MathRef, SSAValMath

########################################################################################
__all__ = ["lower_analog_circuit"]

########################################################################################

def lower_analog_circuit(circuit: AnalogCircuit) -> AnalogCircuitSSA:
    return Post(AnalogCircuitToSSARule())(circuit)


class AnalogCircuitToSSARule(ConversionRule):
    def __init__(self):
        super().__init__()
        self._label_id = 0
    
    def _next_label(self, prefix = "b"):
        self._label_id += 1
        return f"{prefix}{self._label_id}"

    def map_AnalogCircuit(self, model: AnalogCircuit, operands):
        declarations = operands.get("declarations", model.declarations)
        sequence = operands.get("sequence", model.sequence)

        # Non-SSA declarations
        non_ssa_decls = [
            d for d in declarations
            if not isinstance(d, (SSADefBool, SSADefMath))
        ]

        # SSA defs from transformed Bool/Math declarations go into first block
        ssa_defs = [
            d for d in declarations
            if isinstance(d, (SSADefBool, SSADefMath))
        ]

        blocks = self._to_blocks(sequence, "entry", entry_defs=ssa_defs)

        return AnalogCircuitSSA(
            qreg=operands.get("qreg", model.qreg),
            creg=operands.get("creg", model.creg),
            declarations=non_ssa_decls,
            blocks=blocks,
        )
    
    def map_Evolve(self, model: Evolve, operands):
        gate = operands.get("gate", model.gate)
        return Evolve(key="evolve", duration=model.duration, gate=gate)

    def map_Initialize(self, model, operands):
        return Initialize(key="initialize")

    def map_Measure(self, model, operands):
        return Measure(key="measure")
    
    def map_IfElse(self, model: IfElse, operands):
        cond = operands.get("condition", model.condition)
        then_frags = operands.get("then_branch", [])
        else_frags = operands.get("else_branch", [])

        then_label = self._next_label("then")
        else_label = self._next_label("else")
        merge_label = self._next_label("merge")

        then_blocks = self._to_blocks(then_frags, then_label, branch_target=merge_label)
        else_blocks = self._to_blocks(else_frags, else_label, branch_target=merge_label)

        terminator = CondBranch(
            condition=cond,
            true_target=then_label,
            true_args=[],
            false_target=else_label,
            false_args=[],
        )
        return ("ctrl", terminator, then_blocks + else_blocks, merge_label)
    
    def map_While(self, model: While, operands):
        cond = operands.get("condition", model.condition)
        body_frags = operands.get("body", [])
        
        cond_label = self._next_label("cond")
        body_label = self._next_label("body")
        exit_label = self._next_label("exit")
        
        body_blocks = self._to_blocks(body_frags, body_label, branch_target=cond_label)
        
        cond_block = Block(
            label=cond_label,
            args=[],
            body=[],
            terminator=CondBranch(
                condition=cond,
                true_target=body_label,
                true_args=[],
                false_target=exit_label,
                false_args=[],
            ),
        )
        
        if not body_blocks:
            body_blocks = [
                Block(
                    label=body_label,
                    args=[],
                    body=[],
                    terminator=Branch(target=cond_label, args=[])
                )
            ]
            
        terminator=Branch(target=cond_label, args=[])
        return ("ctrl", terminator, [cond_block] + body_blocks, exit_label)
    
    def _to_blocks(self, seq: List, first_label: str, *, entry_defs = None, branch_target = None) -> List[Block]:
        
        body = list(entry_defs) if entry_defs else []
        blocks = []
        
        for i, item in enumerate(seq):
            if isinstance(item, tuple) and item[0] == "ctrl" and len(item) == 4:
                _, terminator, sub_blocks, merge_label = item
                if body:
                    blocks.append(
                        Block(
                            label=first_label if not blocks else self._next_label(),
                            args=[],
                            body=body,
                            terminator=terminator
                        )
                    )
                    body = []
                else:
                    blocks.append(
                        Block(
                            label=first_label if not blocks else self._next_label(),
                            args=[],
                            body=[],
                            terminator=terminator,
                        )
                    )
                blocks.extend(sub_blocks)
                merge_blocks = self._to_blocks(seq[i+1:], merge_label)
                blocks.extend(merge_blocks)
                return blocks
            
            body.append(item)
            
        if body:
            terminator = Branch(target=branch_target, args = []) if branch_target else Exit()
            blocks.append(
                Block(
                    label=first_label if not blocks else self._next_label(),
                    args=[],
                    body=body,
                    terminator=terminator,
                )
            )
        
        if not body and branch_target:
            blocks.append(
                Block(
                    label=first_label,
                    args=[],
                    body=[],
                    terminator=Branch(target=branch_target, args=[]),
                )
            )
                
        return blocks
    
    def map_BoolRef(self, model: BoolRef, operands):
        return SSAValBool(name=model.name)

    def map_MathRef(self, model: MathRef, operands):
        return SSAValMath(name=model.name)

    def map_BoolDeclaration(self, model: BoolDeclaration, operands):
        return SSADefBool(name=model.name, expr=operands["expr"])

    def map_MathExprDeclaration(self, model: MathExprDeclaration, operands):
        return SSADefMath(name=model.name, expr=operands["expr"])

    def generic_map(self, model, operands):
        return model
    
    

