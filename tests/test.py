import os

from rich import print as pprint
from rich.console import Console

import types

import networkx as nx

from matplotlib import pyplot as plt

import numpy as np

from functools import reduce
from quantumion.compiler.flow import FlowOut, Traversal

########################################################################################


from quantumion.interface.math import *
from quantumion.interface.analog.operator import *

from quantumion.compiler.visitor import Visitor, Transformer
from quantumion.compiler.math import *
from quantumion.compiler.analog.base import *
from quantumion.compiler.analog.canonicalize import *
from quantumion.compiler.analog.verify import *

from quantumion.compiler.flow import *

########################################################################################

I, X, Y, Z, P, M = PauliI(), PauliX(), PauliY(), PauliZ(), PauliPlus(), PauliMinus()
A, C, J = Annihilation(), Creation(), Identity()

########################################################################################


def random_hexcolor():
    return "#{:02X}{:02X}{:02X}".format(*np.random.randint(0, 255, 3))


class MermaidMathExpr(Transformer):

    def emit(self, model):
        self.element = 0

        self.mermaid_string = "```mermaid\ngraph TD\n"
        model.accept(self)
        self.mermaid_string += "".join(
            [
                f"classDef {model} stroke:{random_hexcolor()},stroke-width:3px\n"
                for model in [
                    "MathAdd",
                    "MathSub",
                    "MathMul",
                    "MathDiv",
                    "MathFunc",
                    "MathNum",
                    "MathVar",
                    "MathImag",
                ]
            ]
        )
        self.mermaid_string += "```\n"

        return self.mermaid_string

    def visit_MathImag(self, model):
        element = self.element
        self.mermaid_string += 'element{}("{}"):::{}\n'.format(
            self.element,
            model.__class__.__name__,
            model.__class__.__name__,
        )
        self.element += 1

        return f"element{element}"

    def visit_MathVar(self, model):
        element = self.element
        self.mermaid_string += 'element{}("{}<br/>{}<br/>{}"):::{}\n'.format(
            self.element,
            model.__class__.__name__,
            "-" * len(model.__class__.__name__),
            "name = #quot;{}#quot;".format(model.name),
            model.__class__.__name__,
        )
        self.element += 1

        return f"element{element}"

    def visit_MathNum(self, model):
        element = self.element
        self.mermaid_string += 'element{}("{}<br/>{}<br/>{}"):::{}\n'.format(
            self.element,
            model.__class__.__name__,
            "-" * len(model.__class__.__name__),
            "value = {}".format(model.value),
            model.__class__.__name__,
        )
        self.element += 1

        return f"element{element}"

    def visit_MathBinaryOp(self, model):
        left = self.visit(model.expr1)
        right = self.visit(model.expr2)

        element = self.element
        self.mermaid_string += 'element{}("{}"):::{}\n'.format(
            self.element,
            model.__class__.__name__,
            model.__class__.__name__,
        )

        self.mermaid_string += f"element{element} --> {left} & {right}\n"

        self.element += 1

        return f"element{element}"

    def visit_MathFunc(self, model):
        expr = self.visit(model.expr)

        element = self.element
        self.mermaid_string += 'element{}("{}<br/>{}<br/>{}"):::{}\n'.format(
            self.element,
            model.__class__.__name__,
            "-" * len(model.__class__.__name__),
            "func = {}".format(model.func),
            model.__class__.__name__,
        )

        self.mermaid_string += f"element{element} --> {expr}\n"

        self.element += 1

        return f"element{element}"


class MermaidOperator(Transformer):

    def emit(self, model):
        self.element = 0

        self.mermaid_string = "```mermaid\ngraph TD\n"
        model.accept(self)
        self.mermaid_string += "".join(
            [
                f"classDef {model} stroke:{random_hexcolor()},stroke-width:3px\n"
                for model in [
                    "Pauli",
                    "Ladder",
                    "OperatorAdd",
                    "OperatorScalarMul",
                    "OperatorKron",
                    "OperatorMul",
                ]
            ]
        )
        self.mermaid_string += "```\n"

        return self.mermaid_string

    def visit_MathExpr(self, model):
        element = self.element
        self.mermaid_string += 'element{}("{}<br/>{}<br/>{}"):::{}\n'.format(
            self.element,
            "MathExpr",
            "-" * len("MathExpr"),
            "expr = #quot;{}#quot;".format(model.accept(PrintMathExpr())),
            "MathExpr",
        )
        self.element += 1

        return f"element{element}"

    def visit_Pauli(self, model):
        element = self.element
        self.mermaid_string += 'element{}("{}"):::{}\n'.format(
            self.element, model.__class__.__name__, "Pauli"
        )
        self.element += 1

        return f"element{element}"

    def visit_Ladder(self, model):
        element = self.element
        self.mermaid_string += 'element{}("{}"):::{}\n'.format(
            self.element, model.__class__.__name__, "Ladder"
        )
        self.element += 1

        return f"element{element}"

    def visit_OperatorBinaryOp(self, model):
        left = self.visit(model.op1)
        right = self.visit(model.op2)

        element = self.element
        self.mermaid_string += 'element{}("{}"):::{}\n'.format(
            self.element,
            model.__class__.__name__,
            model.__class__.__name__,
        )

        self.mermaid_string += f"element{element} --> {left} & {right}\n"

        self.element += 1

        return f"element{element}"

    def visit_OperatorScalarMul(self, model):
        expr = self.visit(model.expr)
        op = self.visit(model.op)

        element = self.element
        self.mermaid_string += 'element{}("{}"):::{}\n'.format(
            self.element,
            model.__class__.__name__,
            model.__class__.__name__,
        )

        self.mermaid_string += f"element{element} --> {expr} & {op}\n"

        self.element += 1

        return f"element{element}"


########################################################################################


def mermaid_rules(flowgraph, tabname="Main"):
    mermaid_string = '=== "{}"\n\t'.format(tabname.title())
    mermaid_string += "\n\t".join(
        flowgraph.forward_decorators.rules.accept(MermaidFlowGraph()).splitlines()
    )
    for node in flowgraph.nodes:
        if isinstance(node, FlowGraph):
            mermaid_string += "\n\t".join(
                ("\n" + mermaid_rules(node, node.name) + "\n").splitlines()
            )
    return mermaid_string


def mermaid_traversal(traversal, tabname="Main"):
    mermaid_string = '=== "{}"\n\t'.format(tabname.title())
    mermaid_string += "\n\t".join(traversal.accept(MermaidFlowGraph()).splitlines())
    for site in traversal.sites:
        if site.subtraversal:
            mermaid_string += "\n\t".join(
                (
                    "\n"
                    + mermaid_traversal(
                        site.subtraversal,
                        tabname=f"{site.node} (Site {site.site})",
                    )
                    + "\n"
                ).splitlines()
            )
    return mermaid_string


def markdown_flow(traversal, tabname="Main"):
    md_string = '=== "{}"\n\t'.format(tabname.title())
    model = None
    for site in traversal.sites:
        if site.subtraversal:
            md_string += "\n\t".join(
                markdown_flow(
                    site.subtraversal, tabname=f"{site.node.title()} (Site {site.site})"
                ).splitlines()
            )
            md_string += "\n\t"
            continue
        if site.model and model != site.model:
            md_string += f'=== "{site.node.title()} (Site {site.site})"\n\t\t'
            if isinstance(site.model, MathExpr):
                md_string += "\n\t\t".join(
                    MermaidMathExpr().emit(site.model).splitlines()
                )
            if isinstance(site.model, Operator):
                md_string += "\n\t\t".join(
                    MermaidOperator().emit(site.model).splitlines()
                )
            md_string += "\n\t"
            model = site.model
    return md_string


########################################################################################


class MathCanonicalizationFlow(FlowGraph):
    nodes = [
        FlowTerminal(name="terminal"),
        TransformerFlowNode(visitor=DistributeMathExpr(), name="dist"),
        TransformerFlowNode(visitor=ProperOrderMathExpr(), name="proper"),
        TransformerFlowNode(visitor=PartitionMathExpr(), name="part"),
    ]
    rootnode = "dist"
    forward_decorators = ForwardDecorators()

    @forward_decorators.forward_fixed_point(done="proper")
    def forward_dist(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="part")
    def forward_proper(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="terminal")
    def forward_part(self, model):
        pass

    pass


########################################################################################


def random_mathexpr(terms):
    return MathStr(
        string="(".join(
            [
                "{}{}({}{}{})".format(
                    "-+"[np.random.randint(0, 2)],
                    "\nsin\ncos\ntan\nexp\nlog\nsinh\ncosh\ntanh".splitlines()[
                        np.random.randint(0, 9) * np.random.randint(0, 2)
                    ],
                    "-+"[np.random.randint(0, 2)],
                    [
                        str(np.random.randint(0, 26)),
                        chr(np.random.randint(0, 26) + 97),
                    ][np.random.randint(0, 2)],
                    " j"[np.random.randint(0, 2)],
                )
                + "+*"[np.random.randint(0, 2)]
                for _ in range(terms)
            ]
        )
        + f"{terms}"
        + ")" * (terms - 1)
    )


def random_operator(terms, pauli, ladder, math_terms):
    return reduce(
        (
            lambda a, b: [Operator.__add__, Operator.__mul__][np.random.randint(0, 2)](
                a, b
            )
        ),
        [
            reduce(
                Operator.__matmul__,
                [[I, X, Y, Z][np.random.randint(0, 4)] for _ in range(pauli)]
                + [[J, A, C][np.random.randint(0, 3)] for _ in range(ladder)],
            )
            * (random_mathexpr(math_terms))
            for _ in range(terms)
        ],
    )


########################################################################################


class TestFlowNode(FlowNode):
    def __call__(self, model, traversal=Traversal()) -> FlowOut:
        try:
            traversal.sites[-1].emission["terminate"]
        except:
            return FlowOut(model=model + 1, emission={"terminate": False})

        if not traversal.sites[-1].emission["terminate"]:
            return FlowOut(model=model, emission={"terminate": True})


class TestFlowGraph(FlowGraph):
    nodes = [
        TestFlowNode(name="n1"),
        FlowTerminal(name="terminal"),
    ]
    rootnode = "n1"

    forward_decorators = ForwardDecorators()

    @forward_decorators.catch_error(redirect="terminal")
    @forward_decorators.forward_branch_from_emission(
        key="terminate", branch={True: "terminal", False: "n1"}
    )
    def forward_n1(self, model):
        pass


class TestFlowGraph2(FlowGraph):
    nodes = [
        TestFlowGraph(name="g1"),
        FlowTerminal(name="terminal"),
    ]
    rootnode = "g1"

    forward_decorators = ForwardDecorators()

    @forward_decorators.forward_once(done="terminal")
    def forward_g1(self, model):
        pass


########################################################################################

if __name__ == "__main__":

    model = MathStr(string="1")
    fg = TestFlowGraph2(name="_")
    model = fg(model).model

    ########################################################################################

    # # console = Console(record=True)
    # # with console.capture() as capture:
    # #     console.print(fg.traversal)
    # # string = console.export_text()

    # # with open("_console.py", mode="w", encoding="utf8") as f:
    # #     f.write(string)

    ########################################################################################

    import argparse

    from flowgraph.mkdocs import graph_to_mkdocs

    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()

    fr = fg.forward_decorators.rules
    ft = fg.traversal

    graph_to_mkdocs(
        mermaid_rules(fg), mermaid_traversal(ft), markdown_flow(ft), serve=args.serve
    )

    pass
