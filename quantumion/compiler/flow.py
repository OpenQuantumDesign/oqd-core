from typing import Any, List, Optional, Union, Dict

from abc import ABC, abstractmethod, abstractproperty

import functools

from pydantic import field_validator

########################################################################################

from quantumion.interface.base import TypeReflectBaseModel
from quantumion.compiler.visitor import Visitor
from quantumion.compiler.analog import *
from quantumion.compiler.math import *

########################################################################################


class TraversalSite(TypeReflectBaseModel):
    site: str
    node: str
    subtraversal: Optional["Traversal"] = None
    emission: Any = None


class Traversal(TypeReflectBaseModel):
    sites: List[TraversalSite] = []

    class Config:
        validate_assignment = True


########################################################################################


class ForwardRule(TypeReflectBaseModel):
    name: str
    decorators: List[str]
    destinations: Dict[str, str]

    class Config:
        validate_assignment = True


class ForwardRules(TypeReflectBaseModel):
    rules: List[ForwardRule] = []

    class Config:
        validate_assignment = True


########################################################################################


class ForwardDecorators:
    def __init__(self):
        self._rules = ForwardRules()

    @property
    def rules(self):
        return self._rules

    def update_rule(self, forward_rule):
        rules = self._rules.rules

        rule = next((rule for rule in rules if rule.name == forward_rule.name), None)
        if rule:
            decorators = rule.decorators
            destinations = rule.destinations

            decorators += forward_rule.decorators
            destinations.update(forward_rule.destinations)

            rule.decorators = decorators
            rule.destinations = destinations
        else:
            rule = forward_rule
            rules.append(rule)

        self._rules.rules = rules

    def forward_once(self, done):
        def _forward_once(method):
            self.update_rule(
                ForwardRule(
                    name=method.__name__,
                    decorators=[
                        "forward_once",
                    ],
                    destinations=dict(done=done),
                )
            )

            @functools.wraps(method)
            def _method(self, model: Any) -> FlowOut:
                flowout = self.namespace[self.current_node](model)

                self.next_node = done

                return flowout

            return _method

        return _forward_once

    def forward_fixed_point(self, done):
        def _forward_fixed_point(method):
            self.update_rule(
                ForwardRule(
                    name=method.__name__,
                    decorators=[
                        "forward_fixed_point",
                    ],
                    destinations=dict(done=done),
                )
            )

            @functools.wraps(method)
            def _method(self, model: Any) -> FlowOut:
                flowout = self.namespace[self.current_node](model)

                if model == flowout.model:
                    self.next_node = done
                else:
                    self.next_node = self.current_node

                return flowout

            return _method

        return _forward_fixed_point

    def forward_detour(self, done, detour):
        def _forward_detour(method):
            self.update_rule(
                ForwardRule(
                    name=method.__name__,
                    decorators=[
                        "forward_detour",
                    ],
                    destinations=dict(done=done, detour=detour),
                ),
            )

            @functools.wraps(method)
            def _method(self, model: Any) -> FlowOut:
                flowout = self.namespace[self.current_node](model)

                if model == flowout.model:
                    self.next_node = done
                else:
                    self.next_node = detour

                return flowout

            return _method

        return _forward_detour

    def catch_error(self, redirect):
        def _catch_error(method):
            self.update_rule(
                ForwardRule(
                    name=method.__name__,
                    decorators=[
                        "catch_error",
                    ],
                    destinations=dict(redirect=redirect),
                )
            )

            @functools.wraps(method)
            def _method(self, model: Any) -> FlowOut:
                try:
                    flowout = method(self, model)
                    return flowout
                except Exception as e:
                    self.next_node = redirect
                    return FlowOut(model=model, emission=e)

            return _method

        return _catch_error

    def catch_errors_and_branch(self, branch):
        def _catch_errors_and_branch(method):
            self.update_rule(
                ForwardRule(
                    name=method.__name__,
                    decorators=[
                        "catch_error_and_branch",
                    ],
                    destinations={f"{k.__name__}_branch": v for k, v in branch.items()},
                )
            )

            @functools.wraps(method)
            def _method(self, model: Any) -> FlowOut:
                try:
                    flowout = method(self, model)
                    return flowout
                except tuple(branch.keys()) as e:
                    self.next_node = branch[e.__class__]
                    return FlowOut(model=model, emission=e)

            return _method

        return _catch_errors_and_branch


########################################################################################


class FlowError(Exception):
    pass


########################################################################################


class FlowBase(ABC):
    def __init__(self, name, **kwargs):
        self.name = name
        pass

    @abstractmethod
    def __call__(self, model: Any) -> "FlowOut":
        pass

    @abstractproperty
    def traversal(self) -> Union[Traversal, None]:
        pass

    pass


class FlowOut(TypeReflectBaseModel):
    model: Any
    emission: Any = None


########################################################################################


class FlowNode(FlowBase):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    def __call__(self, model: Any) -> FlowOut:
        raise NotImplementedError

    @property
    def traversal(self) -> None:
        return None

    pass


class FlowTerminal(FlowNode):
    def __call__(self, model: Any):
        raise NotImplementedError

    pass


class VisitorFlowNode(FlowNode):
    def __init__(self, visitor, **kwargs):
        self.visitor = visitor
        super().__init__(**kwargs)
        pass

    def __call__(self, model: Any) -> FlowOut:
        emission = self.visitor.emit(model)
        return FlowOut(model=model, emission=emission)

    pass


class TransformFlowNode(VisitorFlowNode):
    def __call__(self, model: Any) -> FlowOut:
        return FlowOut(model=model.accept(self.visitor))


########################################################################################


class FlowGraph(FlowBase):
    nodes: List[FlowBase] = [FlowTerminal(name="terminal")]
    rootnode: str = ""

    @property
    def namespace(self):
        _namespace = {}
        overlap = set()
        for node in self.nodes:
            if node.name in _namespace.keys():
                overlap.add(node.name)
            _namespace[node.name] = node

        if overlap:
            raise NameError(
                "Multiple nodes with names: {}".format(
                    ", ".join({f'"{name}"' for name in overlap})
                )
            )

        return _namespace

    def __init__(self, max_steps=1000, **kwargs):
        self.max_steps = max_steps

        if FlowTerminal not in [node.__class__ for node in self.nodes]:
            raise FlowError("FlowGraph does not contain a terminal node")

        super().__init__(**kwargs)
        pass

    @property
    def exceeded_max_steps(self):
        if self.current_step >= self.max_steps:
            raise RecursionError(f"Exceeded number of steps allowed for {self}")
        pass

    @property
    def current_node(self):
        return self._current_node

    @current_node.setter
    def current_node(self, value):
        if value in self.namespace.keys():
            self._current_node = value
            return
        raise NameError(f'No node named "{value}" in namespace of {self}')

    @property
    def next_node(self):
        return self._next_node

    @next_node.setter
    def next_node(self, value):
        if value in self.namespace.keys():
            self._next_node = value
            return
        raise NameError(f'No node named "{value}" in namespace of {self}')

    @next_node.deleter
    def next_node(self):
        del self._next_node
        pass

    @property
    def current_step(self):
        return self._current_step

    @property
    def traversal(self) -> Traversal:
        return self._traversal

    def __iter__(self):
        self.current_node = self.rootnode
        self._current_step = 0
        self._traversal = Traversal()
        return self

    def __next__(self):
        self.exceeded_max_steps

        if isinstance(self.namespace[self.current_node], FlowTerminal):
            self._traversal.sites += [
                TraversalSite(
                    site=str(self.current_step),
                    node=self.current_node,
                ),
            ]
            raise StopIteration

        def _forward(model: Any):
            flowout = getattr(self, "forward_{}".format(self.current_node))(model)

            self._traversal.sites += [
                TraversalSite(
                    site=str(self.current_step),
                    node=self.current_node,
                    subtraversal=self.namespace[self.current_node].traversal,
                    emission=flowout.emission,
                ),
            ]

            self.current_node = self.next_node
            self._current_step += 1

            del self.next_node

            return flowout.model

        return _forward

    def __call__(self, model) -> FlowOut:
        for node in self:
            model = node(model)
        return FlowOut(model=model)


########################################################################################


class NormalOrderFlow(FlowGraph):
    nodes = [
        TransformFlowNode(visitor=NormalOrder(), name="normal"),
        TransformFlowNode(visitor=OperatorDistribute(), name="distribute"),
        TransformFlowNode(visitor=GatherMathExpr(), name="gathermath"),
        TransformFlowNode(visitor=ProperOrder(), name="proper"),
        TransformFlowNode(visitor=PruneIdentity(), name="prune"),
        FlowTerminal(name="terminal"),
    ]
    rootnode = "normal"
    forward_decorators = ForwardDecorators()

    @forward_decorators.forward_once(done="distribute")
    def forward_normal(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="gathermath")
    def forward_distribute(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="proper")
    def forward_gathermath(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="prune")
    def forward_proper(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="terminal")
    def forward_prune(self, model):
        pass


class CanonicalizationFlow(FlowGraph):
    nodes = [
        VisitorFlowNode(visitor=VerifyHilbertSpace(), name="hspace"),
        TransformFlowNode(visitor=OperatorDistribute(), name="distribute"),
        TransformFlowNode(visitor=GatherMathExpr(), name="gathermath"),
        TransformFlowNode(visitor=ProperOrder(), name="proper"),
        TransformFlowNode(visitor=PauliAlgebra(), name="paulialgebra"),
        TransformFlowNode(visitor=GatherMathExpr(), name="gathermath2"),
        TransformFlowNode(visitor=GatherPauli(), name="gatherpauli"),
        NormalOrderFlow(name="normalflow"),
        TransformFlowNode(visitor=SortedOrder(), name="sorted"),
        TransformFlowNode(visitor=DistributeMathExpr(), name="distmath"),
        TransformFlowNode(visitor=ProperOrderMathExpr(), name="propermath"),
        TransformFlowNode(visitor=PartitionMathExpr(), name="partmath"),
        FlowTerminal(name="terminal"),
    ]
    rootnode = "hspace"
    forward_decorators = ForwardDecorators()

    @forward_decorators.forward_once(done="distribute")
    def forward_hspace(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="gathermath")
    def forward_distribute(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="proper")
    def forward_gathermath(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="paulialgebra")
    def forward_proper(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="gathermath2")
    def forward_paulialgebra(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="gatherpauli")
    def forward_gathermath2(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="normalflow")
    def forward_gatherpauli(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="sorted")
    def forward_normalflow(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="distmath")
    def forward_sorted(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="propermath")
    def forward_distmath(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="partmath")
    def forward_propermath(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="terminal")
    def forward_partmath(self, model):
        pass


########################################################################################


class CanonicalizationFlow2(FlowGraph):
    nodes = [
        VisitorFlowNode(visitor=VerifyHilbertSpace(), name="hspace"),
        TransformFlowNode(visitor=OperatorDistribute(), name="distribute"),
        TransformFlowNode(visitor=GatherMathExpr(), name="gathermath"),
        TransformFlowNode(visitor=ProperOrder(), name="proper"),
        TransformFlowNode(visitor=PauliAlgebra(), name="paulialgebra"),
        TransformFlowNode(visitor=GatherMathExpr(), name="gathermath2"),
        TransformFlowNode(visitor=GatherPauli(), name="gatherpauli"),
        TransformFlowNode(visitor=NormalOrder(), name="normal"),
        TransformFlowNode(visitor=OperatorDistribute(), name="distribute2"),
        TransformFlowNode(visitor=GatherMathExpr(), name="gathermath3"),
        TransformFlowNode(visitor=ProperOrder(), name="proper2"),
        TransformFlowNode(visitor=PruneIdentity(), name="prune"),
        TransformFlowNode(visitor=SortedOrder(), name="sorted"),
        TransformFlowNode(visitor=DistributeMathExpr(), name="distmath"),
        TransformFlowNode(visitor=ProperOrderMathExpr(), name="propermath"),
        TransformFlowNode(visitor=PartitionMathExpr(), name="partmath"),
        FlowTerminal(name="terminal"),
    ]
    rootnode = "hspace"
    forward_decorators = ForwardDecorators()

    @forward_decorators.forward_once(done="distribute")
    def forward_hspace(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="gathermath")
    def forward_distribute(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="proper")
    def forward_gathermath(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="paulialgebra")
    def forward_proper(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="gathermath2")
    def forward_paulialgebra(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="gatherpauli")
    def forward_gathermath2(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="normal")
    def forward_gatherpauli(self, model):
        pass

    @forward_decorators.forward_detour(done="sorted", detour="distribute2")
    def forward_normal(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="gathermath3")
    def forward_distribute2(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="proper2")
    def forward_gathermath3(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="prune")
    def forward_proper2(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="normal")
    def forward_prune(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="distmath")
    def forward_sorted(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="propermath")
    def forward_distmath(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="partmath")
    def forward_propermath(self, model):
        pass

    @forward_decorators.forward_fixed_point(done="terminal")
    def forward_partmath(self, model):
        pass
