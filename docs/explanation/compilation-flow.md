# Compilation flow

## Examples
=== "FlowNode"
    ```mermaid
    flowchart LR

    START:::hidden -- Input --> node(FlowNode) -- Output --> END:::hidden
    node -. Emission .-> MID:::hidden
    classDef hidden display: none;
    ```

=== "FlowGraph"
    === "Only FlowNodes"
        ```mermaid
        flowchart LR
        
        START:::hidden -- Input --> node1(FlowNode 1) --> change1{Change} -- no --> node2(FlowNode 2) --> terminal(Terminal) -- Output --> END:::hidden

        change1 -- yes --> node1

        node1 -. Emission .-> MID1:::hidden
        node2 -. Emission .-> MID2:::hidden
    
        classDef hidden display:none;
        ```
    === "In General"
        ```mermaid
        flowchart LR

        START:::hidden -- Input --> node1(FlowNode 1) --> change1{Change} -- no --> graph1{{"FlowGraph1<br/>(See below)"}} --> change2{Change} -- no --> terminal2(Terminal) -- Output --> END:::hidden

        change1 -- yes --> node1
        change2 -- yes --> graph1

        node1 -. Emission .-> MID1:::hidden

        classDef hidden display: none;
        ```

        /// details | FlowGraph1
        ```mermaid
        flowchart LR

        START:::hidden -- Input --> node2(FlowNode 2) --> change2{Change} -- no --> node3(FlowNode 3) --> terminal(Terminal) -- Output --> END:::hidden

        change2 -- yes --> node4(FlowNode 4) --> change4{Change} -- no --> node2
        change4 -- yes --> node4

        node2 -. Emission .-> MID2:::hidden
        node3 -. Emission .-> MID3:::hidden
        node4 -. Emission .-> MID4:::hidden

        classDef hidden display: none;
        ```
        ///

## Forward Decorators
=== "Forward Once"
    ```python linenums="1"
    forward_decorators = ForwardDecorators()

    @forward_decorators.forward_once(done="FN2")
    def forward_FN1(self, model):
        pass
    ```
    ```mermaid
    flowchart LR
    
    node1("FlowNode1<br/>----------<br/>name: FN1")
    node2("FlowNode2<br/>----------<br/>name: FN2")

    
    START:::hidden -- Input --> node1 --> node2 -- continue --> END:::hidden

    node1 -. Emission .-> MID1:::hidden

    classDef hidden display: none;
    ```
=== "Forward Fixed Point"
    ```python linenums="1"
    forward_decorators = ForwardDecorators()

    @forward_decorators.forward_fixed_point(done="FN2")
    def forward_FN1(self, model):
        pass
    ```
    ```mermaid
    flowchart LR
    
    node1("FlowNode1<br/>----------<br/>name: FN1")
    node2("FlowNode2<br/>----------<br/>name: FN2")

    
    START:::hidden -- Input --> node1 --> change1{Change} -- no --> node2 -- continue --> END:::hidden
    
    change1 -- yes --> node1

    node1 -. Emission .-> MID1:::hidden

    classDef hidden display: none;
    ```
=== "Forward Detour"
    ```python linenums="1"
    forward_decorators = ForwardDecorators()

    @forward_decorators.forward_detour(done="FN2", detour="FN3")
    def forward_FN1(self, model):
        pass
    ```
    ```mermaid
    flowchart LR
    
    node1("FlowNode1<br/>----------<br/>name: FN1")
    node2("FlowNode2<br/>----------<br/>name: FN2")
    node3("FlowNode3<br/>----------<br/>name: FN3")


    START:::hidden -- Input --> node1 --> change1{Change} -- no --> node2 -- continue --> END:::hidden
    
    change1 -- yes --> node3 -- continue --> END2:::hidden

    node1 -. Emission .-> MID1:::hidden

    classDef hidden display: none;
    ```
=== "Forward Return"
    ```python linenums="1"
    forward_decorators = ForwardDecorators()

    @forward_decorators.forward_return()
    def forward_FN3(self, model):
        pass
    ```
    ```mermaid
    flowchart LR
    
    node1("FlowNode1<br/>----------<br/>name: FN1")
    node2("FlowNode2<br/>----------<br/>name: FN2")
    node3("FlowNode3<br/>----------<br/>name: FN3")

    
    START1:::hidden -- from --> node1 -- continue --> END1:::hidden
    START2:::hidden -- from --> node2 -- continue --> END2:::hidden

    node1 -. Emission .-> MID1:::hidden
    node2 -. Emission .-> MID2:::hidden

    node1 <-- round trip --> node3
    node2 <-- round trip --> node3

    classDef hidden display: none;
    ```
=== "Forward Branch from Emission"
    ```python linenums="1"
    forward_decorators = ForwardDecorators()
    
    @forward_decorators.forward_branch_from_emission(
            key="dest", branch={"fn2": "FN2", "fn3": "FN3"}
        )
    def forward_FN1(self, model):
        pass
    ```
    ```mermaid
    flowchart LR
    
    node1("FlowNode1<br/>----------<br/>name: FN1")
    node2("FlowNode2<br/>----------<br/>name: FN2")
    node3("FlowNode3<br/>----------<br/>name: FN3")

    
    START1:::hidden -- from --> node1 --> branch{Branch}

    node1 -. Emission .-> branch

    branch -- "(emission.dest == fn2)" --> node2 -- continue --> END1:::hidden
    branch -- "(emission.dest == fn3)" --> node3 -- continue --> END2:::hidden

    node1 -. Emission .-> MID:::hidden

    node1 

    classDef hidden display: none;
    ```

=== "Forward Branch from Subgraph Exit"
    ```python linenums="1"
    forward_decorators = ForwardDecorators()
    
    @forward_decorators.forward_branch_from_subgraph_exit(
            branch={"terminal1": "FN2", "terminal2": "FN3"}
        )
    def forward_FG1(self, model):
        pass
    ```
    ```mermaid
    flowchart LR

    node2("FlowNode2<br/>----------<br/>name: FN2")
    node3("FlowNode3<br/>----------<br/>name: FN3")
    
    subgraph graph1 ["FlowGraph1 (name: FG1)"]
        direction LR
        terminal1("FlowTerminal1<br/>----------<br/>name: terminal1")
        terminal2("Flowterminal2<br/>----------<br/>name: terminal2")

        START1:::hidden -- from --> terminal1
        START2:::hidden -- from --> terminal2
    end
    
    
    terminal1 & terminal2 --> branch{Branch}



    branch -- terminal1 --> node2 -- continue --> END1:::hidden
    branch -- terminal2 --> node3 -- continue --> END2:::hidden

    classDef hidden display: none;
    ```
=== "Catch Error"
    ```python linenums="1"
    forward_decorators = ForwardDecorators()

    @forward_decorators.catch_error(redirect="FN3")
    @forward_decorators.forward_once(done="FN2")
    def forward_FN1(self, model):
        pass
    ```
    ```mermaid
    flowchart LR
    
    node1("FlowNode1<br/>----------<br/>name: FN1")
    node2("FlowNode2<br/>----------<br/>name: FN2")
    node3("FlowNode3<br/>----------<br/>name: FN3")
    
    START:::hidden -- Input --> node1 --> status{Status} -- success --> node2 -- continue --> END:::hidden

    status -- error --> node3 -- continue --> END2:::hidden
    status -. Emission .-> MID1:::hidden

    classDef hidden display: none;
    ```
=== "Catch Error and Branch"
    ```python linenums="1"
    forward_decorators = ForwardDecorators()

    @forward_decorators.catch_error_and_branch(
        branch={TypeError: "FN3", ValueError: "FN4"},
    )
    @forward_decorators.forward_once(done="FN2")
    def forward_FN1(self, model):
        pass
    ```
    ```mermaid
    flowchart LR
    
    node1("FlowNode1<br/>----------<br/>name: FN1")
    node2("FlowNode2<br/>----------<br/>name: FN2")
    node3("FlowNode3<br/>----------<br/>name: FN3")
    node4("FlowNode4<br/>----------<br/>name: FN4")
    
    START:::hidden -- Input --> node1 --> status{Status} -- success --> node2 -- continue --> END:::hidden

    status -- error --> branch{Branch}
    branch -- TypeError --> node3 -- continue --> END2:::hidden
    branch -- ValueError --> node4 -- continue --> END3:::hidden
    
    status -. Emission .-> MID1:::hidden

    classDef hidden display: none;
    ```

### Forward Rules
A FlowGraph can instantiate a ForwardDecorators object as a class attribute. When a method of ForwardDecorators are used, the logic is recorded to a ForwardRules object as a ForwardRule object.
=== "Forward Once"
    ```python linenums="1"
    ForwardRule(
        class_="ForwardRule",
        name="forward_FN1",
        decorators=["forward_once"],
        destinations={"done": "FN2"},
    )
    ```
=== "Forward Fixed Point"
    ```python linenums="1"
    ForwardRule(
        class_="ForwardRule",
        name="forward_FN1",
        decorators=["forward_fixed_point"],
        destinations={"done": "FN2"},
    )
    ```
=== "Forward Detour"
    ```python linenums="1"
    ForwardRule(
        class_="ForwardRule",
        name="forward_FN1",
        decorators=["forward_detour"],
        destinations={"done": "FN2", "detour": "FN3"},
    )
    ```
=== "Forward Return"
    ```python linenums="1"
    ForwardRule(
        class_="ForwardRule",
        name="forward_FN3",
        decorators=["forward_return"],
        destinations={},
    )
    ```
=== "Forward Branch from Emission"
    ```python linenums="1"
    ForwardRule(
        class_="ForwardRule",
        name="forward_FN1",
        decorators=["forward_branch_from_emission"],
        destinations={
            "emission.dest==fn2_branch": "FN2",
            "emission.dest==fn3_branch": "FN3",
        },
    )
    ```
=== "Forward Branch from Subgraph Exit"
    ```python linenums="1"
    ForwardRule(
        class_="ForwardRule",
        name="forward_FG1",
        decorators=["forward_branch_from_subgraph_exit"],
        destinations={"terminal1_branch": "FN2", "terminal2_branch": "FN3"},
    )

    ```
=== "Catch Error"
    ```python linenums="1"
    ForwardRule(
        class_="ForwardRule",
        name="forward_FN1",
        decorators=["forward_once", "catch_error"],
        destinations={"done": "FN2", "redirect": "FN3"},
    )
    ```
=== "Catch Error and Branch"
    ```python linenums="1"
    ForwardRule(
        class_="ForwardRule",
        name="forward_FN1",
        decorators=["forward_once", "catch_error_and_branch"],
        destinations={
            "done": "FN2",
            "TypeError_branch": "FN3",
            "ValueError_branch": "FN4",
        },
    )
    ```

## Traversal
```mermaid
flowchart LR

START:::hidden -- Input --> node1

node1("FlowNode 1<br/>----------<br/>name: FN1<br/>site: 0, 1, 2, 4")
node2("FlowNode 2<br/>----------<br/>name: FN2<br/>site: 3 and 5")
graph1("FlowGraph 1<br/>----------<br/>name: FG1<br/>site: 6 and 7")
terminal("Terminal<br/>----------<br/>name: terminal<br/>site: 8")

node1 -- "(0,1)" --> node1 -- "(1,2)" --> node1 -- "(2,3)" --> node2

node2 -- "(3,4)" --> node1 -- "(4,5)" --> node2 -- "(5,6)" --> graph1

graph1 -- "(6,7)" --> graph1 -- "(7,8)" --> terminal

terminal -- Output --> END:::hidden

classDef hidden display: none;
```
The Traversal records each site it passes through as a TraversalSite:

```python linenums="1" 
traversal.sites[0] = TraversalSite(
    site=0, node="FN1", subtraversal=None, emission=...
)
traversal.sites[1] = TraversalSite(
    site=1, node="FN1", subtraversal=None, emission=...
)
...
traversal.sites[6] = TraversalSite(
    site=6, node="FG1", subtraversal=Traversal(...), emission=None
)
...
travelsal.sites[8] = TraversalSite(
    site=8, node="terminal", subtraversal=None, emission=None
)
```