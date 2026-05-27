This module contains tools that allow for dataflow analysis and optimizations for both analog and atomic frontends. The code lies in [`utils.py`](../src/oqd_core/analysis/utils.py).

## Control Flow Graph (CFG)
A generic definition for a Control Flow Graph Node is defined by the `CFGNode` class. This can be used to construct a CFG for type checking, and dataflow analysis.

## Strongly Connected Components (SCC)
The `SCCAnalysis` class identifies the strongly connected components of the CFG to check for infinite loops in the program. The implementation follows [Tarjan's algorithm](https://www.geeksforgeeks.org/dsa/tarjan-algorithm-find-strongly-connected-components/).

