# ![Open Quantum Design](https://raw.githubusercontent.com/OpenQuantumDesign/oqd-compiler-infrastructure/main/docs/img/oqd-logo-text.png)

<h2 align="center">
    Open Quantum Design: Core
</h2>


[![doc](https://img.shields.io/badge/documentation-lightblue)](https://docs.openquantumdesign.org/open-quantum-design-core)
[![PyPI Version](https://img.shields.io/pypi/v/oqd-core)](https://pypi.org/project/oqd-core)
[![CI](https://github.com/OpenQuantumDesign/oqd-core/actions/workflows/pytest.yml/badge.svg)](https://github.com/OpenQuantumDesign/oqd-core/actions/workflows/pytest.yml)![versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


## Installation <a name="installation"></a>

### Using pip
```bash
pip install oqd-core
```
or through `git`,

```bash
pip install git+https://github.com/OpenQuantumDesign/oqd-core.git
```

### Using Nix
If you have Nix installed with flakes enabled, you can create a development environment with all dependencies using:

```bash
# Clone the repository
git clone https://github.com/OpenQuantumDesign/oqd-core
cd oqd-core

# Enter the development shell with all dependencies
nix develop

# You can also directly develop without cloning using:
nix develop github:OpenQuantumDesign/oqd-core

# Once in the development shell, you can:
# 1. Run tests: pytest
# 2. Build documentation: mkdocs serve
# 3. Use the package directly in Python

# The development environment includes:
# - Python with all required dependencies
# - Development tools (ruff, black, isort, mypy)
# - Documentation tools (mkdocs)
# - Testing tools (pytest)
```

The `nix develop` command will create an isolated development environment with all the necessary dependencies. This is the recommended way to develop and contribute to the project, as it ensures a consistent development environment across all contributors.

### Development Installation
To develop without Nix, clone the repository locally:

```bash
git clone https://github.com/OpenQuantumDesign/oqd-core
pip install .
```

## Getting Started <a name="Getting Started"></a>
Please see the [documentation](https://docs.openquantumdesign.org) for tutorials, examples, and API reference.
Below is a simple example of an analog quantum program, in which a single qubit evolves under 
a $\sigma_X$ Hamiltonian for 10 units of time and is then measured.
```python
from oqd_core.interface.analog.operator import *
from oqd_core.interface.analog.operation import *

X = PauliX()
Z = PauliZ()

Hx = AnalogGate(hamiltonian=X)

circuit = AnalogCircuit()
circuit.evolve(duration=10, gate=Hx)
circuit.measure()
```

## Documentation <a name="documentation"></a>

Documentation is implemented with [MkDocs](https://www.mkdocs.org/).
To install the dependencies for documentation, run:

```
pip install -e ".[docs]"
```

To deploy the documentation server locally:

```
cp -r examples/ docs/examples/
mkdocs serve
```

### Where in the stack
```mermaid
block-beta
   columns 3
   
   block:Interface
       columns 1
       InterfaceTitle("<i><b>Interfaces</b><i/>")
       InterfaceDigital["<b>Digital Interface</b>\nQuantum circuits with discrete gates"] 
       space
       InterfaceAnalog["<b>Analog Interface</b>\n Continuous-time evolution with Hamiltonians"] 
       space
       InterfaceAtomic["<b>Atomic Interface</b>\nLight-matter interactions between lasers and ions"]
       space
    end
    
    block:IR
       columns 1
       IRTitle("<i><b>IRs</b><i/>")
       IRDigital["Quantum circuit IR\nopenQASM, LLVM+QIR"] 
       space
       IRAnalog["openQSIM"]
       space
       IRAtomic["openAPL"]
       space
    end
    
    block:Emulator
       columns 1
       EmulatorsTitle("<i><b>Classical Emulators</b><i/>")
       
       EmulatorDigital["Pennylane, Qiskit"] 
       space
       EmulatorAnalog["QuTiP, QuantumOptics.jl"]
       space
       EmulatorAtomic["TrICal, QuantumIon.jl"]
       space
    end
    
    space
    block:RealTime
       columns 1
       RealTimeTitle("<i><b>Real-Time</b><i/>")
       space
       RTSoftware["ARTIQ, DAX, OQDAX"] 
       space
       RTGateware["Sinara Real-Time Control"]
       space
       RTHardware["Lasers, Modulators, Photodetection, Ion Trap"]
       space
       RTApparatus["Trapped-Ion QPU (<sup>171</sup>Yt<sup>+</sup>, <sup>133</sup>Ba<sup>+</sup>)"]
       space
    end
    space
    
   InterfaceDigital --> IRDigital
   InterfaceAnalog --> IRAnalog
   InterfaceAtomic --> IRAtomic
   
   IRDigital --> IRAnalog
   IRAnalog --> IRAtomic
   
   IRDigital --> EmulatorDigital
   IRAnalog --> EmulatorAnalog
   IRAtomic --> EmulatorAtomic
   
   IRAtomic --> RealTimeTitle
   
   RTSoftware --> RTGateware
   RTGateware --> RTHardware
   RTHardware --> RTApparatus
   
    classDef title fill:#d6d4d4,stroke:#333,color:#333;
    classDef digital fill:#E7E08B,stroke:#333,color:#333;
    classDef analog fill:#E4E9B2,stroke:#333,color:#333;
    classDef atomic fill:#D2E4C4,stroke:#333,color:#333;
    classDef realtime fill:#B5CBB7,stroke:#333,color:#333;

    classDef highlight fill:#f2bbbb,stroke:#333,color:#333,stroke-dasharray: 5 5;
    
    class InterfaceTitle,IRTitle,EmulatorsTitle,RealTimeTitle title
    class InterfaceDigital,IRDigital,EmulatorDigital digital
    class InterfaceAnalog,IRAnalog,EmulatorAnalog analog
    class InterfaceAtomic,IRAtomic,EmulatorAtomic atomic
    class RTSoftware,RTGateware,RTHardware,RTApparatus realtime
   
   class Interface,IRAnalog,IRAtomic highlight
```
