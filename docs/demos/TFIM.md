# Transverse Field Ising Model

Let's implement our favourite Hamiltonian -- the transverse-field Ising model.
The general Hamiltonian looks like,
$$
H = \sum_{\langle ij \rangle} \sigma^x_i \sigma^x_j + h \sum_i \sigma^z_i
$$

Let's implement it with two qubits and with $h=1$.
$$
H = \sigma^x_1 \sigma^x_2 + \sigma^z_1 + \sigma^z_2
$$

We will go through this step by step. First we get the necessary imports:
``` py
from quantumion.interface.analog.operator import *
from quantumion.interface.analog.dissipation import Dissipation
from quantumion.interface.analog.operations import *
from quantumion.compiler.analog.interface import *
from quantumion.backend.metric import *
from quantumion.backend.task import Task, TaskArgsAnalog
from quantumion.backend import QutipBackend
from examples.emulation.utils import plot_metrics_counts
from rich import print as pprint
```

Then we define the `AnalogGate` object

``` py
"""For simplicity we initialize all Operators"""
X, Y, Z, I, A, C, J = PauliX(), PauliY(), PauliZ(), PauliI(), Annihilation(), Creation(), Identity()
    
H = AnalogGate(hamiltonian= (X @ X) + (Z @ I) + (I @ Z), dissipation=Dissipation())
```

Then we define the `AnalogCircuit` object and evolve it according to the hamiltonian defined above

``` py
circuit = AnalogCircuit()
circuit.evolve(duration = 5, gate = H)
```

For QuTip simulation we need to define the arguements which contain the number of shots and the metrics we want to evaluate.
``` py
args = TaskArgsAnalog(
    n_shots=100,
    fock_cutoff=4,
    metrics={
        'entanglement_entropy': EntanglementEntropyVN(qreg = [1]),
    },
    dt=1e-2,
)
```

We can then wrap the `AnalogCircuit` and the args to a `Task` object and run using the QuTip backend. Note that there are 2 ways to run and the 2 ways are explained.

### Compile and then Simulate

The `Task` can be compiled first to a `QuTipExperiment` object and then this `QuTipExperiment` object can be run. This is to allow you to see what parameters are used to specify the particular QuTip experiment.

``` py
experiment = backend.compile(task = task)
results = backend.run(experiment = experiment)
```

### Directly Simulate

The `Task` object can be directly simulated by the `run()` method. 

``` py
results = backend.run(task = task)
```

Finally we can plot the metrics and relevant statistics from the final quantum state:


``` py
plot_metrics_counts(results = results, experiment_name = 'tfim_2_site.png')
```

The generated image is like:

<!-- ![Two Site TFIM](img/plots/tfim_2_site.png)  -->


![Entropy of entanglement](../img/plots/tfim_2_site.png) 