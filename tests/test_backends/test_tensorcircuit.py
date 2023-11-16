# %%
from rich import print

from backends.digital.python.tc import TensorCircuitBackend
from backends.task import Task, TaskArgsDigital

from quantumion.digital.circuit import DigitalCircuit
from quantumion.digital.gate import H, CNOT
from quantumion.digital.register import QuantumRegister, ClassicalRegister

# %%
qreg = QuantumRegister(id="q", reg=2)
creg = ClassicalRegister(id="c", reg=2)

circ = DigitalCircuit(qreg=qreg, creg=creg)
circ.add(H(qreg=qreg[0]))
circ.add(CNOT(qreg=qreg[0:2]))
# circ.add(Measure())
print(circ)

# %%
args = TaskArgsDigital(repetitions=10)
task = Task(program=circ, args=args)

# %%
backend = TensorCircuitBackend()
backend.run(task)

# %%
