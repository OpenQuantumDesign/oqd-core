import tensorcircuit as tc
# from tensorcircuit.translation import
from quantumion.digital.circuit import DigitalCircuit
from backends.task import Task
from backends.digital.data import TaskArgsDigital, TaskResultDigital

import matplotlib.pyplot as plt


class TensorCircuitBackend:

    def __init__(self):

        self.kwargs = {}

    def run(self, task: Task) -> TaskResultDigital:
        assert isinstance(task.program, DigitalCircuit), "Wrong program type"
        circuit = task.program
        qasm = circuit.qasm
        print(qasm)
        circ = tc.Circuit.from_openqasm(qasm)

        print(circ.draw(output='text'))

        result = TaskResultDigital(
            counts=circ.sample(batch=task.args.n_shots, allow_state=True, format='count_dict_bin'),
            state=circ.state(),
        )
        return result
