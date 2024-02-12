import argparse

import numpy as np

import time
from timeit import default_timer

########################################################################################

from quantumion.backend.client import Client
from quantumion.backend.provider import Provider

from quantumion.backend.task import Task, TaskArgsAnalog, TaskArgsDigital

from quantumion.interface.analog import *
from quantumion.interface.digital import *

########################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u",
        "--url",
        default="http://127.0.0.1:8000",
        type=str,
        help="server URL for job submission",
    )

    args = parser.parse_args()

    url = args.url

    ########################################################################################

    provider = Provider(url)
    client = Client()

    client.connect(provider)

    ########################################################################################

    op = np.pi * PauliX @ PauliX
    gate = AnalogGate(duration=1 / 4, hamiltonian=[op], dissipation=[])
    ex = AnalogCircuit()
    ex.evolve(gate=gate)

    analog_args = TaskArgsAnalog(n_shots=1)
    analog_task = Task(program=ex, args=analog_args)

    ########################################################################################

    qreg = QuantumRegister(id="q", reg=2)
    creg = ClassicalRegister(id="c", reg=2)
    circ = DigitalCircuit(qreg=qreg, creg=creg)
    circ.add(H(qreg=qreg[0]))
    circ.add(CNOT(qreg=qreg[0:2]))

    digital_args = TaskArgsDigital(repetitions=1)
    digital_task = Task(program=circ, args=digital_args)

    ########################################################################################

    t0 = default_timer()
    for i in range(3):
        client.submit_job(analog_task, backend="qutip")
        client.submit_job(digital_task, backend="tensorcircuit")

    ########################################################################################

    while client.pending:
        client.status_update()

        queued = client.status_report["queued"]["count"]
        print(
            "\r{} Jobs Pending ...".format(queued),
            end="",
        )

        time.sleep(queued / 4)

    print("\n{:<48} {:<16} {}".format("Job ID", "Status", "Results"))

    for job in client.jobs.values():
        print("{:<48} {:<16} {}".format(job.job_id, job.status, job.result))

    ########################################################################################

    t = default_timer() - t0
    print(f"Totat time taken (s): {t}s")
