# %%
import pathlib
import json
from juliacall import Main as jl

########################################################################################

from quantumion.backend.base import BackendBase
from quantumion.backend.task import Task, TaskResultAnalog
from quantumion.interfaces.analog.circuit import AnalogCircuit

########################################################################################


# %%
class QuantumOpticsBackend(BackendBase):
    name = "quantumoptics"

    def __init__(self):
        # path to the quantumoptics.jl modules
        p = f'include("{pathlib.Path(__file__).parent.joinpath("quantumoptics.jl")}")'
        jl.seval(p)
        return

    def run(self, task: Task) -> TaskResultAnalog:
        assert isinstance(task.program, AnalogCircuit), "Must be AnalogCircuit"
        jl.run(task.model_dump_json())
        result_json = jl.run(task.model_dump_json())
        result = TaskResultAnalog(**json.loads(result_json))
        return result
