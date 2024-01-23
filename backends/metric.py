from typing import Literal, List, TypedDict, Union
from pydantic import BaseModel
from pydantic.types import NonNegativeInt

from quantumion.analog.operator import Operator


class Expectation(BaseModel):
    operator: List[Operator]


class EntanglementEntropyVN(BaseModel):
    qreg: List[NonNegativeInt] = []
    qmode: List[NonNegativeInt] = []


class EntanglementEntropyReyni(BaseModel):
    alpha: NonNegativeInt = 1
    qreg: List[NonNegativeInt] = []
    qmode: List[NonNegativeInt] = []


Metric = Union[EntanglementEntropyVN, EntanglementEntropyReyni, Expectation]