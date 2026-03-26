

from oqd_compiler_infrastructure import TypeReflectBaseModel


class AnalogCircuit(TypeReflectBaseModel):
    
    sequence: List[...]
    
class Evolve(TypeReflectBaseModel):
    hamiltonian: ...
    duration: float
    targets: AtomicTypes
    
class QuantumRegister(TypeReflectBaseModel):
    size: int

class Declaration(TypeReflectBaseModel):
    name: str
    value: Union[QuantumRegister, QuantumBit]

class QuantumBit(TypeReflectBaseModel):
    name: str
    index: int

class MyList(TypeReflectBaseModel):
    values: List[AtomicTypes]

AtomicTypes = Union[QuantumBit, QuantumRegister, MyList, Access]

class Access(TypeReflectBaseModel):
    name: resctricted_type

# resctricted_type = Annotated[str, lambda x: x.isidentifier()]


def _is_varname(value: str) -> str:
    if not value.isidentifier():
        raise ValueError
    return value


resctricted_type = Annotated[str, AfterValidator(_is_varname)]
