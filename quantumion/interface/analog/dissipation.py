from quantumion.interface.base import VisitableBaseModel

########################################################################################

__all__ = [
    "Dissipation",
]

########################################################################################


class Dissipation(VisitableBaseModel):
    """
    Class representing a dissipative term in the evolution of a quantum system
    """

    jumps: int = None  # todo: discuss ir for dissipation
