from quantumion.datastruct.base import VisitableBaseModel

########################################################################################

__all__ = [
    "Dissipation",
]

########################################################################################


class Dissipation(VisitableBaseModel):
    jumps: int = None  # todo: discuss ir for dissipation