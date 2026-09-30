# Copyright 2024-2025 Open Quantum Design

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import inspect
import warnings
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

########################################################################################

__all__ = ["MetaBackendRegistry", "BackendRegistry", "BackendBase", "BackendOptions"]

########################################################################################


class MetaBackendRegistry(type):
    def __new__(cls, clsname, superclasses, attributedict):
        attributedict["backends"] = dict()
        return super().__new__(cls, clsname, superclasses, attributedict)

    def register(cls, backend):
        if not issubclass(backend, BackendBase):
            raise TypeError("You may only register subclasses of BackendBase.")

        if backend.__name__ in cls.backends.keys():
            warnings.warn(
                f"Overwriting previously registered backend `{backend.__name__}` of the same name.",
                UserWarning,
                stacklevel=2,
            )

        cls.backends[backend.__name__] = backend


class BackendRegistry(metaclass=MetaBackendRegistry):
    pass


class BackendOptions(BaseModel): ...


class BackendBase(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        args = inspect.getfullargspec(cls.run)

        for arg in args.args[1:] + args.kwonlyargs:
            if arg not in args.annotations:
                warnings.warn(
                    f"Missing type hint for argument `{arg}` in run method of {cls.__name__}. Defaults to Any."
                )

            cls.run.__annotations__[arg] = Any

        BackendRegistry.register(cls)

    @abstractmethod
    def run(self, program, *args, **kwargs):
        pass
