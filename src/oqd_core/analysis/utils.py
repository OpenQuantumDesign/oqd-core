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

########################################################################################

from __future__ import annotations

from types import UnionType
from typing import Annotated, Union, get_args, get_origin

########################################################################################


def alias_types(alias: object) -> tuple[type, ...]:
    """Flatten `Annotated`/`Union` aliases into a tuple of concrete Python types."""
    origin = get_origin(alias)
    if origin is Annotated:
        return alias_types(get_args(alias)[0])

    if origin in (Union, UnionType):
        out: list[type] = []
        for arg in get_args(alias):
            out.extend(alias_types(arg))
        return tuple(dict.fromkeys(out))

    if isinstance(alias, type):
        return (alias,)
    return ()
