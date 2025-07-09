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

from warnings import warn

import numpy as np
from oqd_compiler_infrastructure import RewriteRule

from oqd_core.compiler.math.passes import canonicalize_math_expr
from oqd_core.interface.math import MathNum

########################################################################################


class VerifyQuantumNumbers(RewriteRule):
    def map_Level(self, model):
        N = model.principal
        S = model.spin
        L = model.orbital
        I = model.nuclear  # noqa: E741
        J = model.spin_orbital
        F = model.spin_orbital_nuclear
        M = model.spin_orbital_nuclear_magnetization

        if L >= N:
            warn("Incompatible orbital and principal quantum numbers (L >= N)")

        if J < abs(S - L) or J > S + L:
            warn(
                "Incompatible spin, orbital and spin_orbital quantum numbers (not |L-S| >= J >= L+S)"
            )

        if I < abs(J - I) or I > J + I:
            warn(
                "Incompatible spin_orbital, nuclear and spin_orbital_nuclear quantum numbers (not |J-I| >= F >= J+I)"
            )

        if abs(M) > F:
            warn(
                "Incompatible magnetization and total angular momentum (not -F <= m_F <= F)"
            )

        if (M - F).is_integer():
            warn(
                "Incompatible magnetization and total angular momentum (m_F - F is not integer)"
            )


# class VerifySelectionRules(RewriteRule):
#     def map_Transition(self, model):
#         N1 = model.level1.principal
#         S1 = model.level1.spin
#         L1 = model.level1.orbital
#         I1 = model.level1.nuclear
#         J1 = model.level1.spin_orbital
#         F1 = model.level1.spin_orbital_nuclear
#         M1 = model.level1.spin_orbital_nuclear_magnetization

#         N2 = model.level2.principal
#         S2 = model.level2.spin
#         L2 = model.level2.orbital
#         I2 = model.level2.nuclear
#         J2 = model.level2.spin_orbital
#         F2 = model.level2.spin_orbital_nuclear
#         M2 = model.level2.spin_orbital_nuclear_magnetization

#         if model.multipole == "M1":
#             pass

#         elif model.multipole == "E1":
#             pass

#         elif model.multipole == "E2":
#             pass

#         else:
#             warn(f"Multipole {model.multipole} currently not supported.")


class VerifyBeam(RewriteRule):
    def map_Beam(self, model):
        k = np.array(model.wavevector)
        E = np.array(model.polarization)

        M1 = model.transition.level1.spin_orbital_nuclear_magnetization
        M2 = model.transition.level2.spin_orbital_nuclear_magnetization

        if model.transition.multipole == "M1":
            polarization_map = {
                -1: 1 / np.sqrt(2) * np.array([1, 1j, 0]),
                0: np.array([0, 0, 1]),
                1: 1 / np.sqrt(2) * np.array([1, -1j, 0]),
            }

            B = np.cross(k, E)
            q = polarization_map[int(M1 - M2)]
            g = q.dot(B)

            if canonicalize_math_expr(g) == MathNum(value=0):
                warn("Beam does not couple to reference M1 transition.")

        elif model.transition.multipole == "E1":
            polarization_map = {
                -1: 1 / np.sqrt(2) * np.array([1, 1j, 0]),
                0: np.array([0, 0, 1]),
                1: 1 / np.sqrt(2) * np.array([1, -1j, 0]),
            }

            q = polarization_map[int(M1 - M2)]
            g = q.dot(E)

            if canonicalize_math_expr(g) == MathNum(value=0):
                warn("Beam does not couple to reference E1 transition.")

        elif model.transition.multipole == "E2":
            polarization_map = {
                -2: 1 / np.sqrt(6) * np.array([[1, 1j, 0], [1j, -1, 0], [0, 0, 0]]),
                -1: 1 / np.sqrt(6) * np.array([[0, 0, 1], [0, 0, 1j], [1, 1j, 0]]),
                0: 1 / 3 * np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 2]]),
                1: 1 / np.sqrt(6) * np.array([[0, 0, -1], [0, 0, 1j], [-1, 1j, 0]]),
                2: 1 / np.sqrt(6) * np.array([[1, -1j, 0], [-1j, -1, 0], [0, 0, 0]]),
            }

            q = polarization_map[int(M1 - M2)]
            g = k.dot(np.matmul(q, E))

            if canonicalize_math_expr(g) == MathNum(value=0):
                warn("Beam does not couple to reference E2 transition.")

        else:
            warn(f"Multipole {model.multipole} currently not supported.")
