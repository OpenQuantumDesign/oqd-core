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


import numpy as np
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from oqd_compiler_infrastructure import (
    ConversionRule,
)

########################################################################################

__all__ = [
    "IonVisualization",
]

########################################################################################


class IonVisualization(ConversionRule):
    def __init__(
        self,
        label_generator=None,
        scale_cutoff=10,
        relative_scale_jump=1.5,
        orbital_displacement=3,
        transition_whitelist=None,
        transition_blacklist=None,
    ):
        super().__init__()

        self.label_generator = label_generator
        self.scale_cutoff = scale_cutoff
        self.relative_scale_jump = relative_scale_jump
        self.orbital_displacement = orbital_displacement
        self.transition_whitelist = transition_whitelist
        self.transition_blacklist = transition_blacklist

        self.fig, self.ax = plt.subplots(1, 1)
        plt.close(self.fig)
        self.ax.set_axis_off()

    @staticmethod
    def term_label_generator(level):
        N = str(int(level.principal))
        S = str(int(2 * level.spin + 1))
        L = ["S", "P", "D"][int(level.orbital)]
        J = (
            str(int(level.spin_orbital))
            if level.spin_orbital % 1 == 0
            else f"{int(level.spin_orbital * 2)} / 2"
        )
        F = (
            str(int(level.spin_orbital_nuclear))
            if level.spin_orbital_nuclear % 1 == 0
            else f"{int(level.spin_orbital_nuclear * 2)} / 2"
        )
        m_F = (
            str(int(level.spin_orbital_nuclear_magnetization))
            if level.spin_orbital_nuclear_magnetization % 1 == 0
            else f"{int(level.spin_orbital_nuclear_magnetization * 2)} / 2"
        )

        return rf"$| {N} ^{{{S}}}\mathrm{{{L}}}_{{{J}}}, F={F}, m_F={m_F} \rangle$"

    def _get_level_position(self, levels):
        Es = np.array(list(map(lambda x: x.energy, levels)))
        deltaE = Es[1:] - Es[:-1]

        x = deltaE
        energy_scales = [x]
        while not np.isnan(x).all():
            x_mask = (x > np.nanmin(x) * self.scale_cutoff).astype(float)
            x_mask[np.logical_not(x_mask.astype(bool))] = np.nan

            energy_scales.append(x * x_mask)

            x = x * x_mask

        scale_displacement = []
        for i in range(len(energy_scales) - 1):
            scale_displacement.append(
                energy_scales[i]
                * np.isnan(energy_scales[i + 1])
                / energy_scales[i][
                    np.isnan(energy_scales[i + 1])
                    & np.logical_not(np.isnan(energy_scales[i]))
                ].min()
            )

        scale_displacement = (
            np.nan_to_num(np.stack(scale_displacement), nan=0)
            * np.cumprod(
                np.concat(
                    [
                        np.ones(1),
                        (
                            np.nanmax(scale_displacement, axis=1)
                            * self.relative_scale_jump
                        )[:-1],
                    ]
                )
            )[:, None]
        )

        position = scale_displacement.sum(0).cumsum()
        position = np.concatenate([np.zeros(1), position])

        return position

    def map_Ion(self, model, operands):
        energies = np.array(list(map(lambda level: level.energy, model.levels)))

        order = np.argsort(energies)
        energies = energies[order]
        levels = np.array(model.levels)[order]
        labels = np.array(operands["levels"])[order]

        pos = self._get_level_position(levels)

        for n in range(len(levels)):
            self.ax.plot(
                np.arange(2)
                + 1.5 * levels[n].spin_orbital_nuclear_magnetization
                + 1.5 * self.orbital_displacement * levels[n].orbital,
                pos[n] * np.ones(2),
                color="black",
                zorder=3,
            )

            self.ax.text(
                0.5
                + 1.5 * levels[n].spin_orbital_nuclear_magnetization
                + 1.5 * self.orbital_displacement * levels[n].orbital,
                pos[n],
                labels[n],
                ha="center",
                va="top",
                zorder=2,
            )

        level_labels = np.array(list(map(lambda level: level.label, levels)))
        for t in model.transitions:
            n1 = (
                np.where(level_labels == t.level1)[0]
                if isinstance(t.level1, str)
                else np.where(levels == t.level1)[0]
            ).item()
            n2 = (
                np.where(level_labels == t.level2)[0]
                if isinstance(t.level1, str)
                else np.where(levels == t.level1)[0]
            ).item()

            if (
                self.transition_whitelist
                and t.label not in self.transition_whitelist
                and not (
                    level_labels[n1] in self.transition_whitelist
                    or level_labels[n2] in self.transition_whitelist
                )
            ):
                continue

            if self.transition_blacklist and (
                t.label in self.transition_blacklist
                or (
                    level_labels[n1] in self.transition_blacklist
                    or level_labels[n2] in self.transition_blacklist
                )
            ):
                continue

            cmap = cm.get_cmap("gist_rainbow")
            norm = colors.Normalize(400e12 * 2 * np.pi, 790e12 * 2 * np.pi)

            self.ax.plot(
                [
                    0.5
                    + 1.5 * levels[n1].spin_orbital_nuclear_magnetization
                    + 1.5 * self.orbital_displacement * levels[n1].orbital,
                    0.5
                    + 1.5 * levels[n2].spin_orbital_nuclear_magnetization
                    + 1.5 * self.orbital_displacement * levels[n2].orbital,
                ],
                [
                    pos[n1],
                    pos[n2],
                ],
                ls=":",
                color=np.array(cmap(norm(np.abs(energies[n1] - energies[n2])))) * 0.75,
                alpha=0.5,
                zorder=1,
            )

        return self.fig, self.ax

    def map_Level(self, model, operands):
        if self.label_generator:
            return self.label_generator(model)
        else:
            return rf"$| \mathrm{{{model.label}}} \rangle$"

    # def _levelgroups(self, leveltree):
    #     idcs = np.lexsort(np.flip(leveltree.transpose(), 0))

    #     leveltree = leveltree[idcs]

    #     groups = [np.zeros(leveltree.shape[-1]).astype(int)]
    #     previous = leveltree[0]
    #     for current in leveltree[1:]:
    #         groups.append(
    #             groups[-1]
    #             + np.logical_or.accumulate(np.logical_not(current == previous))
    #         )
    #         previous = current

    #     groups = np.stack(groups)
    #     return groups[np.argsort(idcs)]

    # leveltree = np.stack(sorted(operands["levels"], key=lambda x: x[-1]))
    # levelgroups = self._levelgroups(leveltree)

    # orbitalgroups = levelgroups[:, 2]
    # for i in range(np.max(orbitalgroups.astype(int)) + 1):
    #     self.ax.text(-2, np.mean(pos[orbitalgroups == i]), ["S", "P", "D"][i])

    # orbitalgroups = levelgroups[:, 2]
    # for i in range(np.max(orbitalgroups.astype(int)) + 1):
    #     self.ax.text(-2, np.mean(pos[orbitalgroups == i]), ["S", "P", "D"][i])
