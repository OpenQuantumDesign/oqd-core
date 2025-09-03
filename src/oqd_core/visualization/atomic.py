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


import matplotlib
import numpy as np
from matplotlib import cm, colors
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
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
        level_labelgen=None,
        transition_labelgen=None,
        transition_labelgen_whitelist=None,
        scale_cutoff=10,
        relative_scale_jump=1.1,
        transition_whitelist=None,
        transition_blacklist=None,
    ):
        super().__init__()

        self.level_labelgen = level_labelgen
        self.transition_labelgen = transition_labelgen
        self.transition_labelgen_whitelist = transition_labelgen_whitelist
        self.scale_cutoff = scale_cutoff
        self.relative_scale_jump = relative_scale_jump
        self.transition_whitelist = transition_whitelist
        self.transition_blacklist = transition_blacklist

        self.fig, self.ax = plt.subplots(1, 1)
        plt.close(self.fig)
        self.ax.set_axis_off()

    @staticmethod
    def term_level_labelgen(level):
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

        energy_scales = np.stack(energy_scales)
        scale_label = np.logical_not(np.isnan(energy_scales)).sum(0) - 1

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

        return position, scale_label

    def levelgroups(self, levels):
        groups = np.stack(
            list(
                map(
                    lambda x: np.array(
                        [
                            x.principal,
                            x.spin,
                            x.orbital,
                            x.nuclear,
                            x.spin_orbital,
                            x.spin_orbital_nuclear,
                            x.spin_orbital_nuclear_magnetization,
                        ]
                    ),
                    levels,
                )
            )
        )

        return groups

    def map_Ion(self, model, operands):
        energies = np.array(list(map(lambda level: level.energy, model.levels)))

        order = np.argsort(energies)
        energies = energies[order]
        levels = np.array(model.levels)[order]
        level_labels = np.array(list(map(lambda level: level.label, levels)))
        plot_level_labels = np.array(operands["levels"])[order]
        levelgroups = self.levelgroups(levels)[order]

        orbital_xshifts = np.cumsum(
            [
                levelgroups[:, 5][levelgroups[:, 2] == i].max()
                if i == 0
                else levelgroups[:, 5][levelgroups[:, 2] == i - 1].max()
                + levelgroups[:, 5][levelgroups[:, 2] == i].max()
                + 1
                for i in np.arange(levelgroups[:, 2].max() + 1)
            ],
        )

        pos, scale_label = self._get_level_position(levels)

        scale_cmap = cm.get_cmap("viridis")
        scale_c = scale_cmap(np.linspace(0, 1, scale_label.max() + 1))[scale_label]
        scale_pos = np.stack([-1.5 * np.ones_like(pos), pos], -1)
        scale_segments = np.concatenate(
            [scale_pos[:-1][:, None, :], scale_pos[1:][:, None, :]], axis=1
        )
        scale_lc = LineCollection(
            scale_segments, colors=scale_c, linewidths=3, alpha=0.75
        )
        self.ax.add_collection(scale_lc)

        self.ax.annotate(
            "Relative Scale",
            (-1.5, (pos.min() + pos.max()) / 2),
            (-3, 0),
            textcoords="offset points",
            rotation=90,
            ha="right",
            va="center",
            zorder=2,
            fontsize=matplotlib.rcParams["font.size"] * 1.5,
        )

        for n in range(len(levels)):
            self.ax.plot(
                np.arange(2)
                + 1.2
                * (
                    levels[n].spin_orbital_nuclear_magnetization
                    + orbital_xshifts[int(levels[n].orbital)]
                ),
                pos[n] * np.ones(2),
                color="black",
                zorder=3,
            )

            self.ax.annotate(
                plot_level_labels[n],
                (
                    0.5
                    + 1.2
                    * (
                        levels[n].spin_orbital_nuclear_magnetization
                        + orbital_xshifts[int(levels[n].orbital)]
                    ),
                    pos[n],
                ),
                (0, -3),
                textcoords="offset points",
                ha="center",
                va="top",
                zorder=2,
            )

        plot_transition_labels = np.array(operands["transitions"])
        included_transitions = map(
            lambda t: (
                t[0],
                (
                    np.where(level_labels == t[0].level1)[0]
                    if isinstance(t[0].level1, str)
                    else np.where(levels == t[0].level1)[0]
                ).item(),
                (
                    np.where(level_labels == t[0].level2)[0]
                    if isinstance(t[0].level2, str)
                    else np.where(levels == t[0].level2)[0]
                ).item(),
                t[1],
            ),
            zip(model.transitions, plot_transition_labels),
        )

        if self.transition_whitelist:
            included_transitions = filter(
                lambda x: x[0].label in self.transition_whitelist
                or level_labels[x[1]] in self.transition_whitelist
                or level_labels[x[2]] in self.transition_whitelist,
                included_transitions,
            )

        if self.transition_blacklist:
            included_transitions = filter(
                lambda x: x[0].label not in self.transition_blacklist
                and level_labels[x[1]] not in self.transition_blacklist
                and level_labels[x[2]] not in self.transition_blacklist,
                included_transitions,
            )

        for t, n1, n2, tl in included_transitions:
            ccmap = cm.get_cmap("gist_rainbow")
            cnorm = colors.Normalize(400e12 * 2 * np.pi, 790e12 * 2 * np.pi)

            on_off_seq = [
                *([1, 1] * int(t.multipole[0] == "M") * int(t.multipole[1:])),
                *([3, 1] * int(t.multipole[0] == "E") * int(t.multipole[1:])),
            ]
            on_off_seq[-1] = 5

            self.ax.plot(
                [
                    0.5
                    + 1.2
                    * (
                        levels[n1].spin_orbital_nuclear_magnetization
                        + orbital_xshifts[int(levels[n1].orbital)]
                    ),
                    0.5
                    + 1.2
                    * (
                        levels[n2].spin_orbital_nuclear_magnetization
                        + orbital_xshifts[int(levels[n2].orbital)]
                    ),
                ],
                [
                    pos[n1],
                    pos[n2],
                ],
                lw=2,
                ls=(0, on_off_seq),
                color=np.array(ccmap(cnorm(np.abs(energies[n1] - energies[n2]))))
                * 0.75,
                zorder=1,
            )

            self.ax.annotate(
                tl,
                (
                    0.5
                    + 0.6
                    * (
                        levels[n1].spin_orbital_nuclear_magnetization
                        + orbital_xshifts[int(levels[n1].orbital)]
                        + levels[n2].spin_orbital_nuclear_magnetization
                        + orbital_xshifts[int(levels[n2].orbital)]
                    ),
                    0.5 * (pos[n1] + pos[n2]),
                ),
                ha="center",
                va="top",
                zorder=2,
            )

        return self.fig, self.ax

    def map_Level(self, model, operands):
        if self.level_labelgen:
            return self.level_labelgen(model)
        else:
            return rf"$| \mathrm{{{model.label}}} \rangle$"

    def map_Transition(self, model, operands):
        if self.transition_labelgen and (
            not self.transition_labelgen_whitelist
            or (
                model.label in self.transition_labelgen_whitelist
                or model.level1.label in self.transition_labelgen_whitelist
                or model.level2.label in self.transition_labelgen_whitelist
            )
        ):
            return self.transition_labelgen(model)

        return ""
