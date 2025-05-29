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

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from typing import Dict, List

# Actual imports from oqd_core
from oqd_core.interface.atomic import Ion, Level, Transition


def get_orbital_label(l_value: float) -> str:
    """Converts orbital quantum number l to its spectroscopic letter."""
    # The Level.orbital is NonNegativeAngularMomentumNumber (float)
    # For labels, we typically use integer part if it's an integer
    l_int = int(l_value)
    if l_value == l_int:
        labels = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g', 5: 'h'}
        return labels.get(l_int, f'l={l_int}')
    return f'l={l_value}' # For half-integer or other float cases if they occur


def generate_energy_level_plot(ion: Ion, **plot_kwargs) -> Figure:
    """
    Generates an energy level diagram (Grotrian-like diagram) for a given Ion instance.

    Args:
        ion: An instance of the Ion class containing energy levels and transitions.
        **plot_kwargs: Additional keyword arguments to customize the plot.
            figure (dict): Arguments for plt.subplots (e.g., {'figsize': (10, 8)})
            level_color (str): Color for level lines.
            level_linewidth (float): Linewidth for level lines.
            level_fontsize (float): Fontsize for level labels.
            level_line_width_half (float): Half-width of the horizontal level lines.
            text_offset_x (float): Horizontal offset for level labels from line end.
            text_offset_y_factor (float): Factor to determine vertical offset for level labels (multiplied by energy range).
            show_n_quantum_number (bool): Whether to show principal quantum number in level labels.
            arrow_style (str): Matplotlib arrow style for transitions.
            transition_color (str): Color for transition arrows.
            arrow_shrink_a (float): Shrink factor for arrow tip.
            arrow_shrink_b (float): Shrink factor for arrow base.
            transition_linewidth (float): Linewidth for transition arrows.
            show_transition_labels (bool): Whether to show labels for transitions.
            transition_label_color (str): Color for transition labels.
            transition_label_fontsize (float): Fontsize for transition labels.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            title (str): Title for the plot.

    Returns:
        A matplotlib.figure.Figure object representing the plot.
    """
    fig, ax = plt.subplots(**plot_kwargs.get('figure', {}))

    ion_name = getattr(ion, 'name', 'Ion') # Check if Ion has a name/label, else default

    if not ion.levels:
        ax.text(0.5, 0.5, "No energy levels defined for this ion.",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        ax.set_title(plot_kwargs.get('title', f"Energy Level Diagram for {ion_name}"))
        return fig

    # Create a mapping from level labels to Level objects for quick lookup if transitions use strings
    levels_map_by_label: Dict[str, Level] = {level.label: level for level in ion.levels}

    # Group levels by orbital quantum number (l)
    levels_by_orbital: dict[float, List[Level]] = {}
    for level in ion.levels:
        orbital_val = float(level.orbital) # Ensure it's float for dict key
        if orbital_val not in levels_by_orbital:
            levels_by_orbital[orbital_val] = []
        levels_by_orbital[orbital_val].append(level)

    sorted_orbitals = sorted(levels_by_orbital.keys())
    orbital_x_coords = {l_val: i for i, l_val in enumerate(sorted_orbitals)}
    
    level_line_width_half = plot_kwargs.get("level_line_width_half", 0.35)
    text_offset_x = plot_kwargs.get("text_offset_x", 0.05)
    
    level_coords: Dict[str, tuple[float, float]] = {} 
    min_energy = float('inf')
    max_energy = float('-inf')

    for level in ion.levels: # Iterate once to find min/max energy for dynamic text_offset_y
        min_energy = min(min_energy, level.energy)
        max_energy = max(max_energy, level.energy)
    
    energy_span = max_energy - min_energy if max_energy > min_energy else 1.0
    text_offset_y = plot_kwargs.get("text_offset_y_factor", 0.01) * energy_span


    for l_val, levels_in_group in levels_by_orbital.items():
        x_center = orbital_x_coords[l_val]
        levels_in_group.sort(key=lambda lvl: lvl.energy)
        
        for level in levels_in_group:
            y = level.energy
            
            ax.hlines(y, x_center - level_line_width_half, x_center + level_line_width_half,
                      colors=plot_kwargs.get('level_color', 'black'),
                      linewidth=plot_kwargs.get('level_linewidth', 1.5))
            
            level_coords[level.label] = (x_center, y)
            
            label_text = f"{level.label}"
            if plot_kwargs.get("show_n_quantum_number", True):
                 label_text = f"n={level.principal}, {label_text}"

            ax.text(x_center + level_line_width_half + text_offset_x, y + text_offset_y, label_text,
                    verticalalignment='bottom',
                    fontsize=plot_kwargs.get('level_fontsize', 8))

    for transition in ion.transitions:
        # Resolve level1 and level2 if they are strings
        level1_obj = transition.level1 if isinstance(transition.level1, Level) else levels_map_by_label.get(str(transition.level1))
        level2_obj = transition.level2 if isinstance(transition.level2, Level) else levels_map_by_label.get(str(transition.level2))

        if level1_obj and level2_obj and level1_obj.label in level_coords and level2_obj.label in level_coords:
            x1, y1 = level_coords[level1_obj.label]
            x2, y2 = level_coords[level2_obj.label]
            
            ax.annotate("",
                        xy=(x2, y2), xycoords='data',
                        xytext=(x1, y1), textcoords='data',
                        arrowprops=dict(arrowstyle=plot_kwargs.get('arrow_style', "->"),
                                        connectionstyle=plot_kwargs.get('connectionstyle', "arc3,rad=0.1"),
                                        color=plot_kwargs.get('transition_color', 'gray'),
                                        shrinkA=plot_kwargs.get('arrow_shrink_a', 5),
                                        shrinkB=plot_kwargs.get('arrow_shrink_b', 5),
                                        linewidth=plot_kwargs.get('transition_linewidth', 1)),
                        )
            if plot_kwargs.get("show_transition_labels", False) and transition.label:
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                # Add a small offset if x1 and x2 are the same to avoid label overlap with vertical line
                label_offset_x = 0.1 if x1 == x2 else 0
                ax.text(mid_x + label_offset_x, mid_y, transition.label,
                        color=plot_kwargs.get('transition_label_color', 'blue'),
                        fontsize=plot_kwargs.get('transition_label_fontsize', 7),
                        ha='center', va='center',
                        bbox=plot_kwargs.get('transition_label_bbox', dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.1)))

    ax.set_xlabel(plot_kwargs.get('xlabel', "Orbital Angular Momentum (l)"))
    ax.set_ylabel(plot_kwargs.get('ylabel', "Energy"))
    ax.set_title(plot_kwargs.get('title', f"Energy Level Diagram for {ion_name}"))
    
    if sorted_orbitals:
        ax.set_xticks(list(orbital_x_coords.values()))
        ax.set_xticklabels([get_orbital_label(l) for l in sorted_orbitals])
    else:
        ax.set_xticks([])

    if min_energy != float('inf') and max_energy != float('-inf'):
        padding = 0.1 * energy_span if energy_span > 0 else 1.0
        ax.set_ylim(min_energy - padding, max_energy + padding)

    plt.tight_layout()
    return fig