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
import math
import sys

# Adjust path to import from oqd_core if running from project root
# sys.path.insert(0, '.') # Uncomment if oqd_core is not in PYTHONPATH

from oqd_core.interface.atomic import Ion, Level, Transition
from oqd_core.visualizations import generate_energy_level_plot

if __name__ == '__main__':
    pi = math.pi

    # --- Test Case 1: Hypothetical Ion with s, p, d states ---
    print("Generating plot for Hypothetical Ion...")
    s1 = Level(label="1S", energy=0, orbital=0, principal=1, spin=0.5, nuclear=0.5, spin_orbital=0.5, spin_orbital_nuclear=0, spin_orbital_nuclear_magnetization=0)
    p1_lower = Level(label="2P(1/2)", energy=10, orbital=1, principal=2, spin=0.5, nuclear=0.5, spin_orbital=0.5, spin_orbital_nuclear=0, spin_orbital_nuclear_magnetization=0)
    p1_upper = Level(label="2P(3/2)", energy=10.1, orbital=1, principal=2, spin=0.5, nuclear=0.5, spin_orbital=0.5, spin_orbital_nuclear=0, spin_orbital_nuclear_magnetization=0)
    s2 = Level(label="2S", energy=12, orbital=0, principal=2, spin=0.5, nuclear=0.5, spin_orbital=0.5, spin_orbital_nuclear=0, spin_orbital_nuclear_magnetization=0)
    d1 = Level(label="3D", energy=15, orbital=2, principal=3, spin=0.5, nuclear=0.5, spin_orbital=1.5, spin_orbital_nuclear=1, spin_orbital_nuclear_magnetization=0)
    
    levels_hypo = [s1, p1_lower, p1_upper, s2, d1]

    t1_hypo = Transition(label="T1 (S-P)", level1=s1, level2=p1_lower, einsteinA=1.0, multipole="E1")
    t2_hypo = Transition(label="T2 (P-S)", level1=p1_lower, level2=s2, einsteinA=1.0, multipole="E1")
    t3_hypo = Transition(label="T3 (P-D)", level1=p1_upper, level2=d1, einsteinA=1.0, multipole="E1")
    t4_hypo = Transition(label="T4 (S-P) rev", level1=s2, level2=p1_upper, einsteinA=1.0, multipole="E1")

    transitions_hypo = [t1_hypo, t2_hypo, t3_hypo, t4_hypo]

    hypothetical_ion = Ion(
        mass=1.0, charge=1, position=[0,0,0], # Dummy values for mass, charge, position
        levels=levels_hypo,
        transitions=transitions_hypo
        # Note: Ion class does not have a 'name' or 'label' attribute by default.
        # The plot title will use 'Ion' or what's passed via plot_kwargs.
    )

    fig_hypo = generate_energy_level_plot(
        hypothetical_ion,
        figure={'figsize': (10, 7)},
        show_transition_labels=True,
        level_fontsize=9,
        title="Energy Levels of Hypothetical Ion (Test Script)",
        connectionstyle="arc3,rad=0.2"
    )
    plt.show()
    print("Plot for Hypothetical Ion displayed.")

    # --- Test Case 2: Yb-171 Example from Documentation ---
    print("\nGenerating plot for Yb-171...")
    yb_downstate = Level(
        label="q0", principal=6, spin=1/2, orbital=0, nuclear=1/2,
        spin_orbital=1/2, spin_orbital_nuclear=0, spin_orbital_nuclear_magnetization=0,
        energy=0
    )
    yb_upstate = Level(
        label="q1", principal=6, spin=1/2, orbital=0, nuclear=1/2,
        spin_orbital=1/2, spin_orbital_nuclear=1, spin_orbital_nuclear_magnetization=0,
        energy=2*pi*12.643e9
    )
    yb_estate = Level(
        label="e0", principal=5, spin=1/2, orbital=1, nuclear=1/2, # orbital=1 for P state
        spin_orbital=1/2, spin_orbital_nuclear=0, spin_orbital_nuclear_magnetization=0,
        energy=2*pi*811.52e12 
    )

    Yb171_ion_instance = Ion(
        mass=171, charge=1, position=[0,0,0],
        levels=[yb_downstate, yb_upstate, yb_estate],
        transitions=[
            Transition(label="q0->q1", level1=yb_downstate, level2=yb_upstate, einsteinA=1e-9, multipole="M1"),
            Transition(label="q0->e0", level1=yb_downstate, level2=yb_estate, einsteinA=1e7, multipole="E1"),
            Transition(label="q1->e0", level1=yb_upstate, level2=yb_estate, einsteinA=1e7, multipole="E1")
        ]
    )
    fig_yb = generate_energy_level_plot(
        Yb171_ion_instance, 
        title="Yb-171 Energy Levels (Test Script)",
        figure={'figsize': (8,7)}, 
        show_transition_labels=True
    )
    plt.show()
    print("Plot for Yb-171 displayed.")

    # --- Test Case 3: Ion with no transitions ---
    print("\nGenerating plot for Ion with no transitions...")
    no_trans_ion = Ion(
        mass=1.0, charge=1, position=[0,0,0],
        levels=[
            Level(label="g", energy=0, orbital=0, principal=1, spin=0.5, nuclear=0.5, spin_orbital=0.5, spin_orbital_nuclear=0, spin_orbital_nuclear_magnetization=0),
            Level(label="e", energy=1, orbital=1, principal=1, spin=0.5, nuclear=0.5, spin_orbital=0.5, spin_orbital_nuclear=0, spin_orbital_nuclear_magnetization=0)
        ],
        transitions=[]
    )
    fig_no_trans = generate_energy_level_plot(no_trans_ion, title="Ion with No Transitions (Test Script)")
    plt.show()
    print("Plot for Ion with no transitions displayed.")

    # --- Test Case 4: Ion with no levels ---
    print("\nGenerating plot for Ion with no levels...")
    no_levels_ion = Ion(
        mass=1.0, charge=1, position=[0,0,0],
        levels=[],
        transitions=[]
    )
    fig_no_levels = generate_energy_level_plot(no_levels_ion, title="Ion with No Levels (Test Script)")
    plt.show()
    print("Plot for Ion with no levels displayed.")
    print("\nAll test plots generated. You can now close them.")
