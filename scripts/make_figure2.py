import numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
Figure 2: HI 21 cm spectrum at the peak-excess position (l = 189.29 deg, b = 3.02 deg)
with four descriptive regions and the measured wing edge (v_LSR = -69.1 km/s;
3 sigma threshold, ten consecutive channels). Input: data/observed_spectrum.dat.
"""

data = np.genfromtxt(REPO / 'data' / 'observed_spectrum.dat', skip_header=1)
v, T = data[:, 0], data[:, 2]

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(v, T, color='black', lw=1.5, zorder=6)

# four structural regions (background shading only, boundaries from the
# real data's own shape -- see Section 2.3 of the paper for how these
# were identified)
regions = [
    (-125, -70, '#cccccc', '1) baseline / noise'),
    (-70, -6,   '#2a78d6', '2) rising wing (shock)'),
    (-6, 13,    '#c0392b', '3) three-peak complex'),
    (13, 100,   '#e8a83c', '4) decline'),
]
for v0, v1, color, label in regions:
    ax.axvspan(v0, v1, color=color, alpha=0.18, label=label)

# precise kinematic wing edge (measured: 3-sigma threshold, 10 consecutive
# channels, scanning from the main line toward negative velocity) --
# distinct from the approximate region-1/region-2 shading boundary above
ax.axvline(-69.1, color='black', ls='--', lw=1.8, zorder=7,
           label='wing edge = -69.1 km/s')

ax.set_xlabel('v$_{LSR}$ (km/s)', fontsize=11)
ax.set_ylabel('T$_B$ (K)', fontsize=11)
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.2)
ax.set_xlim(-125, 100)

plt.tight_layout()
plt.savefig(REPO / 'figures' / 'HI_spectrum_wing_edge.png', dpi=150)
print("Saved HI_spectrum_wing_edge.png")
