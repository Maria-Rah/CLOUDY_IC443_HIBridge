import numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
Figure 7: CLOUDY 21 cm brightness temperature at fixed path length (L = 0.6 pc) as a
function of the assumed temperature of the cooled gas, T_obs = 8000, 10000, 12000 K,
for v_shock = 65, 80, 100 km/s (nine CLOUDY models in cloudy_inputs/Tobs_sensitivity).
"""

T_obs_vals = [8000, 10000, 12000]
grid = {
    65:  [35.7, 25.9, 18.9],
    80:  [53.9, 39.2, 28.6],
    100: [85.3, 60.6, 44.2],
}
colors = {65: '#2a78d6', 80: '#e8834a', 100: '#c0392b'}

fig, ax = plt.subplots(figsize=(7, 5.5))
for vsh, TBs in grid.items():
    ax.plot(T_obs_vals, TBs, 'o-', color=colors[vsh], lw=2, ms=8,
            label=f'v$_{{shock}}$={vsh} km/s')

ax.axvline(10000, color='gray', ls='--', lw=1.3, label=r'adopted $T_{\rm obs}$')
ax.set_xlabel('T$_{obs}$ (K)', fontsize=11)
ax.set_ylabel('T$_B$21cm (K)', fontsize=11)
ax.legend(fontsize=9.5, loc='upper right')
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(REPO / 'figures' / 'Tobs_sensitivity.png', dpi=150)
print("Saved Tobs_sensitivity.png")
