"""
Figure 4 -- IC443 Paper 1.

CLOUDY 21 cm brightness temperature for the 20-point grid
(T_obs = 1e4 K, constant-temperature slabs with density
n_H = P2/(k_B T_obs) and fixed path length L = 0.6 pc).
Each point is the direct CLOUDY output for its own shock velocity.
The vertical line marks v_shock = 31.1 km/s, whose adiabatic
post-shock velocity places it at v_LSR = -25 km/s.
"""
import numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

v_shock = np.array([20, 24, 28, 32, 35, 39, 43, 47, 50, 54, 58, 62, 65, 69,
                    73, 77, 80, 82.5, 85, 87.9])
TB21 = np.array([2.32, 3.43, 4.73, 6.09, 7.32, 9.22, 11.1, 13.3, 15.3, 17.5,
                 20.6, 23.1, 25.9, 29.1, 32.6, 35.8, 39.2, 41.0, 44.0, 47.1])
V_BOUNDARY = 31.1

cmap = plt.cm.viridis
norm = plt.Normalize(v_shock.min(), v_shock.max())

fig, ax = plt.subplots(figsize=(8, 5.8))
ax.plot(v_shock, TB21, '-', color='gray', lw=1, zorder=1)
ax.scatter(v_shock, TB21, c=cmap(norm(v_shock)), s=90, edgecolor='black',
           linewidth=0.8, zorder=5)
ax.axvline(V_BOUNDARY, color='black', lw=2.2,
           label=r'$v_{\rm shock}=31.1$ km s$^{-1}$ ($v_{\rm LSR}=-25$ km s$^{-1}$)')
ax.set_xlim(15, 92)
ax.set_xlabel(r'$v_{\rm shock}$ (km s$^{-1}$)', fontsize=11)
ax.set_ylabel(r'$T_{B,21}$ (K)', fontsize=11)
ax.legend(fontsize=9.5, loc='lower right')
ax.grid(alpha=0.25)
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, pad=0.02)
cbar.set_label(r'$v_{\rm shock}$ (km s$^{-1}$)', fontsize=9)
plt.tight_layout()
plt.savefig(REPO / 'figures' / 'CLOUDY_grid_brightness.png', dpi=150)
print("Saved CLOUDY_grid_brightness.png")
