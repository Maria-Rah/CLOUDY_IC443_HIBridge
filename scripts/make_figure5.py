"""
Figure 5 -- IC443 Paper 1: Sedov-Taylor wing model.

The filled bands are the contributions of the 20 PLUTO+CLOUDY grid points,
stacked from the fastest (bottom) to the slowest (top) shock; the top edge of
the stack is the model itself (Equation 4), i.e. the sum of all components.

Model (Equation 4 of the paper):
  T_model(v) = K * sum_i (TB21_i / N_H,i) * v_shock,i^(-p) * dv_i
                   * exp(-(v - v_i)^2 / (2 sigma^2))
  v_i   = v_sys - v_post(v_shock,i)            (adiabatic R-H, validated vs PLUTO)
  N_H,i = column of CLOUDY run i (its own v_shock: n_H = P2/(k_B T_obs), L = 0.6 pc)
  p = 3: energy-conserving blast wave, M ~ v^-2  ->  dM/dv ~ v^-3

Free parameters: K and sigma, fitted to the smoothed spectrum in
v_LSR = -70 ... -25 km/s. The script also prints the p = 8/3 and
free-p fits for comparison.

Input: observed_spectrum.dat  (columns: v_LSR, T_B_raw, T_B_smoothed)
"""
import numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ---- CLOUDY grid (direct output, T_obs = 1e4 K) ----
v_shock = np.array([20, 24, 28, 32, 35, 39, 43, 47, 50, 54, 58, 62, 65, 69,
                    73, 77, 80, 82.5, 85, 87.9])
TB21 = np.array([2.32, 3.43, 4.73, 6.09, 7.32, 9.22, 11.1, 13.3, 15.3, 17.5,
                 20.6, 23.1, 25.9, 29.1, 32.6, 35.8, 39.2, 41.0, 44.0, 47.1])

# ---- physical constants and adopted inputs ----
GAMMA, KB, MH = 5.0 / 3.0, 1.380649e-16, 1.673e-24
N_PRE, T_PRE, T_OBS, V_SYS = 15.0, 7000.0, 1.0e4, -4.0
# column density of each CLOUDY slab, exactly as in the input files
# ("stop column density" = log10(n_H * L), L = 0.6 pc, n_H = P2/(k_B T_obs) for that run's own v_shock)
LOG_N_INPUT = np.array([19.98, 20.15, 20.29, 20.40, 20.48, 20.58, 20.66, 20.74, 20.80, 20.86,
                        20.93, 20.98, 21.03, 21.08, 21.13, 21.17, 21.21, 21.23, 21.26, 21.29])
FIT_RANGE = (-70.0, -25.0)

def jump(vs_kms):
    """Adiabatic Rankine-Hugoniot jump: returns (rho2/rho1, v_post [km/s], P2 [cgs])."""
    vs = vs_kms * 1e5
    rho1, p1 = N_PRE * MH, N_PRE * KB * T_PRE
    mach = vs / np.sqrt(GAMMA * p1 / rho1)
    r = (GAMMA + 1) * mach**2 / ((GAMMA - 1) * mach**2 + 2)
    p2 = p1 * (2 * GAMMA * mach**2 - (GAMMA - 1)) / (GAMMA + 1)
    return r, vs_kms * (1 - 1 / r), p2

v_post = np.array([jump(v)[1] for v in v_shock])
N_H = 10.0**LOG_N_INPUT
emis = TB21 / N_H                     # emissivity per unit column
dv = np.gradient(v_shock)             # grid spacing (quadrature weight)
v_comp = V_SYS - v_post               # LSR velocity of each component

# ---- observed spectrum ----
data = np.genfromtxt(REPO / 'data' / 'observed_spectrum.dat', comments='#')
order = np.argsort(data[:, 0])
v_obs, T_obs = data[order, 0], data[order, 2]
sel = (v_obs >= FIT_RANGE[0]) & (v_obs <= FIT_RANGE[1])

def components(v, p, sigma, logK):
    amp = np.exp(logK) * emis * v_shock**(-p) * dv
    return amp[None, :] * np.exp(-0.5 * ((v[:, None] - v_comp[None, :]) / sigma)**2)

def fit(p=None):
    best = None
    for s0 in (10.0, 20.0, 30.0):
        for p0 in ([p] if p is not None else [2.0, 3.0, 5.0, 7.0]):
            lk0 = np.log(15.0 / (emis * v_shock**(-p0) * dv).max())
            if p is None:
                f = lambda q: components(v_obs[sel], q[0], q[1], q[2]).sum(1) - T_obs[sel]
                x0, lb, ub = [p0, s0, lk0], [0, 1, -300], [20, 80, 300]
            else:
                f = lambda q: components(v_obs[sel], p, q[0], q[1]).sum(1) - T_obs[sel]
                x0, lb, ub = [s0, lk0], [1, -300], [80, 300]
            r = least_squares(f, x0, bounds=(lb, ub))
            rms = np.sqrt(np.mean(r.fun**2))
            if best is None or rms < best[1]:
                best = (r.x, rms)
    return best

(sigma3, logK3), rms3 = fit(3.0)
(sig83, _), rms83 = fit(8.0 / 3.0)
(pfree, sigfree, _), rmsfree = fit(None)
print(f"fit channels: {sel.sum()}")
print(f"p = 3 (fixed)  : sigma = {sigma3:.2f} km/s, rms = {rms3:.3f} K")
print(f"p = 8/3 (fixed): sigma = {sig83:.2f} km/s, rms = {rms83:.3f} K")
print(f"p free         : p = {pfree:.2f}, sigma = {sigfree:.2f} km/s, rms = {rmsfree:.3f} K")

# ---- figure (p = 3 model) ----
cmap = plt.cm.viridis
norm = plt.Normalize(v_shock.min(), v_shock.max())
vfine = np.linspace(-90, 30, 650)
comp = components(vfine, 3.0, sigma3, logK3)

fig, ax = plt.subplots(figsize=(9.5, 6.4))
# stacked contributions of the 20 simulated (PLUTO+CLOUDY) grid points:
# fastest shocks at the bottom, slowest on top; the top edge of the stack
# is the model itself (sum of all simulated components)
order_v = np.argsort(v_shock)[::-1]
cum = np.zeros_like(vfine)
for j in order_v:
    ax.fill_between(vfine, cum, cum + comp[:, j], color=cmap(norm(v_shock[j])),
                    alpha=0.85, lw=0, zorder=2)
    cum = cum + comp[:, j]
ax.plot(v_obs, T_obs, color='black', lw=2, label='observed spectrum', zorder=5)
ax.plot(vfine, cum, color='#e91e8c', lw=2.5, ls='--',
        label=rf'Sedov-Taylor model: sum of the 20 simulated components ($p=3$, rms $={rms3:.2f}$ K)', zorder=6)
ax.axvline(-25, color='black', lw=2, ls=':', zorder=7)
ax.axvline(-70, color='0.4', lw=1, ls=':', zorder=7)
ax.set_xlim(-90, 30)
ax.set_xlabel(r'$v_{\rm LSR}$ (km s$^{-1}$)', fontsize=11)
ax.set_ylabel(r'$T_B$ (K)', fontsize=11)
ax.legend(fontsize=9, loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=2, frameon=True)
ax.grid(alpha=0.2)
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, pad=0.02)
cbar.set_label(r'$v_{\rm shock}$ (km s$^{-1}$)', fontsize=9)
plt.tight_layout()
plt.savefig(REPO / 'figures' / 'SedovTaylor_wing_model.png', dpi=150)
print("Saved SedovTaylor_wing_model.png")
