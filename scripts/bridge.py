"""
PLUTO -> CLOUDY bridge used in the paper (Section 3.2, Equation 2).

For a shock velocity v_shock and an assumed temperature T_obs of the cooled
post-shock gas, the adiabatic Rankine-Hugoniot post-shock pressure P2 sets the
hydrogen density of gas that has cooled at constant pressure,
    n_H = P2 / (k_B T_obs),
and each CLOUDY slab has the fixed path length L = 0.6 pc, so N_H = n_H L.
Pre-shock medium: n_pre = 15 cm^-3, T_pre = 7000 K, gamma = 5/3, mean particle mass m_H.

Usage:  python bridge.py            (prints hden and column for the 20-point grid)
"""
import numpy as np

GAMMA, KB, MH, PC = 5.0 / 3.0, 1.380649e-16, 1.673e-24, 3.086e18
N_PRE, T_PRE, L_PC = 15.0, 7000.0, 0.6
GRID = [20, 24, 28, 32, 35, 39, 43, 47, 50, 54, 58, 62, 65, 69, 73, 77, 80, 82.5, 85, 87.9]


def jump(v_shock_kms, t_pre=T_PRE, n_pre=N_PRE):
    """Adiabatic R-H jump: returns (rho2/rho1, v_post [km/s], P2 [dyn cm^-2])."""
    rho1, p1 = n_pre * MH, n_pre * KB * t_pre
    mach = v_shock_kms * 1e5 / np.sqrt(GAMMA * p1 / rho1)
    r = (GAMMA + 1) * mach**2 / ((GAMMA - 1) * mach**2 + 2)
    p2 = p1 * (2 * GAMMA * mach**2 - (GAMMA - 1)) / (GAMMA + 1)
    return r, v_shock_kms * (1 - 1 / r), p2


def cloudy_parameters(v_shock_kms, t_obs=1.0e4):
    """Returns (log10 n_H [cm^-3], log10 N_H [cm^-2]) as written in the CLOUDY inputs."""
    n_h = jump(v_shock_kms)[2] / (KB * t_obs)
    return np.log10(n_h), np.log10(n_h * L_PC * PC)


if __name__ == '__main__':
    print(f"{'v_shock':>8} {'v_post':>7} {'hden':>7} {'logN':>6}")
    for v in GRID:
        lh, lc = cloudy_parameters(v)
        print(f"{v:8g} {jump(v)[1]:7.1f} {lh:7.4f} {lc:6.2f}")
