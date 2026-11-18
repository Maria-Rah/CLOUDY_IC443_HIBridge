"""
Recomputes the numbers added in the revision of the paper (AAS81928): HI column of the
wing and the pre-shock density it requires (Section 5.2), kinematic limits for other
post-shock states (Sections 5.1, 5.3, 5.5), cooling time and cooling length (Section 5.5),
wing-model fits with both velocity mappings (Section 4.2, Table 2) and their sensitivity
tests (Sections 5.3, 5.8), ionization of the CLOUDY slabs (Section 3.2), and positions of
the extraction points relative to the shocked molecular clumps (Section 5.2).

Inputs: the files of this repository and the solar CIE cooling table of
Sutherland & Dopita (1993), m-00.cie.
Usage:  python revision_numbers.py path/to/m-00.cie
"""
import csv
import re
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import brentq, least_squares
from bridge import jump, cloudy_parameters, KB, MH, N_PRE, T_PRE, PC

REPO = Path(__file__).resolve().parent.parent
SD93 = Path(sys.argv[1])
V_SYS, YR, X_HI_CONST = -4.0, 3.156e7, 1.823e18
def head(s): print(f'\n==== {s}')

# ------------------------------------------------------------------------ spectrum at A
d = np.genfromtxt(REPO / 'data' / 'observed_spectrum.dat', comments='#'); d = d[np.argsort(d[:, 0])]
v, raw, smf = d[:, 0], d[:, 1], d[:, 2]; good = np.isfinite(raw); dv = np.median(np.diff(v))
noise = np.std(raw[good & (v >= 40) & (v <= 85)])

head('Section 5.2: HI column of the wing at A and the pre-shock density it requires')
s = good & (v >= -70) & (v <= -25); W = raw[s].sum() * dv; NHI = X_HI_CONST * W
print(f'W(-70,-25) = {W:.1f} +/- {noise*dv*np.sqrt(s.sum()):.1f} K km/s; N_HI = {NHI:.2e} cm^-2; mean T_B = {raw[s].mean():.1f} K')
print(f'share of W at v_LSR < -35 km/s: {100*raw[s & (v < -35)].sum()*dv/W:.0f}%')
sf = good & (v >= -100) & (v <= -85); Wf, eWf = raw[sf].sum() * dv, noise * dv * np.sqrt(sf.sum())
print(f'W(-100,-85) = {Wf:.1f} +/- {eWf:.1f} K km/s; 3 sigma: N_HI < {X_HI_CONST*(max(Wf, 0)+3*eWf):.1e} cm^-2')
NB = X_HI_CONST * 6.5 * 29.7
for dist in (1.5, 1.9):
    R = np.radians(22.5 / 60) * dist * 1e3 * PC
    cA, cB = np.sqrt(1 - 0.70**2), np.sqrt(1 - 0.39**2)
    print(f'd = {dist} kpc, R = {R/PC:.1f} pc: A shell n0 = {NHI*3*cA/R:.0f}, planar n0 = {NHI*cA/R:.0f}; '
          f'B (excess band, N = {NB:.1e}) shell n0 = {NB*3*cB/R:.0f}, planar n0 = {NB*cB/R:.0f} cm^-3')
    print('    mean T_B over the wing at A for n0 = 0.1, 1: shell ' +
          ', '.join(f'{n0*R/(3*cA)/X_HI_CONST/45.3:.2f}' for n0 in (0.1, 1)) + ' K; planar ' +
          ', '.join(f'{n0*R/cA/X_HI_CONST/45.3:.2f}' for n0 in (0.1, 1)) + ' K')

head('Sections 5.1, 5.3, 5.5: kinematic limits for other post-shock states')
def v_ad(off, tpre=T_PRE): return brentq(lambda x: jump(x, t_pre=tpre)[1] - off, 15, 900)
def v_cool(off, t=1e4, tpre=T_PRE, momentum=False):
    def resid(x):
        r, vp, p2 = jump(x, t_pre=tpre); rho1 = N_PRE * MH
        pc = p2
        if momentum:                                  # P + j u conserved through the cooling layer
            j, u2 = rho1 * x * 1e5, x * 1e5 / r
            for _ in range(50):
                pc = p2 + j * u2 - j * j / (pc * MH / (KB * t))
        return x * (1 - N_PRE * KB * t / pc) - off
    return brentq(resid, 15, 900)
for lab, off in (('A', 65.1), ('B', 78.3)):
    print(f'{lab}: adiabatic {v_ad(off):.1f}; cooled at 1e4 / 5000 / 100 K: {v_cool(off):.1f} / {v_cool(off, 5000):.1f} / {v_cool(off, 100):.1f}; '
          f'with the momentum-flux pressure: {v_cool(off, momentum=True):.1f}; T_pre = 100 K: adiabatic {v_ad(off, 100):.1f}, cooled {v_cool(off, tpre=100):.1f}')
r, vp, p2 = jump(80); j, u2 = N_PRE * MH * 80e5, 80e5 / r; pc = p2
for _ in range(50): pc = p2 + j * u2 - j * j / (pc * MH / (KB * 1e4))
print(f'pressure of the cooled gas / P2 at 80 km/s: {pc/p2:.2f}')
for vl in (-100, -105, -91.3):
    print(f'gas at {vl}: cooled {v_cool(abs(vl - V_SYS)):.0f}, adiabatic {v_ad(abs(vl - V_SYS)):.0f} km/s')
ne, cne = abs(-51.8 - V_SYS), np.sqrt(1 - 0.75**2)
print(f'NE: cooled {v_cool(ne):.0f}, deprojected (cos {cne:.2f}) {v_cool(ne/cne):.0f} km/s')

head('Section 5.5: cooling time and cooling length (Sutherland & Dopita 1993, solar, CIE)')
tab = np.loadtxt(SD93, skiprows=4); lT, xe_t, xt_t, lnet = tab[:, 0], tab[:, 1], tab[:, 3], tab[:, 4]
lam = lambda T: 10**np.interp(np.log10(T), lT, lnet)                     # cooling rate / n_H^2
ntot = lambda T: np.interp(np.log10(T), lT, xt_t) + np.interp(np.log10(T), lT, xe_t)   # particles per H
Ts = lambda vs, mu=0.6: 3 / 16 * mu * MH * (vs * 1e5)**2 / KB
def t_cool(vs, n0, t_end=1.2e4):
    T0 = Ts(vs); P = ntot(T0) * 4 * n0 * KB * T0
    Tg = np.logspace(np.log10(t_end), np.log10(T0), 400); nH = P / (ntot(Tg) * KB * Tg)
    integrate = getattr(np, 'trapezoid', None) or np.trapz                    # NumPy >= 2.0 / < 2.0
    return integrate(2.5 * ntot(Tg) * KB / (nH * lam(Tg)), Tg) / YR       # isobaric: dt = 5/2 n k dT / (n_H^2 Lambda)
for vs in (67, 80, 100, 105):
    print(f'v = {vs}: T_s = {Ts(vs):.1e} K (mu = 0.6); Lambda/(n_e n_H) = {lam(Ts(vs))/np.interp(np.log10(Ts(vs)), lT, xe_t):.1e} erg cm^3 s^-1')
for n0 in (0.1, 1, 15, 50):
    tc = [t_cool(vs, n0) for vs in (67, 80, 100, 105)]
    lc = [vs / 4 * 1e5 * t * YR / PC for vs, t in zip((67, 80, 100, 105), tc)]
    print(f'n0 = {n0:5}: t_cool = {min(tc):.3g}-{max(tc):.3g} yr; L_cool = (v/4) t_cool = {min(lc):.1e}-{max(lc):.1e} pc')
for vs in (67, 80):
    nc = jump(vs)[2] / (KB * 1e4); trec = 1 / (2.6e-13 * nc) / YR
    Nl = N_PRE * vs * 1e5 * (t_cool(vs, N_PRE) + trec) * YR
    print(f'v = {vs}: n_H(1e4 K) = {nc:.0f}; t_rec = {trec:.0f} yr; non-equilibrium column n0 v (t_cool + t_rec) = {Nl:.1e} = {100*Nl/NHI:.1f}% of N_HI(A)')
print(f'fraction of the column swept in 4 kyr that is still above 1.2e4 K (n0 = 15, 80 km/s): {100*t_cool(80, 15)/4e3:.1f}%')

head('Section 4.2, Table 2: wing-model fits')
with open(REPO / 'cloudy_inputs' / 'grid_Tobs10000' / 'tb21.csv') as fh:
    gr = sorted((float(q['v_shock_kms']), float(q['TB21_K'])) for q in csv.DictReader(fh))
vg0 = np.array([q[0] for q in gr]); emis0 = np.array([q[1] for q in gr]) / 10**np.array([cloudy_parameters(q[0])[1] for q in gr])
vpost_cool = lambda x: x * (1 - N_PRE * KB * 1e4 / jump(x)[2])
V_EDGE_COOL = round(brentq(lambda x: vpost_cool(x) - 65.1, 20, 200), 1)
print(f'emissivity per unit column over the grid: max/min = {emis0.max()/emis0.min():.3f}; cooled-gas edge: {V_EDGE_COOL} km/s')
def grid(mapping, vmin=20.0, vmax=None, c=(1.0,)):
    """Components of Eq. 5. Adiabatic: the 20 CLOUDY points (20-87.9 km/s). Cooled: points up to the cooled-gas
    edge, 66.8 km/s, whose emissivity is interpolated in the grid. Points outside the grid (sensitivity tests only)
    take the emissivity of the nearest grid point."""
    vmax = vmax or (87.9 if mapping == 'ad' else V_EDGE_COOL)
    vg = vg0[vg0 <= vmax + 1e-6]
    if vmax > vg.max() + 0.5:
        vg = np.concatenate([vg, np.arange(vg.max() + 3, vmax - 1, 3.0), [vmax]])
    if vmin < 20:
        vg = np.concatenate([np.arange(vmin, 20, 3.0), vg])
    e = np.interp(vg, vg0, emis0)
    vp = np.array([(jump(x)[1] if mapping == 'ad' else vpost_cool(x)) for x in vg])
    return vg, e * np.gradient(vg), [V_SYS - cc * vp for cc in c]
def fit(p, mapping, free=False, weight=None, win=(-70, -25), **kw):
    vg, b, vcs = grid(mapping, **kw)
    if weight is not None: b = b * weight(vg)
    sel = good & (v >= win[0]) & (v <= win[1]); vo, To = v[sel], smf[sel]
    best = None
    for s0 in (6, 10, 15, 20, 25, 35):
        for p0 in ((p,) if not free else (2.5, 4, 6)):
            def res(q):
                w = np.exp(q[1]) * b * vg**(-(q[2] if free else p))
                return sum((w[None, :] * np.exp(-0.5 * ((vo[:, None] - vc[None, :]) / q[0])**2)).sum(1) for vc in vcs) / len(vcs) - To
            x0 = [s0, np.log(15 / (b * vg**(-p0)).max())] + ([p0] if free else [])
            rr = least_squares(res, x0, bounds=([1, -300] + ([0] if free else []), [80, 300] + ([12] if free else [])))
            rms = np.sqrt(np.mean(rr.fun**2))
            if best is None or rms < best[1]: best = (rr.x, rms, rr.fun, vo, vg, b, vcs)
    return best
def show(lab, res, free=False):
    x, rms, rr, vo = res[:4]; i = np.argmax(abs(rr))
    print(f'{lab:44s}' + (f'p = {x[2]:.1f}, ' if free else '') + f'sigma = {x[0]:.1f}, rms = {rms:.2f}, largest residual {abs(rr[i]):.1f} K at {vo[i]:.0f}')
for mp, ps in (('cl', (3, 8/3, 2.2, 2, 1.5)), ('ad', (3, 8/3, 2.2, 2, 1.5, 1))):
    for p in ps: show(f'{"cooled" if mp == "cl" else "adiabatic"} p = {p:.2f}', fit(p, mp))
    show(f'{"cooled" if mp == "cl" else "adiabatic"} p free', fit(3, mp, True), True)
    print('   rms for p = 2.5, 3.5, 4, 5, 6: ' + ', '.join(f'{fit(p, mp)[1]:.2f}' for p in (2.5, 3.5, 4, 5, 6)))
x, rms, rr, vo, vg, b, vcs = fit(3, 'cl'); w = np.exp(x[1]) * b * vg**-3.0
def share(vv, cond):
    cc = w * np.exp(-0.5 * ((vv - vcs[0]) / x[0])**2); return 100 * cc[cond].sum() / cc.sum()
print(f'cooled p = 3: shares of v_shock < 40 at -45 / -25 km/s: {share(-45, vg < 40):.0f} / {share(-25, vg < 40):.0f}%; '
      f'of v_shock >= 60 at -65 km/s: {share(-65, vg >= 60):.0f}%; model at -80 km/s: {(w*np.exp(-0.5*((-80-vcs[0])/x[0])**2)).sum():.1f} K')
xa = fit(3, 'cl', vmax=87.9)
print(f'cooled p = 3 with the full grid to 87.9 km/s (as in the submitted Table 2): sigma = {xa[0][0]:.1f}, rms = {xa[1]:.2f}; '
      f'fastest component at v_LSR = {V_SYS - vpost_cool(87.9):.1f} km/s')

head('Sections 5.3, 5.8: sensitivity of the cooled-gas fit')
show('slower shocks included (grid from 11 km/s), p = 3', fit(3, 'cl', vmin=11.0))
show('slower shocks included, p = 2.2', fit(2.2, 'cl', vmin=11.0))
show('slower shocks included, p free', fit(3, 'cl', True, vmin=11.0), True)
show('fit window -90 to -25 km/s, p = 3', fit(3, 'cl', win=(-90, -25)))
show('fit window -90 to -25 km/s, p = 2.2', fit(2.2, 'cl', win=(-90, -25)))
cA = np.sqrt(1 - 0.70**2)
show(f'projection factor {cA:.2f}, p = 3', fit(3, 'cl', vmax=V_EDGE_COOL / cA, c=(cA,)))
show(f'projection factor {cA:.2f}, p = 2.2', fit(2.2, 'cl', vmax=V_EDGE_COOL / cA, c=(cA,)))
show(f'projection factor {cA:.2f}, p free', fit(3, 'cl', True, vmax=V_EDGE_COOL / cA, c=(cA,)), True)
cs = tuple(np.linspace(0.525, 1, 10))
show('projection factors uniform in 0.5-1, p = 3', fit(3, 'cl', vmax=V_EDGE_COOL / 0.5, c=cs))
show('projection factors uniform in 0.5-1, p = 2.2', fit(2.2, 'cl', vmax=V_EDGE_COOL / 0.5, c=cs))
show('projection factors uniform in 0.5-1, p free', fit(3, 'cl', True, vmax=V_EDGE_COOL / 0.5, c=cs), True)
print(f'-25 km/s corresponds to v_shock = {brentq(lambda x: vpost_cool(x) - 21.0, 12, 100):.1f} km/s for cooled gas')
# two-temperature slabs: fraction f(v) of the column at 1e4 K, the rest at 5000 K.
# Line-integrated emissivity per H atom = x_HI / 1.823e18 (equal at both temperatures to 0.1%);
# line-centre value at the thermal width (CLOUDY's T_B / N_H) = x_HI T^-1/2: an upper bound on the effect.
xhi = {1e4: 1 - 10**-3.176, 5e3: 1.0}
for lab, e5 in (('line-integrated', xhi[5e3] / xhi[1e4]), ('line-centre', np.sqrt(2) * xhi[5e3] / xhi[1e4])):
    for case, f in (('half of the column at 5000 K', lambda vg: 0.5 + 0 * vg),
                    ('slow shocks (<40) at 5000 K', lambda vg: np.clip((vg - 40) / 20, 0, 1)),
                    ('fast shocks (>60) at 5000 K', lambda vg: 1 - np.clip((vg - 40) / 20, 0, 1))):
        show(f'{lab}: {case}, p = 3', fit(3, 'cl', weight=lambda vg, f=f: f(vg) + (1 - f(vg)) * e5))

head('Section 3.2: CLOUDY slabs')
for name in ('Tobs_sensitivity/v65_T8000', 'grid_Tobs10000/v65', 'Tobs_sensitivity/v65_T12000'):
    txt = (REPO / 'cloudy_inputs' / f'{name}.out').read_text()
    m = re.search(r'\n\s*Hydrogen\s+(-?[\d.]+)\s+(-?[\d.]+).*Log10 Mean Ionisation \(over radius\)', txt)
    print(f'{name}: mean H+ fraction = {10**float(m.group(2)):.1e}')
with open(REPO / 'cloudy_inputs' / 'Tobs_sensitivity' / 'tb21.csv') as fh:
    rows = [(float(q['v_shock_kms']), float(q['T_obs_K']), float(q['TB21_K'])) for q in csv.DictReader(fh)]
ratios = [q[2] * np.sqrt(2 * np.pi) * np.sqrt(KB * q[1] / MH) / 1e5 * X_HI_CONST / 10**cloudy_parameters(q[0], q[1])[1] for q in rows]
print(f'line-integrated 21 cm emission of the CLOUDY slabs / (N_H / 1.823e18): {min(ratios):.2f}-{max(ratios):.2f} (9 models)')

head('Section 5.3: widths and optical depth')
print('thermal sigma of H at 1e4, 5000, 100 K: ' + ', '.join(f'{np.sqrt(KB*T/MH)/1e5:.1f}' for T in (1e4, 5000, 100)) +
      f' km/s; channel and 5-channel smoothing: {np.sqrt(dv**2 + (5*dv)**2)/np.sqrt(12):.1f} km/s')
for T in (1e4, 100):
    print(f'non-thermal part of sigma = 15.9 km/s for gas at {T:.0f} K: {np.sqrt(15.9**2 - KB*T/MH/1e10 - (dv**2 + (5*dv)**2)/12):.1f} km/s')
for Tsp in (5000, 100):
    print(f'T_spin = {Tsp} K: tau at the brightest fitted channel ({smf[s].max():.1f} K) = {-np.log(1 - smf[s].max()/Tsp):.3f}; '
          f'N_HI from the optically thick expression / thin = {np.sum(-np.log(1-np.clip(raw[s],0,None)/Tsp)*Tsp)/np.sum(np.clip(raw[s],0,None)):.2f}')

head('Section 5.2: positions (J2000) and separations')
Rg = np.array([[-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
               [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
               [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669]])      # ICRS -> Galactic
M = np.array([[0.9999256782, -0.0111820611, -0.0048579477], [0.0111820610, 0.9999374784, -0.0000271765],
              [0.0048579479, -0.0000271474, 0.9999881997]])                          # FK4 B1950 -> FK5 J2000
vec = lambda a, dd: np.array([np.cos(np.radians(dd)) * np.cos(np.radians(a)), np.cos(np.radians(dd)) * np.sin(np.radians(a)), np.sin(np.radians(dd))])
sep = lambda p, q: np.degrees(np.arccos(np.clip(p @ q, -1, 1))) * 3600
hms = lambda h, m, s_: 15 * (h + m / 60 + s_ / 3600); dms = lambda dd, m, s_: dd + m / 60 + s_ / 3600
P = {'A': Rg.T @ vec(189.291, 3.025), 'B': Rg.T @ vec(189.176, 2.940), 'edge -91': Rg.T @ vec(189.221, 2.995)}
CL = {'IC443 B': vec(hms(6, 17, 16.1), dms(22, 25, 15)), 'IC443 C': vec(hms(6, 17, 42.4), dms(22, 21, 27))}   # Reach et al. 2024, Table 1
BS = {'B&S86 arc (-97 to -105)': M @ vec(hms(6, 14, 15), dms(22, 27, 0))}                                    # B1950, from Hirschauer et al. 2009
for k, p in P.items():
    print(f'{k}: ' + ', '.join(f'{c} {sep(p, q):.0f}"' for c, q in {**CL, **BS}.items()))
print('B&S86 arc to clump IC443 B: ' + f'{sep(BS["B&S86 arc (-97 to -105)"], CL["IC443 B"]):.0f}"')
