"""
Recomputes, from the files in this repository, the numbers quoted in the paper
(Sections 2.2-2.3, 3.2, 4.1-4.2, 5.1-5.2, 5.4 and Tables 1-2), and prints them next to
the values in the text. Inputs: data/observed_spectrum.dat, data/excess_map.npy,
data/excess_map_origin.json, results/*.json, cloudy_inputs/*/tb21.csv (written by
extract_tb21.py). Usage: python reproduce_numbers.py
"""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import brentq, least_squares
from bridge import jump, cloudy_parameters, KB, N_PRE

REPO = Path(__file__).resolve().parent.parent
V_SYS = -4.0
out = []
def rep(label, value, paper):
    out.append((label, value, paper)); print(f"{label:62s} {value:>14s}   paper: {paper}")

def v_ad(off):   return brentq(lambda v: jump(v)[1] - off, 12, 600)
def v_cool(off, t=1e4): return brentq(lambda v: v * (1 - N_PRE * KB * t / jump(v)[2]) - off, 12, 600)

# ---- spectrum at position A: noise and wing edge (Section 2.3) ----
d = np.genfromtxt(REPO / 'data' / 'observed_spectrum.dat', comments='#')
d = d[np.argsort(d[:, 0])]; v, raw, smf = d[:, 0], d[:, 1], d[:, 2]
good = np.isfinite(raw)
noise = np.std(raw[good & (v >= 40) & (v <= 85)])
sm = uniform_filter1d(np.where(good, raw, 0.0), 5)
def edge(n):
    imax = int(np.argmax(np.where(good, sm, -np.inf))); thr = n * noise
    for i in range(imax, 9, -1):
        if np.all(sm[i - 9:i + 1] < thr):
            return v[i]
rep('sigma_rms (K)', f'{noise:.2f}', '0.71'); rep('3 sigma threshold (K)', f'{3*noise:.1f}', '2.1')
for n, p in ((2.5, '-69.9'), (3.0, '-69.1'), (4.0, '-61.7')):
    rep(f'wing edge, {n} sigma (km/s)', f'{edge(n):.1f}', p)
offA = round(abs(edge(3.0) - V_SYS), 1); rep('offset at A (km/s)', f'{offA:.1f}', '65.1')

# ---- kinematic limits (Sections 4.1, 5.1, 5.4) ----
rep('A: v_shock adiabatic / cooled (km/s)', f'{v_ad(offA):.1f} / {v_cool(offA):.1f}', '87.9 / 66.8')
lo, hi = abs(edge(4.0) - V_SYS), abs(edge(2.5) - V_SYS)
rep('A: adiabatic range over the edge range', f'{v_ad(lo):.0f}-{v_ad(hi):.0f}', '78-89')
rep('A: cooled range over the edge range', f'{v_cool(lo):.0f}-{v_cool(hi):.0f}', '60-68')
fast = json.load(open(REPO / 'results' / 'fastest_hi.json'))
B = min((r for r in fast if abs(r['edge_3'] - r['edge_4']) < 1.0), key=lambda r: r['edge_3_median3x3'])   # most negative stable edge
offB = abs(B['edge_3_median3x3'] - V_SYS)
rep('B: l, b; edge 2.5/3/4 sigma (km/s)', f"{B['l']}, {B['b']}; {B['edge_2p5']}/{B['edge_3']}/{B['edge_4']}", '189.176, 2.940; -88.0/-82.3/-81.4')
rep('B: offset; adiabatic / cooled (km/s)', f'{offB:.1f}; {v_ad(offB):.1f} / {v_cool(offB):.1f}', '78.3; 105 / 80')
rep('B: r/R', f"{B['r_over_R']}", '0.39')
rep('projection cosines A, B', f"{np.sqrt(1-0.70**2):.2f}, {np.sqrt(1-B['r_over_R']**2):.2f}", '0.71, 0.92')
sweep = [abs(-69.1 - vs) for vs in (-6.85, -6.14, -4.55, -4.0)]
rep('maser sweep: adiabatic (km/s)', ', '.join(f'{v_ad(o):.1f}' for o in sweep), '84.1 ... 87.9')
rep('maser sweep: cooled (km/s)', ', '.join(f'{v_cool(o):.1f}' for o in sweep), '64.0, 64.7, 66.2, 66.8')
o5 = abs(-69.1 - 5.0); rep('v_sys = +5: offset; adiabatic / cooled', f'{o5:.1f}; {v_ad(o5):.1f} / {v_cool(o5):.1f}', '74.1; 99.8 / 75.6')
rep('P2 at 65 km/s (dyn/cm2)', f'{jump(65)[2]:.2e}', '7.9e-10')

# ---- quadrant maxima and other positions (Section 2.2) ----
summ = {r['name']: r for r in json.load(open(REPO / 'results' / 'diagnostic_summary.json'))}
A = summ['max_excess_SE']
rep('A: l, b; sep (arcmin); PA; r/R', f"{A['l']}, {A['b']}; {A['sep_arcmin']}; {A['PA_deg']:.0f}; {A['r_over_R']}", '189.291, 3.025; 15.7; 142; 0.70')
rep('excess at A / NE quadrant maximum (K)', f"{A['excess_K']:.2f} / {summ['max_excess_NE']['excess_K']:.2f}", '13.6 / 3.7')
rep('NE quadrant maximum: wing edge (km/s)', f"{summ['max_excess_NE']['edges']['3.0sigma']['edge']}", '-52')
rep('adjacent pixel to A: edge 3 sigma (km/s)', f"{summ['paper_position']['edges']['3.0sigma']['edge']}", '-63.3')
offS = summ['max_excess_SW']['edges']['3.0sigma']['offset']
rep('offsets toward S, A, B (km/s)', f'{offS:.0f}, {offA:.0f}, {offB:.0f}', '57-78')
f1 = min(fast, key=lambda r: r['edge_3_median3x3'])   # most negative edge overall
rep('most negative edge: l, b; edge 3 / 4 sigma; change', f"{f1['l']}, {f1['b']}; {f1['edge_3']} / {f1['edge_4']}; {abs(f1['edge_3']-f1['edge_4']):.0f}", '-91 ; 12')

# ---- excess-map statistics (Section 2.2) ----
ex = np.load(REPO / 'data' / 'excess_map.npy'); og = json.load(open(REPO / 'data' / 'excess_map_origin.json'))
L = 188.75 + (og['x0_pixel_in_mosaic'] + np.arange(ex.shape[1]) + 1 - 513) * (-0.0049999994)
Bb = 3.0 + (og['y0_pixel_in_mosaic'] + np.arange(ex.shape[0]) + 1 - 513) * 0.0049999994
LL, BB = np.meshgrid(L, Bb)
r = np.hypot((LL - 189.033) * np.cos(np.radians(3)), BB - 2.978) * 60
offsrc = ex[r > 1.3 * 22.5]
rep('excess map off-source median / rms (K)', f'{np.median(offsrc):.2f} / {1.4826*np.median(abs(offsrc-np.median(offsrc))):.2f}', '0.28 / 0.32')

# ---- CLOUDY grid (Section 3.2, Table 1) ----
def read_tb(folder):
    f = REPO / 'cloudy_inputs' / folder / 'tb21.csv'
    with open(f) as fh:
        return [(float(r['v_shock_kms']), float(r['T_obs_K']), float(r['TB21_K']), float(r['Tspin_K'])) for r in csv.DictReader(fh)]
grid = sorted(read_tb('grid_Tobs10000'))
vg = np.array([g[0] for g in grid]); tb = np.array([g[2] for g in grid])
rep('grid size; T_B range (K)', f'{len(grid)}; {tb.min()}-{tb.max()}', '20; 2.32-47.1')
rep('T_B slope d ln T_B / d ln v', f'{np.polyfit(np.log(vg), np.log(tb), 1)[0]:.1f}', '2.0')
rep('T_spin at 65 / 80 (K); max T_B/T_spin', f"{grid[list(vg).index(65)][3]:.0f} / {grid[list(vg).index(80)][3]:.0f}; {max(g[2]/g[3] for g in grid):.3f}", '5860 / 5370; tau <~ 0.01')
tobs = read_tb('Tobs_sensitivity')
for vs in (65, 80, 100):
    t = {g[1]: g[2] for g in tobs if g[0] == vs}
    e = {T: t[T] / 10**cloudy_parameters(vs, T)[1] for T in t}
    rep(f'T_obs: v={vs}: T_B(8k/10k/12k); emissivity change', f"{t[8000]}/{t[10000]}/{t[12000]}; {100*(e[8000]/e[10000]-1):+.0f}%/{100*(e[12000]/e[10000]-1):+.0f}%",
        {65: '35.7/25.9/18.9; +10..+13% / -12%', 80: '-', 100: '- / T_spin 4950'}[vs])
cr = read_tb('cosmic_ray_test')[0]; rep('cosmic-ray test: T_B; T_spin', f'{cr[2]}; {cr[3]:.0f}', '25.9; 4750')

# ---- wing model (Section 4.2, Table 2) ----
logN = np.array([cloudy_parameters(x)[1] for x in vg]); base = tb / 10**logN * np.gradient(vg)
sel = (v >= -70) & (v <= -25); vo, To = v[sel], smf[sel]
rep('channels in fit range', f'{sel.sum()}', '55')
def comps(mapping):
    return V_SYS - (np.array([jump(x)[1] for x in vg]) if mapping == 'ad' else np.array([x * (1 - N_PRE * KB * 1e4 / jump(x)[2]) for x in vg]))
def fit(p, mapping='ad', free=False):
    vc = comps(mapping); best = None
    for s0 in (6, 10, 15, 20, 25, 35):
        for p0 in ((p,) if not free else (2.5, 4, 6)):
            def resid(q):
                pp = q[2] if free else p
                w = np.exp(q[1]) * base * vg**(-pp)
                return (w[None, :] * np.exp(-0.5 * ((vo[:, None] - vc[None, :]) / q[0])**2)).sum(1) - To
            x0 = [s0, np.log(15 / (base * vg**(-p0)).max())] + ([p0] if free else [])
            bounds = ([1, -300] + ([0] if free else []), [80, 300] + ([12] if free else []))
            rr = least_squares(resid, x0, bounds=bounds)
            rms = np.sqrt(np.mean(rr.fun**2))
            if best is None or rms < best[1]:
                best = (rr.x, rms, rr.fun)
    return best
for lab, p, mp, fr, paper in (('adiabatic p=3', 3, 'ad', False, '19.2, 0.93'), ('adiabatic p=8/3', 8/3, 'ad', False, '17.9, 0.96'),
                              ('adiabatic p free', 3, 'ad', True, 'p=4.9: 23.0, 0.89'), ('adiabatic p=2.2', 2.2, 'ad', False, '15.5, 1.04'),
                              ('adiabatic p=1', 1, 'ad', False, '8.4, 2.39'), ('adiabatic p=1.5', 1.5, 'ad', False, 'rms 1.4'),
                              ('cooled p=3', 3, 'cl', False, '15.4, 1.06'), ('cooled p free', 3, 'cl', True, 'p=6.7: 22.2, 0.86')):
    x, rms, res = fit(p, mp, fr)
    val = (f'p={x[2]:.1f}: ' if fr else '') + f'{x[0]:.1f}, {rms:.2f}'
    rep(f'fit {lab}: sigma, rms', val, paper)
    if lab == 'adiabatic p=3':
        i = np.argmax(abs(res)); rep('p=3: largest residual (K) at v (km/s)', f'{abs(res[i]):.1f} at {vo[i]:.0f}', '1.7 at -32')
        rms6 = [fit(pp)[1] for pp in (2.5, 4, 6)]
        rep('p=2.5..6: rms spread (K)', f'{max(rms6)-min(rms6):.2f}', '< 0.1')
        sig3 = x[0]; K3 = np.exp(x[1])
lin = np.polyfit(vo, To, 1); rep('straight line: rms (K)', f'{np.sqrt(np.mean((np.polyval(lin, vo) - To)**2)):.2f}', '0.80')
vc = comps('ad'); w = K3 * base * vg**(-3.0)
def frac(vv, cond):
    c = w * np.exp(-0.5 * ((vv - vc) / sig3)**2); return 100 * c[cond].sum() / c.sum()
rep('share of v_shock<40 at -45 / -25 km/s (%)', f'{frac(-45, vg < 40):.0f} / {frac(-25, vg < 40):.0f}', '72 / 89')
rep('share of v_shock>=60 at -65 km/s (%)', f'{frac(-65, vg >= 60):.0f}', '30')
rep('-25 km/s corresponds to v_shock (km/s)', f'{v_ad(21.0):.1f}', '31.1')
