"""
IC 443 HI follow-up: full-resolution map of the HI wing edge and the positions of the
fastest shocked HI, with a stability check against threshold and one-pixel shifts.

Runs on the CGPS MER2 HI cube and 1420 MHz image (needs astropy, scipy, matplotlib).
Usage:
    python ic443_hi_diagnostic.py CGPS_MER2_HI_line_image.fits CGPS_MER2_1420_MHz_I_image.fits

What it does (same recipe as the paper, Section 2):
  * IC 443 centre and size from Green's catalogue: 06h17m00s +22d34m, 45 arcmin.
  * Excess map = mean T_B over v_LSR = -55 ... -25 km/s, smoothed 5x5 pixels.
  * Spectra = 5x5-pixel averages; noise = rms of the unsmoothed spectrum at 40-85 km/s;
    wing edge = first channel, scanning from the spectral maximum toward negative
    velocities, below which the 5-channel-smoothed spectrum stays under n*noise for
    10 consecutive channels (n = 2.5, 3, 4).
  * For the excess maximum in each quadrant (NE, SE, SW, NW) and for the two positions
    used so far, reports the edge, the offset from v_sys = -4 km/s, the shock velocity
    (adiabatic and cooled gas), the projected radius r/R, and a spherical-shell
    deprojection.
Outputs (in the current folder): diagnostic_summary.json, diagnostic_map.png,
spectrum_<name>.dat for every reported position, excess_map.npy + excess_map_origin.json.
"""
import json
import sys
import numpy as np
from scipy.ndimage import uniform_filter, uniform_filter1d
from scipy.optimize import brentq
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GREEN = SkyCoord('06h17m00s', '+22d34m00s', frame='icrs')
R_ARCMIN = 22.5
V_SYS = -4.0
BAND = (-55.0, -25.0)
NOISE_WIN = (40.0, 85.0)
HALF = 120                      # half size of the analysed box (pixels, 36 arcmin)
BAD_CHANNELS_1BASED = list(range(1, 19)) + [272]
NSIG = (2.5, 3.0, 4.0)

# ---- Rankine-Hugoniot conversions (paper Eqs. 3 and 4) ----
G, KB, MH, NPRE, TPRE, TOBS = 5/3, 1.380649e-16, 1.673e-24, 15.0, 7000.0, 1e4
def _jump(v):
    vs = v * 1e5; rho1, p1 = NPRE * MH, NPRE * KB * TPRE
    M = vs / np.sqrt(G * p1 / rho1)
    r = (G + 1) * M**2 / ((G - 1) * M**2 + 2)
    p2 = p1 * (2 * G * M**2 - (G - 1)) / (G + 1)
    return v * (1 - 1 / r), v * (1 - NPRE * KB * TOBS / p2)
def vshock(offset, cooled=False):
    k = 1 if cooled else 0
    try:
        return brentq(lambda v: _jump(v)[k] - offset, 12.0, 600.0)
    except ValueError:
        return float('nan')

def wing_edge(v, raw, nsig):
    """v ascending; raw = 5x5-averaged spectrum; returns edge velocity or nan."""
    good = np.isfinite(raw)
    noise = np.nanstd(raw[(v >= NOISE_WIN[0]) & (v <= NOISE_WIN[1]) & good])
    sm = uniform_filter1d(np.nan_to_num(raw), 5)
    imax = int(np.nanargmax(np.where(good, sm, -np.inf)))
    thr = nsig * noise
    for i in range(imax, 9, -1):
        if np.all(sm[i - 9:i + 1] < thr):
            return float(v[i]), float(noise)
    return float('nan'), float(noise)

# ---- read data ----
hi_path, cont_path = sys.argv[1], sys.argv[2]
hdu = fits.open(hi_path, memmap=True, do_not_scale_image_data=True)[0]   # scaled below, only for the cut-out
w3 = WCS(hdu.header)
wcel = w3.celestial
nchan = hdu.header['NAXIS3']
chan = np.arange(nchan)
h = hdu.header
vel = h['CRVAL3'] + (chan + 1 - h['CRPIX3']) * h['CDELT3']
if abs(h['CDELT3']) > 50:          # CGPS velocity axis is in m/s
    vel = vel / 1e3
print('velocity axis: %.2f to %.2f km/s, %d channels' % (vel.min(), vel.max(), nchan))
def sky2pix(c):
    g = c.galactic
    px, py = wcel.wcs_world2pix([[g.l.deg, g.b.deg]], 0)[0]
    return float(px), float(py)


def pix2sky(px, py):
    l, b = wcel.wcs_pix2world(np.ravel(px), np.ravel(py), 0)
    return SkyCoord(l * u.deg, b * u.deg, frame='galactic')


x0, y0 = [int(round(c)) for c in sky2pix(GREEN)]
x1, x2, y1, y2 = x0 - HALF, x0 + HALF, y0 - HALF, y0 + HALF
data = hdu.data
rawcut = np.array(data[0, :, y1:y2, x1:x2] if data.ndim == 4 else data[:, y1:y2, x1:x2])
cube = rawcut.astype(float)
if h.get('BLANK') is not None:
    cube[rawcut == h['BLANK']] = np.nan
cube = cube * h.get('BSCALE', 1.0) + h.get('BZERO', 0.0)
print('cut-out cube: %s, T_B range %.1f to %.1f K' % (cube.shape, np.nanmin(cube), np.nanmax(cube)))
for c in BAD_CHANNELS_1BASED:
    cube[c - 1] = np.nan
order = np.argsort(vel); vel = vel[order]; cube = cube[order]

band = (vel >= BAND[0]) & (vel <= BAND[1])
excess = uniform_filter(np.nanmean(cube[band], axis=0), 5)
np.save('excess_map.npy', excess)
json.dump({'x0_pixel_in_mosaic': x1, 'y0_pixel_in_mosaic': y1, 'note': '0-based pixel of excess[0,0]'},
          open('excess_map_origin.json', 'w'))
cube5 = uniform_filter(np.nan_to_num(cube), size=(1, 5, 5))
cube5[np.all(np.isnan(cube), axis=(1, 2))] = np.nan     # flagged channels stay excluded (noise, spectra)

# sky geometry of every pixel
yy, xx = np.mgrid[y1:y2, x1:x2]
sky = pix2sky(xx, yy).icrs
sep = GREEN.separation(sky).arcmin.reshape(xx.shape)
pa = (GREEN.position_angle(sky).deg % 360).reshape(xx.shape)

def report(name, iy, ix):
    c = pix2sky(ix + x1, iy + y1)[0]
    spec = cube5[:, iy, ix]
    out = {'name': name, 'l': round(float(c.galactic.l.deg), 3), 'b': round(float(c.galactic.b.deg), 3),
           'ra': c.icrs.ra.to_string(u.hour, precision=1), 'dec': c.icrs.dec.to_string(precision=0),
           'sep_arcmin': round(float(sep[iy, ix]), 1), 'PA_deg': round(float(pa[iy, ix]), 0),
           'excess_K': round(float(excess[iy, ix]), 2), 'edges': {}}
    rr = sep[iy, ix] / R_ARCMIN
    cos_t = np.sqrt(1 - rr**2) if rr < 1 else float('nan')
    for n in NSIG:
        e, noise = wing_edge(vel, spec, n)
        off = abs(e - V_SYS)
        out['edges'][f'{n}sigma'] = {
            'edge': round(e, 1), 'offset': round(off, 1),
            'vshock_adiabatic': round(vshock(off), 1), 'vshock_cooled': round(vshock(off, True), 1),
            'deprojected_adiabatic': round(vshock(off / cos_t), 1) if np.isfinite(cos_t) else None,
            'deprojected_cooled': round(vshock(off / cos_t, True), 1) if np.isfinite(cos_t) else None}
        out['noise_K'] = round(noise, 2)
    out['r_over_R'] = round(float(rr), 2)
    sm = uniform_filter1d(np.nan_to_num(spec), 5); sm[~np.isfinite(spec)] = np.nan
    np.savetxt(f'spectrum_{name}.dat', np.c_[vel, spec, sm], fmt='%10.4f',
               header='v_LSR(km/s)  T_B_raw(K)  T_B_smoothed(K)  [5x5 average]')
    return out


# ---- full-resolution wing-edge maps (2.5, 3, 4 sigma) inside 1.1 R ----
inside = sep < 1.1 * R_ARCMIN
emaps = {n: np.full(excess.shape, np.nan) for n in NSIG}
for iy, ix in zip(*np.where(inside)):
    for n in NSIG:
        emaps[n][iy, ix] = wing_edge(vel, cube5[:, iy, ix], n)[0]
np.save('edge_map_3sigma.npy', emaps[3.0])
from scipy.ndimage import median_filter
e3_med = median_filter(np.nan_to_num(emaps[3.0], nan=0.0), size=3)   # 3x3 median = one-pixel stability
order_px = np.argsort(np.where(inside & (excess > 3.0), e3_med, np.inf), axis=None)
rows, taken = [], []
for k in order_px:
    iy, ix = np.unravel_index(k, excess.shape)
    if any(abs(iy - a) < 10 and abs(ix - b) < 10 for a, b in taken):
        continue
    taken.append((iy, ix))
    c = pix2sky(ix + x1, iy + y1)[0]
    e = {n: emaps[n][iy, ix] for n in NSIG}
    rr = sep[iy, ix] / R_ARCMIN; cos_t = np.sqrt(max(1 - rr**2, 1e-6))
    off = abs(e3_med[iy, ix] - V_SYS)
    rows.append(dict(l=round(float(c.galactic.l.deg), 3), b=round(float(c.galactic.b.deg), 3),
                     sep_arcmin=round(float(sep[iy, ix]), 1), PA=round(float(pa[iy, ix])), r_over_R=round(float(rr), 2),
                     excess_K=round(float(excess[iy, ix]), 1), edge_2p5=round(e[2.5], 1), edge_3=round(e[3.0], 1),
                     edge_4=round(e[4.0], 1), edge_3_median3x3=round(float(e3_med[iy, ix]), 1),
                     vsh_ad=round(vshock(off), 1), vsh_cool=round(vshock(off, True), 1),
                     deproj_ad=round(vshock(off / cos_t), 1), deproj_cool=round(vshock(off / cos_t, True), 1)))
    spec = cube5[:, iy, ix]
    np.savetxt('spectrum_fast_%d.dat' % len(rows), np.c_[vel, spec, np.where(np.isfinite(spec), uniform_filter1d(np.nan_to_num(spec), 5), np.nan)], fmt='%10.4f',
               header='v_LSR(km/s)  T_B_raw(K)  T_B_smoothed(K)  [5x5 average]')
    if len(rows) == 6:
        break
json.dump(rows, open('fastest_hi.json', 'w'), indent=1)
print('fastest shocked HI (3x3-median 3-sigma edge, excess > 3 K, separated by > 3 arcmin):')
for i, r in enumerate(rows, 1):
    print(f"{i}: l={r['l']} b={r['b']} PA={r['PA']:3d} r/R={r['r_over_R']} excess={r['excess_K']} K | "
          f"edge 2.5/3/4 sig = {r['edge_2p5']}/{r['edge_3']}/{r['edge_4']}, 3x3 median {r['edge_3_median3x3']} | "
          f"v_sh ad={r['vsh_ad']} cool={r['vsh_cool']} | deproj ad={r['deproj_ad']} cool={r['deproj_cool']}")
print('wrote fastest_hi.json, edge_map_3sigma.npy, spectrum_fast_*.dat')
