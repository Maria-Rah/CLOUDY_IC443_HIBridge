"""
IC 443 HI diagnostic: where is the fastest shocked HI, and how fast is it?

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
    sm = uniform_filter1d(spec, 5)
    np.savetxt(f'spectrum_{name}.dat', np.c_[vel, spec, sm], fmt='%10.4f',
               header='v_LSR(km/s)  T_B_raw(K)  T_B_smoothed(K)  [5x5 average]')
    return out

results = []
inner = (sep > 0.3 * R_ARCMIN) & (sep < 1.2 * R_ARCMIN)
for qname, (lo, hi) in {'NE': (0, 90), 'SE': (90, 180), 'SW': (180, 270), 'NW': (270, 360)}.items():
    m = inner & (pa >= lo) & (pa < hi)
    iy, ix = np.unravel_index(np.nanargmax(np.where(m, excess, -np.inf)), excess.shape)
    results.append(report(f'max_excess_{qname}', iy, ix))
for name, (l, b) in {'paper_position': (189.29, 3.02), 'earlier_NE_position': (189.054, 3.229)}.items():
    px, py = sky2pix(SkyCoord(l * u.deg, b * u.deg, frame='galactic'))
    results.append(report(name, int(round(py)) - y1, int(round(px)) - x1))
json.dump(results, open('diagnostic_summary.json', 'w'), indent=1)
for r in results:
    e3 = r['edges']['3.0sigma']
    print(f"{r['name']:22s} l={r['l']:.3f} b={r['b']:.3f} PA={r['PA_deg']:4.0f} r/R={r['r_over_R']:.2f} "
          f"excess={r['excess_K']:5.2f} K | edge(2.5/3/4 sig)={r['edges']['2.5sigma']['edge']}/{e3['edge']}/"
          f"{r['edges']['4.0sigma']['edge']} | v_sh(3sig) ad={e3['vshock_adiabatic']} cool={e3['vshock_cooled']}"
          f" | deproj ad={e3['deprojected_adiabatic']} cool={e3['deprojected_cooled']}")

# ---- edge-velocity map (3 sigma) on a coarse grid for the figure ----
step = 3
edge_map = np.full(excess.shape, np.nan)
for iy in range(0, excess.shape[0], step):
    for ix in range(0, excess.shape[1], step):
        if sep[iy, ix] < 1.3 * R_ARCMIN:
            edge_map[iy:iy + step, ix:ix + step] = wing_edge(vel, cube5[:, iy, ix], 3.0)[0]

cont = np.squeeze(fits.open(cont_path, memmap=False)[0].data)[y1:y2, x1:x2]
fig, axs = plt.subplots(1, 2, figsize=(13, 6), subplot_kw={'projection': wcel[y1:y2, x1:x2]})
axs[0].imshow(cont, origin='lower', cmap='gray_r', vmin=np.nanpercentile(cont, 2), vmax=np.nanpercentile(cont, 98))
axs[0].contour(excess, levels=np.nanpercentile(excess, [70, 80, 90, 96, 99]), cmap='autumn', linewidths=1)
im = axs[1].imshow(edge_map, origin='lower', cmap='viridis', vmin=-90, vmax=-20)
plt.colorbar(im, ax=axs[1], label='wing edge, 3 sigma (km/s)')
for a in axs:
    gx, gy = sky2pix(GREEN); gx -= x1; gy -= y1
    a.plot(gx, gy, 'c+', ms=14, mew=2)
    a.add_patch(plt.Circle((gx, gy), R_ARCMIN / 0.3, fill=False, color='c', ls='--'))
    for r in results:
        px, py = sky2pix(SkyCoord(r['l'] * u.deg, r['b'] * u.deg, frame='galactic')); px -= x1; py -= y1
        a.plot(px, py, 'o', mfc='none', mec='red' if 'max' in r['name'] else 'magenta', ms=9, mew=1.8)
        a.text(px + 3, py + 3, r['name'].replace('max_excess_', ''), color='red', fontsize=7)
    a.coords[0].set_axislabel('Galactic Longitude'); a.coords[1].set_axislabel('Galactic Latitude')
axs[0].set_title('1420 MHz + HI excess (-55..-25 km/s); cross/circle = Green centre, R=22.5\'')
axs[1].set_title('HI wing edge (3 sigma, 5x5 average)')
plt.tight_layout(); plt.savefig('diagnostic_map.png', dpi=130)
print('wrote diagnostic_summary.json, diagnostic_map.png, spectrum_*.dat, excess_map.npy')
