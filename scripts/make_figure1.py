"""
Figure 1: (a) CGPS 1420 MHz continuum with contours of the HI excess
(mean T_B over v_LSR = -55 ... -25 km/s, 5x5-pixel smoothing; contours at 1.3, 2.5, 5,
10 K, where 1.3 K is the off-source level plus 3 sigma); (b) map of the HI
wing edge (3 sigma criterion, 5x5-pixel average spectra) with continuum contours.
The cross and dashed circle mark the centre and radius of IC 443 from Green's catalogue
(06h17m00s, +22d34m; diameter 45 arcmin). A: position of maximum HI excess (spectrum
of Figure 2); B: most negative wing edge that is stable against the detection threshold;
NE: maximum HI excess in the northeastern quadrant.

Inputs (data/): excess_map.npy, edge_map_3sigma.npy, excess_map_origin.json (written by
ic443_hi_diagnostic.py and ic443_fastest_hi.py) and the CGPS 1420 MHz image
CGPS_MER2_1420_MHz_I_image.fits (from the Canadian Astronomy Data Centre).
The Galactic coordinates are taken from the linear CGPS grid (CRVAL/CRPIX/CDELT);
no astropy is needed.
Usage: python make_figure1.py [continuum.fits] [data_dir]
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

REPO = Path(__file__).resolve().parent.parent
CONT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / 'data' / 'CGPS_MER2_1420_MHz_I_image.fits'
DATA = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / 'data'
OUT = REPO / 'figures' / 'continuum_HIexcess_overlay.png' if len(sys.argv) <= 3 else Path(sys.argv[3])

GREEN_LB = (189.033, 2.978)          # Green-catalogue centre (06h17m00s, +22d34m) in Galactic coordinates
R_DEG = 22.5 / 60.0
POS = {'A': (189.291, 3.025), 'B': (189.181, 2.930), 'NE': (189.166, 3.225)}
LEVELS = [1.3, 2.5, 5.0, 10.0]   # K; 1.3 K = off-source level + 3 sigma of the excess map (0.28 + 3 x 0.32 K)


def read_fits(path):
    hdr = {}
    with open(path, 'rb') as f:
        done = False
        while not done:
            blk = f.read(2880)
            for i in range(0, 2880, 80):
                c = blk[i:i + 80].decode('ascii', 'replace')
                if c.startswith('END'):
                    done = True
                    break
                if '=' in c[:10]:
                    k, v = c[:8].strip(), c[10:].split('/')[0].strip().strip("'").strip()
                    try:
                        v = float(v)
                    except ValueError:
                        pass
                    hdr[k] = v
        shape = [int(hdr['NAXIS%d' % i]) for i in range(int(hdr['NAXIS']), 0, -1)]
        dt = {-32: '>f4', -64: '>f8', 16: '>i2', 32: '>i4'}[int(hdr['BITPIX'])]
        data = np.frombuffer(f.read(int(np.prod(shape)) * np.dtype(dt).itemsize), dt).reshape(shape).astype(float)
    if 'BLANK' in hdr and int(hdr['BITPIX']) > 0:
        data[data == hdr['BLANK']] = np.nan
    return hdr, np.squeeze(data * hdr.get('BSCALE', 1.0) + hdr.get('BZERO', 0.0))


h, cont = read_fits(CONT)
orig = json.load(open(DATA / 'excess_map_origin.json'))
x1, y1 = orig['x0_pixel_in_mosaic'], orig['y0_pixel_in_mosaic']
excess = np.load(DATA / 'excess_map.npy')
edge = np.load(DATA / 'edge_map_3sigma.npy')
ny, nx = excess.shape
cont = cont[y1:y1 + ny, x1:x1 + nx]

lpix = lambda x: h['CRVAL1'] + (x + 1 - h['CRPIX1']) * h['CDELT1']
bpix = lambda y: h['CRVAL2'] + (y + 1 - h['CRPIX2']) * h['CDELT2']
ext = [lpix(x1 - 0.5), lpix(x1 + nx - 0.5), bpix(y1 - 0.5), bpix(y1 + ny - 0.5)]
L = lpix(x1 + np.arange(nx)); B = bpix(y1 + np.arange(ny))

levels = LEVELS
ccol = plt.cm.autumn(np.linspace(0, 1, len(LEVELS)))
mk = {'A': dict(marker='o', ms=11), 'B': dict(marker='s', ms=10), 'NE': dict(marker='^', ms=11)}

fig, axs = plt.subplots(1, 2, figsize=(12.4, 6.2))
axs[0].imshow(cont, origin='lower', extent=ext, cmap='gray_r', aspect='equal',
              vmin=np.nanpercentile(cont, 2), vmax=np.nanpercentile(cont, 98))
axs[0].contour(L, B, excess, levels=levels, colors=ccol, linewidths=1.0)
im = axs[1].imshow(edge, origin='lower', extent=ext, cmap='viridis', aspect='equal', vmin=-95, vmax=-20)
axs[1].contour(L, B, cont, levels=np.nanpercentile(cont, [80, 95]), colors='0.25', linewidths=0.8)
cb = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.03)
cb.set_label(r'HI wing edge, $v_{\rm LSR}$ (km s$^{-1}$)')

for a, lab in zip(axs, ('(a)', '(b)')):
    a.plot(*GREEN_LB, '+', color='cyan', ms=16, mew=2.2)
    a.add_patch(Circle(GREEN_LB, R_DEG, fill=False, color='cyan', ls='--', lw=1.4))
    for k, (l, b) in POS.items():
        a.plot(l, b, color='magenta', mfc='none', mew=2.2, lw=0, **mk[k])
        a.text(l - 0.025, b + 0.022, k, color='magenta', fontsize=11, fontweight='bold')
    a.set_xlim(ext[0], ext[1]); a.set_ylim(ext[2], ext[3])
    a.set_xlabel('Galactic longitude (deg)'); a.set_ylabel('Galactic latitude (deg)')
    a.text(0.03, 0.96, lab, transform=a.transAxes, fontsize=13, fontweight='bold', va='top', color='black')

handles = [Line2D([0], [0], color=c, lw=1.8, label='%.1f K' % lv) for c, lv in zip(ccol, levels)]
handles += [Line2D([0], [0], color='cyan', marker='+', lw=0, ms=12, mew=2, label='IC 443 center'),
            Line2D([0], [0], color='cyan', ls='--', lw=1.4, label='radius 22.5 arcmin')]
handles += [Line2D([0], [0], color='magenta', mfc='none', mew=2, lw=0, label=k, **mk[k]) for k in POS]
fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=9, frameon=True,
           title=r'(a) HI excess (mean $T_B$, $v_{\rm LSR}=-55$ to $-25$ km s$^{-1}$); (b) gray: 1420 MHz contours; positions A, B, NE', title_fontsize=9,
           bbox_to_anchor=(0.5, 0.0))
plt.tight_layout(rect=[0, 0.13, 1, 1])
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=200)
print('saved', OUT, '| contour levels (K):', np.round(levels, 2))
