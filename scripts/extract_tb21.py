"""
Extracts the 21 cm brightness temperature and spin temperature from CLOUDY outputs.

CLOUDY prints, for each iteration, a line containing 'T(<nH/Tspin>)' and 'TB21cm:';
the values of the last iteration are reported.

Usage:  python extract_tb21.py ../cloudy_inputs/grid_Tobs10000   (writes tb21.csv there)
"""
import csv
import re
import sys
from pathlib import Path

def _vshock_tobs(text, fname):
    """v_shock from the title (v_shock=65) or the file name (v82_5 -> 82.5); T_obs from the title or the
    'constant temperature, t=... K' line."""
    m_v = re.search(r'v[ _]shock\s*=\s*([\d.]+)', text) or re.search(r'^v(\d+(?:_\d+)?)', fname)
    m_t = (re.search(r'T[ _]obs\s*=\s*([\d.]+)', text)
           or re.search(r'constant temperature,?\s*t\s*=\s*([\d.]+)', text, re.I))
    v = float(m_v.group(1).replace('_', '.')) if m_v else None
    t = float(m_t.group(1)) if m_t else None
    return v, t

folder = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
rows = []
for f in sorted(folder.glob('*.out')):
    text = f.read_text(errors='ignore')
    hits = re.findall(r'T\(<nH/Tspin>\)\s+([\d.Ee+-]+)\s+TB21cm:\s*([\d.Ee+-]+)', text)
    v_sh, t_obs = _vshock_tobs(text, f.stem)
    if not hits:
        print(f"no TB21cm line in {f.name}"); continue
    tspin, tb = hits[-1]
    rows.append([f'{v_sh:g}' if v_sh else '', f'{t_obs:g}' if t_obs else '', f"{float(tb):.3g}", f"{float(tspin):.0f}", f.name])
rows.sort(key=lambda r: (float(r[1] or 0), float(r[0] or 0)))
with open(folder / 'tb21.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['v_shock_kms', 'T_obs_K', 'TB21_K', 'Tspin_K', 'file']); w.writerows(rows)
for r in rows:
    print(', '.join(r))
