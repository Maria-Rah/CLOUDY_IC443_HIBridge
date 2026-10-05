"""
Checks every CLOUDY input file against the bridge recipe (bridge.py).

For each *.in file, v_shock and T_obs are read from the title line
("v_shock=65 km/s, T_obs=10000K") and the hden and stop-column values in the
file are compared with those computed by bridge.cloudy_parameters().

Usage:  python check_cloudy_inputs.py ../cloudy_inputs
"""
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
from bridge import cloudy_parameters

root = Path(sys.argv[1] if len(sys.argv) > 1 else '../cloudy_inputs')
bad = 0
for f in sorted(root.rglob('*.in')):
    text = f.read_text()
    v_sh, t_obs = _vshock_tobs(text, f.stem)
    m_h = re.search(r'^hden\s+([\d.]+)', text, re.M)
    m_c = re.search(r'^stop column density\s+([\d.]+)', text, re.M)
    if not (v_sh and t_obs and m_h and m_c):
        print(f"SKIP  {f}: could not read v_shock, T_obs, hden or stop column"); continue
    lh, lc = cloudy_parameters(v_sh, t_obs)
    ok = abs(lh - float(m_h.group(1))) < 6e-4 and abs(lc - float(m_c.group(1))) < 6e-3
    bad += not ok
    print(f"{'OK  ' if ok else 'DIFF'}  {f.relative_to(root)}: hden {m_h.group(1)} (bridge {lh:.4f}), "
          f"column {m_c.group(1)} (bridge {lc:.2f})")
print("all inputs consistent with the bridge" if bad == 0 else f"{bad} file(s) differ from the bridge")
