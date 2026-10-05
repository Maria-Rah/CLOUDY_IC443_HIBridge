# CLOUDY_IC443_HIBridge

Observational analysis, CLOUDY input files, PLUTO-CLOUDY bridge, and wing-model scripts for the paper

> M. Rah, M. Alizadeh, & A. G. Yeghikyan, *Independent Kinematic Confirmation of the IC443
> Northeast Shock Velocity from HI 21 cm Observations*, submitted to The Astrophysical Journal (2026).

The companion repository [PLUTO_IC443_ShockValidation](https://github.com/Maria-Rah/PLUTO_IC443_ShockValidation)
contains the PLUTO shock-tube runs and the kinematic analysis.

## What is here

| Path | Content | Paper |
|---|---|---|
| `data/observed_spectrum.dat` | HI 21 cm spectrum at position A, l = 189.291 deg, b = 3.025 deg (5x5 pixel average; columns: v_LSR, T_B raw, T_B 5-channel smoothed) | Section 2.3, Figures 2 and 5 |
| `data/excess_map.npy`, `data/excess_map_origin.json` | HI excess: mean T_B over v_LSR = -55 to -25 km/s, 5x5 smoothed, 240 x 240 pixels around the centre of IC 443 | Section 2.2, Figure 1a |
| `data/edge_map_3sigma.npy` | Velocity of the HI wing edge (3 sigma) in 5x5-average spectra | Section 2.3, Figure 1b |
| `scripts/ic443_hi_diagnostic.py` | From the CGPS cubes: excess map, spectra, and wing edges (2.5, 3, 4 sigma) at the excess maxima of each quadrant | Section 2.2 |
| `scripts/ic443_fastest_hi.py` | Full-resolution wing-edge map and the positions of the fastest HI, with threshold and one-pixel stability | Section 2.3 |
| `scripts/bridge.py` | The bridge n_H = P2/(k_B T_obs) and the slab column for L = 0.6 pc | Section 3.2, Equation 2 |
| `scripts/check_cloudy_inputs.py` | Checks every CLOUDY input against `bridge.py` | |
| `scripts/extract_tb21.py` | Reads T_B(21 cm) and the spin temperature from CLOUDY outputs | Table 1, Figures 4 and 7 |
| `cloudy_inputs/grid_Tobs10000` | The 20-point grid (v_shock = 20 to 87.9 km/s, T_obs = 10^4 K): inputs and outputs | Section 3.2, Table 1, Figure 4 |
| `cloudy_inputs/Tobs_sensitivity` | v_shock = 65, 80, 100 km/s at T_obs = 8000, 10000, 12000 K | Section 5.2, Figure 7 |
| `cloudy_inputs/cosmic_ray_test` | v_shock = 65 km/s, T_obs = 10^4 K, cosmic-ray background x10 | Section 5.3 |
| `scripts/make_figure1.py` ... `make_figure7.py` | Figures 1, 2, 4, 5, 7 | |
| `scripts/reproduce_numbers.py` | Recomputes every number quoted in the paper (Sections 2-5, Tables 1-2) from the files in this repository and prints it next to the value in the text | all |
| `figures/` | Figures as used in the paper | |

## CLOUDY models

CLOUDY C23.01 (Chatzikos et al. 2023, RMxAA, 59, 327). Each model is a constant-temperature slab;
the input specifies only `hden`, `constant temperature`, and `stop column density` (plus the
cosmic-ray command in the test), with no incident radiation field, cosmic-ray background, or grains.
The 21 cm brightness temperature is the `TB21cm` value of the last iteration in the `.out` file.

## Reproducing the results

```bash
pip install -r requirements.txt
cd scripts
python bridge.py                                         # hden and column of the grid
python check_cloudy_inputs.py ../cloudy_inputs           # inputs vs. bridge
python extract_tb21.py ../cloudy_inputs/grid_Tobs10000   # T_B(21 cm) of the grid
python make_figure2.py; python make_figure4.py; python make_figure5.py; python make_figure7.py
python reproduce_numbers.py                              # all numbers of the paper
python ic443_hi_diagnostic.py <HI cube> <1420 MHz image>   # maps and spectra (needs astropy)
python ic443_fastest_hi.py   <HI cube> <1420 MHz image>   # wing-edge map, fastest HI
python make_figure1.py    # needs the CGPS 1420 MHz FITS file in data/ (see data/README.md)
```

To rerun a CLOUDY model: `export CLOUDY_DATA_PATH=<cloudy>/data` and `<cloudy>/source/cloudy.exe -r <name>`.

## Data

The HI 21 cm and 1420 MHz continuum data are from the Canadian Galactic Plane Survey
(Taylor et al. 2003, AJ, 125, 3145), mosaic MER2, available from the Canadian Astronomy Data
Centre. See `data/README.md`.

## License and citation

Code: MIT license (see `LICENSE`). If you use this material, please cite the paper above
(see `CITATION.cff`).

Note: the title lines of the CLOUDY inputs use the working name "IC443 NE diffuse shock". The spectrum analyzed in the paper is at position A, on the southeastern side of the remnant (paper Section 2.2); the models themselves do not depend on position.
