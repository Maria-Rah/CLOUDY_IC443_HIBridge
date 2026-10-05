# Data

* `observed_spectrum.dat`: HI 21 cm spectrum extracted from the CGPS MER2 HI cube at position A,
  l = 189.291 deg, b = 3.025 deg (maximum of the 5x5-smoothed excess map), averaged over 5x5 pixels.
  Columns: v_LSR (km/s), T_B raw (K), T_B smoothed with a five-channel boxcar (K).
  Channels 1-18 and 272 (flagged in the survey metadata) are excluded.
* `excess_map.npy`: mean T_B over v_LSR = -55 to -25 km/s, smoothed 5x5 pixels, 240 x 240 pixels
  centred on IC 443 (Green's catalogue, 06h17m00s +22d34m); `excess_map_origin.json` gives the
  mosaic pixel of element [0, 0].
* `edge_map_3sigma.npy`: velocity of the HI wing edge (3 sigma, ten consecutive channels) in
  5x5-average spectra, same grid as the excess map (NaN outside 1.1 R).
* The CGPS FITS files are not redistributed here. Download mosaic MER2
  (`CGPS_MER2_1420_MHz_I_image.fits`, HI line cube) from the Canadian Astronomy Data Centre
  and place the continuum image in this folder to run `make_figure1.py`; the two diagnostic
  scripts read the HI cube and the continuum image directly.

The research presented here has used data from the Canadian Galactic Plane Survey, a Canadian
project with international partners, supported by the Natural Sciences and Engineering Research Council.
