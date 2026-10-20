# gpcr-md-analysis

RMSD, per-residue RMSF and hydrogen bond occupancy for membrane-protein MD
trajectories, from the command line. Built on
[MDAnalysis](https://www.mdanalysis.org/), so GROMACS, AMBER, NAMD and
CHARMM output all work without conversion.

Written while learning membrane-protein simulation: the measurements are
standard, but the handling around them is the part that is easy to get
wrong, so the reasoning is written into the code rather than left implicit.

## Install

```bash
git clone https://github.com/mahmoud11112/gpcr-md-analysis.git
cd gpcr-md-analysis
pip install -e .
```

Python 3.9 or newer.

## Use

```bash
python -m mdtools \
    --topology   system.gro \
    --trajectory production.xtc \
    --selection  "name CA and protein" \
    --ligand     "resname LIG" \
    --out        results/
```

Output in `results/`: `rmsd.csv`, `rmsf.csv`, `hbonds.csv`, `summary.json`
and a two-panel `rmsd_rmsf.png`.

```
200 frame(s), 60 atom(s) in the selection
RMSD  mean 1.79 A, last quarter 1.77 A
RMSF  most flexible residues: ALA37 (2.8 A), ALA30 (2.8 A), ALA32 (2.8 A)
H-bonds  14 contact(s) ranked by occupancy
              LYS233:NZ ... ASP114:OD1      94.5%
```

No trajectory to hand? Generate a demo one:

```bash
python examples/make_demo_trajectory.py
python -m mdtools --topology examples/demo.pdb \
    --trajectory examples/demo.xtc --selection "name CA" --out demo_out
```

The demo chain has a deliberately mobile stretch at residues 29–38, which
the RMSF output recovers.

From Python:

```python
from mdtools import load, compute_rmsd, compute_rmsf

u = load("system.gro", "production.xtc")
rmsd = compute_rmsd(u, "name CA and protein")
print(rmsd.mean, rmsd.plateau)

rmsf = compute_rmsf(u, "name CA and protein")
for resid, resname, value in rmsf.most_flexible(10):
    print(resid, resname, round(value, 2))
```

## Three things this handles deliberately

**Alignment before measurement.** A protein in a bilayer diffuses and
rotates over a run, so raw coordinates move even when the fold is rigid.
Every frame is superposed on a reference selection first, and the default
selection excludes lipid residues so that bilayer motion does not pull the
alignment around.

**RMSF against the average structure, not frame 0.** RMSF is defined as
fluctuation about the mean position. Aligning to the first frame instead
adds whatever that one frame happened to be doing to every residue's value.
The trajectory is therefore superposed onto its own average structure
first.

**RMSD plateau as well as RMSD mean.** `plateau` is the mean over the last
quarter of the run. Read next to the overall mean, it answers the question
the mean alone cannot: has the structure settled, or is it still drifting?

Hydrogen bonds use the usual 3.5 Å / 150° geometric criterion and are
ranked by occupancy, the fraction of frames in which the bond is present.
Occupancy is what separates a contact that holds something together from
one that forms once and goes. Systems without explicit hydrogens are
reported as such rather than silently returning nothing.

## Tests

```bash
python -m pytest tests -q
```

The suite builds a small trajectory in memory with a known answer — one
stretch of residues is given ten times the displacement of the rest — so
the tests check that RMSF actually finds it, and no data files or downloads
are needed.

## License

GPL-3.0. See `LICENSE`.
