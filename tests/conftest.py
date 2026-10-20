"""A small synthetic trajectory, built in memory.

Using a generated system rather than a downloaded one keeps the test suite
self-contained and fast, and lets each test know the right answer in
advance: the noise added to each residue is set here, so RMSF has a value
to be checked against.
"""

import numpy as np
import pytest
from MDAnalysis import Universe

N_RESIDUES = 20
N_FRAMES = 50
SEED = 20261020

# Residues 10-14 are given ten times the displacement of the rest, so a
# correct RMSF must single them out.
FLEXIBLE = slice(10, 15)
BASE_NOISE = 0.05
FLEXIBLE_NOISE = 0.5


@pytest.fixture
def universe() -> Universe:
    rng = np.random.default_rng(SEED)

    # One CA atom per residue on a straight line, 3.8 A apart.
    start = np.zeros((N_RESIDUES, 3), dtype=np.float32)
    start[:, 0] = np.arange(N_RESIDUES) * 3.8

    noise = np.full(N_RESIDUES, BASE_NOISE)
    noise[FLEXIBLE] = FLEXIBLE_NOISE

    frames = np.empty((N_FRAMES, N_RESIDUES, 3), dtype=np.float32)
    for frame in range(N_FRAMES):
        frames[frame] = start + rng.normal(0, noise[:, None], (N_RESIDUES, 3))

    u = Universe.empty(
        n_atoms=N_RESIDUES,
        n_residues=N_RESIDUES,
        atom_resindex=np.arange(N_RESIDUES),
        residue_segindex=np.zeros(N_RESIDUES, dtype=int),
        trajectory=True,
    )
    u.add_TopologyAttr("name", ["CA"] * N_RESIDUES)
    u.add_TopologyAttr("type", ["C"] * N_RESIDUES)
    u.add_TopologyAttr("resname", ["ALA"] * N_RESIDUES)
    u.add_TopologyAttr("resid", list(range(1, N_RESIDUES + 1)))
    u.add_TopologyAttr("mass", [12.011] * N_RESIDUES)
    u.load_new(frames, order="fac", dt=10.0)  # 10 ps per frame
    return u
