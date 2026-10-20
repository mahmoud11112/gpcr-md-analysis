"""Write a small demo trajectory so the CLI can be run without any
simulation data to hand.

    python examples/make_demo_trajectory.py
    python -m mdtools --topology examples/demo.pdb \
        --trajectory examples/demo.xtc --selection "name CA" --out demo_out

The system is a 60-residue helix-like chain in which one stretch is made
deliberately mobile, so RMSF has something to find. It is not a real
receptor; real runs take the same command with your own topology and
trajectory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import MDAnalysis as mda
from MDAnalysis import Universe

HERE = Path(__file__).parent
N_RESIDUES = 60
N_FRAMES = 200
SEED = 20261020
LOOP = slice(28, 38)  # the mobile stretch


def main() -> None:
    rng = np.random.default_rng(SEED)

    # An alpha-helical backbone trace: 1.5 A rise and 100 degrees of
    # rotation per residue, on a 2.3 A radius.
    index = np.arange(N_RESIDUES)
    angle = np.deg2rad(100.0) * index
    start = np.column_stack([
        2.3 * np.cos(angle),
        2.3 * np.sin(angle),
        1.5 * index,
    ]).astype(np.float32)

    amplitude = np.full(N_RESIDUES, 0.25)
    amplitude[LOOP] = 1.6

    frames = np.empty((N_FRAMES, N_RESIDUES, 3), dtype=np.float32)
    for frame in range(N_FRAMES):
        jitter = rng.normal(0, amplitude[:, None], (N_RESIDUES, 3))
        # A slow collective breathing motion on top of the per-frame noise.
        breathing = 0.4 * np.sin(2 * np.pi * frame / N_FRAMES)
        frames[frame] = start * (1 + 0.01 * breathing) + jitter

    u = Universe.empty(
        n_atoms=N_RESIDUES,
        n_residues=N_RESIDUES,
        atom_resindex=index,
        residue_segindex=np.zeros(N_RESIDUES, dtype=int),
        trajectory=True,
    )
    u.add_TopologyAttr("name", ["CA"] * N_RESIDUES)
    u.add_TopologyAttr("type", ["C"] * N_RESIDUES)
    u.add_TopologyAttr("element", ["C"] * N_RESIDUES)
    u.add_TopologyAttr("resname", ["ALA"] * N_RESIDUES)
    u.add_TopologyAttr("resid", list(range(1, N_RESIDUES + 1)))
    u.add_TopologyAttr("segid", ["A"])
    u.add_TopologyAttr("mass", [12.011] * N_RESIDUES)
    u.load_new(frames, order="fac", dt=20.0)  # 20 ps per frame, 4 ns total

    u.atoms.write(HERE / "demo.pdb", frames=u.trajectory[:1])
    with mda.Writer(str(HERE / "demo.xtc"), n_atoms=len(u.atoms)) as w:
        for _ in u.trajectory:
            w.write(u.atoms)

    print(f"wrote demo.pdb and demo.xtc ({N_FRAMES} frames, "
          f"{N_RESIDUES} residues) to {HERE}")
    print(f"mobile residues: {LOOP.start + 1}-{LOOP.stop}")


if __name__ == "__main__":
    main()
