"""Structural analysis of membrane-protein MD trajectories.

Three measurements, which together describe how a receptor behaves over a
simulation: global drift from the starting structure (RMSD), per-residue
flexibility (RMSF), and the hydrogen bonds that hold a binding site or an
interface together.

Everything here is built on MDAnalysis, so any trajectory format it reads
(AMBER, GROMACS, NAMD, CHARMM) works without changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from MDAnalysis import Universe
from MDAnalysis.analysis import align, rms
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis


class NoHydrogensError(ValueError):
    """Raised when hydrogen bond analysis is asked of a system with no
    explicit hydrogens, such as a coarse-grained or CA-only trajectory."""


# A membrane protein diffuses and rotates in the bilayer, so raw coordinates
# drift even when the fold is perfectly stable. Every measurement below is
# therefore taken after superposing each frame onto a reference selection.
DEFAULT_ALIGN = "name CA and not (resname POPC POPE POPS CHOL)"


@dataclass(frozen=True)
class RMSDResult:
    time_ps: np.ndarray
    rmsd: np.ndarray

    @property
    def mean(self) -> float:
        return float(self.rmsd.mean())

    @property
    def plateau(self) -> float:
        """Mean RMSD over the last quarter of the trajectory.

        Compared with the overall mean, this says whether the structure has
        settled: a plateau well above the mean means it is still drifting.
        """
        tail = self.rmsd[max(1, 3 * len(self.rmsd) // 4):]
        return float(tail.mean())


@dataclass(frozen=True)
class RMSFResult:
    resids: np.ndarray
    resnames: np.ndarray
    rmsf: np.ndarray

    def most_flexible(self, n: int = 10) -> list[tuple[int, str, float]]:
        order = np.argsort(self.rmsf)[::-1][:n]
        return [(int(self.resids[i]), str(self.resnames[i]),
                 float(self.rmsf[i])) for i in order]


def load(topology: str, trajectory: str | None = None) -> Universe:
    """Open a trajectory. With no trajectory file, a single structure."""
    return Universe(topology, trajectory) if trajectory else Universe(topology)


def compute_rmsd(universe: Universe,
                 selection: str = DEFAULT_ALIGN,
                 reference_frame: int = 0) -> RMSDResult:
    """Backbone RMSD against one frame, after superposition."""
    universe.trajectory[reference_frame]
    reference = universe.copy()

    analysis = rms.RMSD(universe, reference,
                        select=selection, ref_frame=reference_frame).run()
    # columns: frame, time (ps), RMSD (A)
    return RMSDResult(time_ps=analysis.results.rmsd[:, 1],
                      rmsd=analysis.results.rmsd[:, 2])


def compute_rmsf(universe: Universe,
                 selection: str = DEFAULT_ALIGN) -> RMSFResult:
    """Per-residue fluctuation about the average structure.

    The trajectory is first superposed onto its own average, which is the
    reference RMSF is defined against; aligning to frame 0 instead inflates
    the values of every residue by whatever that one frame happened to do.
    """
    average = align.AverageStructure(universe, universe,
                                     select=selection, ref_frame=0).run()
    align.AlignTraj(universe, average.results.universe,
                    select=selection, in_memory=True).run()

    atoms = universe.select_atoms(selection)
    analysis = rms.RMSF(atoms).run()
    return RMSFResult(resids=atoms.resids,
                      resnames=atoms.resnames,
                      rmsf=analysis.results.rmsf)


def compute_hbonds(universe: Universe,
                   donors_sel: str = "protein",
                   acceptors_sel: str = "protein",
                   distance: float = 3.5,
                   angle: float = 150.0):
    """Hydrogen bonds between two selections, frame by frame.

    The 3.5 A / 150 degree cutoff is the common geometric criterion. Pass a
    ligand selection as `donors_sel` or `acceptors_sel` to follow a specific
    protein-ligand contact instead of the whole protein.
    """
    if len(universe.select_atoms("name H*")) == 0:
        raise NoHydrogensError(
            "the system contains no hydrogen atoms, so hydrogen bonds "
            "cannot be measured; use a trajectory with explicit hydrogens"
        )

    analysis = HydrogenBondAnalysis(
        universe,
        donors_sel=None,
        hydrogens_sel=None,
        acceptors_sel=None,
        d_a_cutoff=distance,
        d_h_a_angle_cutoff=angle,
        between=[donors_sel, acceptors_sel],
    )
    analysis.run()
    return analysis


def hbond_occupancy(analysis, n_frames: int, top: int = 15):
    """Rank hydrogen bonds by the fraction of frames they are present in.

    Occupancy is what separates a structural contact from a transient one:
    a bond present in 5% of frames is noise, one present in 90% is holding
    something together.
    """
    counts = analysis.count_by_ids()
    rows = []
    for donor_ix, hydrogen_ix, acceptor_ix, count in counts:
        donor = analysis.u.atoms[int(donor_ix)]
        acceptor = analysis.u.atoms[int(acceptor_ix)]
        rows.append({
            "donor": f"{donor.resname}{donor.resid}:{donor.name}",
            "acceptor": f"{acceptor.resname}{acceptor.resid}:{acceptor.name}",
            "frames": int(count),
            "occupancy": count / n_frames,
        })
    rows.sort(key=lambda r: r["occupancy"], reverse=True)
    return rows[:top]
