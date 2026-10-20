import numpy as np
import pytest

from mdtools import compute_rmsd, compute_rmsf

from .conftest import FLEXIBLE, N_FRAMES, N_RESIDUES

SELECTION = "name CA"


def test_fixture_shape(universe):
    assert len(universe.trajectory) == N_FRAMES
    assert len(universe.select_atoms(SELECTION)) == N_RESIDUES


def test_rmsd_starts_at_zero_and_stays_finite(universe):
    result = compute_rmsd(universe, SELECTION)
    assert len(result.rmsd) == N_FRAMES
    assert result.rmsd[0] == pytest.approx(0.0, abs=1e-4)
    assert np.all(np.isfinite(result.rmsd))
    assert np.all(result.rmsd >= 0)


def test_rmsd_of_a_stationary_system_does_not_drift(universe):
    """The fixture fluctuates around a fixed structure, so the first and
    last quarters should agree: no net drift."""
    result = compute_rmsd(universe, SELECTION)
    assert result.plateau == pytest.approx(result.mean, rel=0.35)


def test_plateau_is_the_tail_mean(universe):
    result = compute_rmsd(universe, SELECTION)
    tail = result.rmsd[3 * N_FRAMES // 4:]
    assert result.plateau == pytest.approx(tail.mean())


def test_rmsf_finds_the_flexible_residues(universe):
    """Residues 11-15 were built with ten times the displacement of the
    rest, so they must dominate the RMSF ranking."""
    result = compute_rmsf(universe, SELECTION)
    assert len(result.rmsf) == N_RESIDUES

    flexible_resids = set(range(FLEXIBLE.start + 1, FLEXIBLE.stop + 1))
    top_five = {resid for resid, _, _ in result.most_flexible(5)}
    assert top_five == flexible_resids

    rigid = np.delete(result.rmsf, np.s_[FLEXIBLE])
    assert result.rmsf[FLEXIBLE].min() > rigid.max() * 3


def test_most_flexible_is_sorted(universe):
    result = compute_rmsf(universe, SELECTION)
    values = [v for _, _, v in result.most_flexible(5)]
    assert values == sorted(values, reverse=True)
