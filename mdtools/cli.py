"""Command line entry point.

    python -m mdtools --topology system.pdb --trajectory run.xtc \
        --out results/
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .analysis import (
    DEFAULT_ALIGN,
    NoHydrogensError,
    compute_hbonds,
    compute_rmsd,
    compute_rmsf,
    hbond_occupancy,
    load,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mdtools",
        description="RMSD, RMSF and hydrogen bond analysis of an MD run.",
    )
    parser.add_argument("--topology", required=True,
                        help="PDB, GRO, PSF or PRMTOP file")
    parser.add_argument("--trajectory",
                        help="XTC, DCD, TRR or NetCDF file")
    parser.add_argument("--selection", default=DEFAULT_ALIGN,
                        help="atoms to align and measure (MDAnalysis syntax)")
    parser.add_argument("--ligand",
                        help="ligand selection, e.g. 'resname LIG'; when "
                             "given, hydrogen bonds are measured between the "
                             "ligand and the protein instead of within it")
    parser.add_argument("--out", default="md_analysis",
                        help="output directory (default: md_analysis)")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    universe = load(args.topology, args.trajectory)
    n_frames = len(universe.trajectory)
    n_atoms = len(universe.select_atoms(args.selection))
    print(f"{n_frames} frame(s), {n_atoms} atom(s) in the selection")
    if n_atoms == 0:
        raise SystemExit(f"selection matched nothing: {args.selection!r}")

    summary: dict[str, object] = {
        "n_frames": n_frames,
        "n_selected_atoms": n_atoms,
        "selection": args.selection,
    }

    if n_frames > 1:
        rmsd = compute_rmsd(universe, args.selection)
        _write_csv(out / "rmsd.csv", ["time_ps", "rmsd_A"],
                   zip(rmsd.time_ps, rmsd.rmsd))
        summary["rmsd_mean_A"] = round(rmsd.mean, 3)
        summary["rmsd_plateau_A"] = round(rmsd.plateau, 3)
        print(f"RMSD  mean {rmsd.mean:.2f} A, "
              f"last quarter {rmsd.plateau:.2f} A")

        rmsf = compute_rmsf(universe, args.selection)
        _write_csv(out / "rmsf.csv", ["resid", "resname", "rmsf_A"],
                   zip(rmsf.resids, rmsf.resnames, rmsf.rmsf))
        summary["rmsf_most_flexible"] = [
            {"resid": r, "resname": n, "rmsf_A": round(v, 3)}
            for r, n, v in rmsf.most_flexible()
        ]
        print("RMSF  most flexible residues: " + ", ".join(
            f"{n}{r} ({v:.1f} A)" for r, n, v in rmsf.most_flexible(5)))

        if not args.no_plots:
            _plot(rmsd, rmsf, out)
            print(f"plots written to {out}")
    else:
        print("single frame: RMSD and RMSF skipped")

    donors = args.ligand or "protein"
    try:
        hbonds = compute_hbonds(universe, donors_sel=donors,
                                acceptors_sel="protein")
    except NoHydrogensError as exc:
        print(f"H-bonds  skipped: {exc}")
        rows = []
    else:
        rows = hbond_occupancy(hbonds, n_frames)
    if rows:
        _write_csv(out / "hbonds.csv",
                   ["donor", "acceptor", "frames", "occupancy"],
                   ((r["donor"], r["acceptor"], r["frames"],
                     round(r["occupancy"], 4)) for r in rows))
        summary["hbonds_top"] = rows[:5]
        print(f"H-bonds  {len(rows)} contact(s) ranked by occupancy")
        for row in rows[:5]:
            print(f"         {row['donor']:>14} ... {row['acceptor']:<14} "
                  f"{row['occupancy']:6.1%}")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary written to {out / 'summary.json'}")
    return 0


def _write_csv(path: Path, header: list[str], rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _plot(rmsd, rmsf, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 6))
    top.plot(rmsd.time_ps / 1000, rmsd.rmsd, linewidth=1)
    top.set_xlabel("time (ns)")
    top.set_ylabel("RMSD (Å)")
    top.set_title("Backbone RMSD")

    bottom.plot(rmsf.resids, rmsf.rmsf, linewidth=1)
    bottom.set_xlabel("residue")
    bottom.set_ylabel("RMSF (Å)")
    bottom.set_title("Per-residue fluctuation")

    fig.tight_layout()
    fig.savefig(out / "rmsd_rmsf.png", dpi=200)


if __name__ == "__main__":
    raise SystemExit(main())
