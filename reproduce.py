"""Single public entry point for the manuscript's data-driven results."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_data() -> None:
    metadata = pd.read_csv(ROOT / "data" / "metadata.csv")
    if metadata["case_id"].duplicated().any():
        raise RuntimeError("metadata.csv contains duplicate case IDs")
    if set(metadata["dataset_role"]) != {"development", "external"}:
        raise RuntimeError("Unexpected dataset roles")
    if int(metadata["dataset_role"].eq("development").sum()) != 32:
        raise RuntimeError("Expected 32 development cases")
    if int(metadata["dataset_role"].eq("external").sum()) != 1:
        raise RuntimeError("Expected one external case")

    required = {
        "case_id",
        "time_min",
        "f_liquid",
        "Tin_C",
        "Tout_C",
        "Q_cum_J",
        "log1p_Q_J",
        "Ri_m",
        "L_m",
        "Phi",
    }
    for row in metadata.itertuples(index=False):
        path = ROOT / "data" / row.file
        if sha256(path) != row.sha256:
            raise RuntimeError(f"Data checksum mismatch: {row.file}")
        frame = pd.read_csv(path)
        if len(frame) != int(row.row_count):
            raise RuntimeError(f"Row-count mismatch: {row.file}")
        if not required.issubset(frame.columns):
            raise RuntimeError(f"Schema mismatch: {row.file}")
        if not frame["case_id"].astype(str).eq(str(row.case_id)).all():
            raise RuntimeError(f"Case identity mismatch: {row.file}")
        intervals = frame["time_min"].diff().dropna()
        if not intervals.round(10).eq(0.5).all():
            raise RuntimeError(f"Sampling interval mismatch: {row.file}")
    print("PASS: 32 development cases and one locked external case")


def quick_verify(make_figure_outputs: bool = False) -> None:
    from evaluate import run as evaluate_main
    from evaluate_robustness import run as evaluate_robustness

    verify_data()
    output = ROOT / "outputs" / "reproduced"
    evaluate_main(output, compare=True)
    evaluate_robustness(output / "robustness_metrics.csv", compare=True)
    if make_figure_outputs:
        from make_figures import run as make_figures

        make_figures(ROOT / "outputs" / "figures")
    summary = {
        "status": "PASS",
        "verified": ["Tables 6-7", "Tables 8-9", "Tables 11-12"],
        "excluded": ["Table 10", "Appendix A2 retraining"],
        "note": "Appendix A2 is available through --mode multiseed.",
    }
    (output / "verification_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("PASS: quick frozen-version reproduction complete")


def run_subprocess(script: str, extra: list[str]) -> None:
    subprocess.run([sys.executable, str(SRC / script), *extra], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["verify", "figures", "multiseed", "retrain", "all"],
        default="verify",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.mode == "verify":
        quick_verify(False)
    elif args.mode == "figures":
        run_subprocess("make_figures.py", [])
    elif args.mode == "multiseed":
        extra = ["--smoke"] if args.smoke else []
        run_subprocess("evaluate_multiseed.py", extra)
    elif args.mode == "retrain":
        extra = ["--smoke"] if args.smoke else []
        run_subprocess("train.py", extra)
    elif args.mode == "all":
        quick_verify(True)


if __name__ == "__main__":
    main()
