"""
Generate SciBuddy-shaped synthetic task packages with varied data.

One package per variant, each with its own temperature series so the expected
answer differs. Variants cover all three trend classes, which keeps a passing
score from being reachable by guessing one constant.

Usage:
    python examples/scibuddy_rollout/synthetic_data/make_variants.py --count 8
"""

from __future__ import annotations

import argparse
import random
import shutil
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Slope per day and noise amplitude per trend class. `flat` stays inside the
# instruction's +/-0.01 dead band so its expected label is genuinely "flat".
_TRENDS = {
    "warming": (0.045, 0.6),
    "cooling": (-0.038, 0.5),
    "flat": (0.002, 0.4),
}


def _series(seed: int, slope: float, noise: float, days: int, base: float) -> str:
    """Build a CSV series whose OLS slope lands in the intended trend class."""
    rng = random.Random(seed)
    rows = ["day,temp_c"]
    for day in range(1, days + 1):
        temp = base + slope * day + rng.uniform(-noise, noise)
        rows.append(f"{day},{temp:.3f}")
    return "\n".join(rows) + "\n"


def build_variant(template: Path, out_dir: Path, name: str, series_text: str) -> Path:
    """Write one package tarball that differs from the template only in its data.

    Args:
        template: Directory holding the reference `package/` tree.
        out_dir: Where the tarball is written.
        name: Variant slug, used for the task name and the file name.
        series_text: CSV contents for this variant.

    Returns:
        Path to the written tarball.
    """
    staging = out_dir / f"_staging_{name}"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(template / "package", staging / "package")

    # Each Docker context needs its own copy, because COPY cannot reach outside
    # the context directory. Both must move together or the grader disagrees.
    for rel in ("environment/series.csv", "tests/series.csv"):
        (staging / "package" / rel).write_text(series_text)

    toml = staging / "package" / "task.toml"
    toml.write_text(toml.read_text().replace("synthetic-tempstats-smoke", name))

    tarball = out_dir / f"job-{name}-package.tar.gz"
    if tarball.exists():
        tarball.unlink()
    with tarfile.open(tarball, "w:gz") as tf:
        tf.add(staging / "package", arcname="package")
    shutil.rmtree(staging)
    return tarball


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=8, help="Number of variants.")
    parser.add_argument(
        "--template",
        type=Path,
        default=HERE / "template",
        help="Directory holding the reference package/ tree.",
    )
    parser.add_argument("--out", type=Path, default=HERE / "variants")
    args = parser.parse_args()

    if not (args.template / "package" / "task.toml").exists():
        raise SystemExit(f"No package/task.toml under {args.template}.")

    args.out.mkdir(parents=True, exist_ok=True)
    trends = list(_TRENDS)
    for i in range(args.count):
        trend = trends[i % len(trends)]
        slope, noise = _TRENDS[trend]
        name = f"synthetic-tempstats-{trend}-{i:02d}"
        text = _series(
            seed=20260926 + i * 101,
            slope=slope,
            noise=noise,
            days=40 + (i * 7) % 45,
            base=6.0 + (i % 5) * 1.5,
        )
        path = build_variant(args.template, args.out, name, text)
        print(f"  {path.name}  trend={trend}")

    print(f"\nWrote {args.count} package(s) to {args.out}")


if __name__ == "__main__":
    main()
