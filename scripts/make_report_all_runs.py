#!/usr/bin/env python3

import argparse
import glob
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Batch report Ollama runs.")
    parser.add_argument(
        "--runs_dir",
        default="results/metrics",
        help="Directory containing .csv run files",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern to match run files",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing",
    )

    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"❌ Runs directory does not exist: {runs_dir}")
        sys.exit(1)

    files = sorted(glob.glob(str(runs_dir / args.pattern)))

    if not files:
        print("⚠️ No matching run files found.")
        return

    print(f"Found {len(files)} run files.\n")

    for f in files:
        cmd = [
            sys.executable,
            "scripts/make_report.py",
            "--metrics",
            f,
        ]

        print("Running:", " ".join(cmd))

        if not args.dry_run:
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"❌ Error scoring: {f}")
                sys.exit(result.returncode)

    print("\n✅ Reports for all RUNS successfully recorded.")


if __name__ == "__main__":
    main()
