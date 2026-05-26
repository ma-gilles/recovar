#!/usr/bin/env python3
"""Launch a parity replay job and watch its dump dir for per-iter perf checks.

Forks ``scripts/run_multi_iter_parity.py`` (or any subprocess via ``--cmd``)
in the background, polls ``--dump-dir`` every ``--poll-interval`` seconds,
and runs ``scripts/parity/check_perf.py --single-iter N`` for each new
``iter_NNN.npz`` that lands. Streams everything to stdout so it shows up in
the slurm log.

Exit code:
    Mirrors the inner subprocess by default. With ``--cancel-on-regression``,
    additionally calls ``scancel $SLURM_JOB_ID`` (when set) and SIGKILLs the
    inner process if any iter is flagged ``REGRESSED``.

Example:
    pixi run python scripts/parity/launch_with_perf_check.py \
        --dump-dir _agent_scratch/parity/recovar_HEAD \
        --baseline tests/baselines/parity/perf_baseline_5k_128_a100.json \
        --cmd "pixi run python scripts/run_multi_iter_parity.py \
                  --relion_dir /scratch/.../relion_ref_os0 \
                  --data_star /scratch/.../particles.star \
                  --iter 3 --max_iter 14 --output_dir /scratch/.../recovar_HEAD"
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _list_iter_npz(dump_dir: Path) -> set[int]:
    """Return the set of iter indices that currently have an iter_NNN.npz file."""
    if not dump_dir.exists():
        return set()
    out: set[int] = set()
    for p in dump_dir.glob("iter_*.npz"):
        try:
            out.add(int(p.stem.split("_", 1)[1]))
        except (ValueError, IndexError):
            continue
    return out


def _run_check(check_script: Path, dump_dir: Path, baseline: Path, iter_num: int) -> str:
    """Run check_perf.py for a single iter and return its stdout."""
    cmd = [
        sys.executable,
        str(check_script),
        "--dump-dir",
        str(dump_dir),
        "--baseline",
        str(baseline),
        "--single-iter",
        str(iter_num),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout.rstrip("\n")
    if proc.stderr:
        out = (out + "\n" if out else "") + "[check_perf.py stderr] " + proc.stderr.rstrip("\n")
    return out


def _kill_inner(inner: subprocess.Popen) -> None:
    if inner.poll() is None:
        try:
            inner.send_signal(signal.SIGTERM)
            time.sleep(2)
            if inner.poll() is None:
                inner.send_signal(signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--dump-dir",
        required=True,
        type=Path,
        help="Where parity_dump writes iter_NNN.npz files (== RECOVAR_PARITY_DUMP_DIR).",
    )
    ap.add_argument("--baseline", required=True, type=Path, help="Path to the perf baseline JSON.")
    ap.add_argument("--cmd", required=True, help="Shell command to launch the parity job (will be parsed with shlex).")
    ap.add_argument("--poll-interval", type=float, default=30.0, help="Seconds between dump-dir polls (default 30).")
    ap.add_argument(
        "--check-script",
        type=Path,
        default=Path(__file__).resolve().parent / "check_perf.py",
        help="Path to check_perf.py (default: alongside this script).",
    )
    ap.add_argument(
        "--cancel-on-regression",
        action="store_true",
        help="If set, kill the inner process and (if SLURM_JOB_ID set) scancel on REGRESSED iter.",
    )
    args = ap.parse_args()

    args.dump_dir.mkdir(parents=True, exist_ok=True)

    # Make sure the inner process inherits a parity-dump-active env.
    env = os.environ.copy()
    env["RECOVAR_PARITY_DUMP_DIR"] = str(args.dump_dir)

    print(f"[{_now()}] launcher: dump_dir={args.dump_dir}", flush=True)
    print(f"[{_now()}] launcher: baseline={args.baseline}", flush=True)
    print(f"[{_now()}] launcher: cmd={args.cmd}", flush=True)
    print(f"[{_now()}] launcher: poll_interval={args.poll_interval}s", flush=True)

    inner = subprocess.Popen(
        shlex.split(args.cmd),
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=env,
    )

    seen: set[int] = _list_iter_npz(args.dump_dir)
    print(f"[{_now()}] launcher: pre-existing iters in dump dir: {sorted(seen)}", flush=True)

    regressed = False
    try:
        while inner.poll() is None:
            time.sleep(args.poll_interval)
            current = _list_iter_npz(args.dump_dir)
            new_iters = sorted(current - seen)
            for iter_num in new_iters:
                # Small wait for the npz to be fully written.
                time.sleep(0.5)
                line = _run_check(args.check_script, args.dump_dir, args.baseline, iter_num)
                print(f"[{_now()}] perf-check: {line}", flush=True)
                if "REGRESSED" in line:
                    regressed = True
                    if args.cancel_on_regression:
                        print(f"[{_now()}] launcher: REGRESSION → cancelling inner job", flush=True)
                        _kill_inner(inner)
                        slurm_jid = os.environ.get("SLURM_JOB_ID")
                        if slurm_jid:
                            print(f"[{_now()}] launcher: scancel {slurm_jid}", flush=True)
                            subprocess.run(["scancel", slurm_jid])
                        break
            seen = current
            if regressed and args.cancel_on_regression:
                break

        # Final sweep after the inner exits.
        current = _list_iter_npz(args.dump_dir)
        for iter_num in sorted(current - seen):
            line = _run_check(args.check_script, args.dump_dir, args.baseline, iter_num)
            print(f"[{_now()}] perf-check: {line}", flush=True)
            if "REGRESSED" in line:
                regressed = True

    except KeyboardInterrupt:
        print(f"[{_now()}] launcher: interrupted; killing inner", flush=True)
        _kill_inner(inner)
        return 130

    rc = inner.wait()
    print(f"[{_now()}] launcher: inner exited rc={rc} regressed={regressed}", flush=True)
    if args.cancel_on_regression and regressed:
        return 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
