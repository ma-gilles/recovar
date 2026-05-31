"""Interactive wizard for configuring and launching RECOVAR pipeline runs.

Works over SSH — uses only Python builtins (no curses/textual dependency).
Launched via ``recovar quickstart``.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

# ── Formatting helpers ──────────────────────────────────────────────────────

_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"


def _heading(text):
    w = min(shutil.get_terminal_size((80, 24)).columns, 80)
    print()
    print(f"{_BOLD}{_CYAN}{'─' * w}{_RESET}")
    print(f"{_BOLD}{_CYAN}  {text}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'─' * w}{_RESET}")


def _info(text):
    print(f"  {_DIM}{text}{_RESET}")


def _success(text):
    print(f"  {_GREEN}{text}{_RESET}")


def _warn(text):
    print(f"  {_YELLOW}{text}{_RESET}")


def _error(text):
    print(f"  {_RED}{text}{_RESET}")


def _prompt(label, default=None, required=True):
    """Prompt for a single value.  Returns string or None."""
    suffix = f" [{default}]" if default else ""
    while True:
        try:
            answer = input(f"  {_BOLD}{label}{_RESET}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if not answer:
            if default is not None:
                return default
            if not required:
                return None
            _warn("  This field is required.")
            continue
        return answer


def _prompt_choice(label, options, default=None):
    """Prompt user to pick from a numbered list."""
    print(f"  {_BOLD}{label}{_RESET}")
    for i, (val, desc) in enumerate(options, 1):
        marker = " (default)" if val == default else ""
        print(f"    {_CYAN}{i}{_RESET}) {val}  {_DIM}— {desc}{marker}{_RESET}")
    while True:
        raw = _prompt("Choice", default=str(next((i for i, (v, _) in enumerate(options, 1) if v == default), 1)))
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
        except ValueError:
            # Allow typing the value directly
            for v, _ in options:
                if raw.lower() == v.lower():
                    return v
        _warn(f"  Enter a number 1–{len(options)}")


def _prompt_yesno(label, default=True):
    hint = "Y/n" if default else "y/N"
    answer = _prompt(f"{label} [{hint}]", default="y" if default else "n", required=False)
    if answer is None:
        return default
    return answer.lower().startswith("y")


def _find_files(patterns, directory="."):
    """Find files matching any of the glob patterns."""
    results = []
    for pat in patterns:
        results.extend(glob.glob(os.path.join(directory, pat)))
        results.extend(glob.glob(os.path.join(directory, "**", pat), recursive=True))
    # Deduplicate and sort
    seen = set()
    unique = []
    for f in sorted(results):
        real = os.path.realpath(f)
        if real not in seen:
            seen.add(real)
            unique.append(f)
    return unique[:20]  # Cap at 20 to keep display manageable


def _pick_file(label, patterns, required=True, allow_special=None):
    """Let user pick a file from auto-detected candidates or type a path."""
    candidates = _find_files(patterns)
    if candidates:
        print(f"  {_DIM}Found matching files:{_RESET}")
        for i, f in enumerate(candidates, 1):
            print(f"    {_CYAN}{i}{_RESET}) {f}")
        if allow_special:
            for i, (val, desc) in enumerate(allow_special, len(candidates) + 1):
                print(f"    {_CYAN}{i}{_RESET}) {val}  {_DIM}— {desc}{_RESET}")
        raw = _prompt(f"{label} (number or path)", required=required)
        if raw is None:
            return None
        try:
            idx = int(raw)
            all_opts = candidates + ([v for v, _ in allow_special] if allow_special else [])
            if 1 <= idx <= len(all_opts):
                return all_opts[idx - 1]
        except ValueError:
            pass
        return raw
    else:
        if allow_special:
            print(f"  {_DIM}No files found matching {patterns}. Special options:{_RESET}")
            for i, (val, desc) in enumerate(allow_special, 1):
                print(f"    {_CYAN}{i}{_RESET}) {val}  {_DIM}— {desc}{_RESET}")
        return _prompt(label, required=required)


# ── Main wizard ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Interactive wizard for configuring RECOVAR pipeline runs.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command but don't execute it")
    args = parser.parse_args()

    print()
    print(f"{_BOLD}╔═══════════════════════════════════════════════════╗{_RESET}")
    print(f"{_BOLD}║       RECOVAR Pipeline — Interactive Setup        ║{_RESET}")
    print(f"{_BOLD}╚═══════════════════════════════════════════════════╝{_RESET}")
    print()
    _info("This wizard helps you configure and launch a RECOVAR pipeline run.")
    _info("Press Ctrl-C at any time to cancel.")

    cmd_parts = ["recovar", "pipeline"]

    # ── Step 1: Particles ────────────────────────────────────────────────
    _heading("Step 1: Particle data")
    _info("Select your particles file. RECOVAR auto-extracts poses and CTF")
    _info("from .star and .cs files — no preprocessing needed.")

    particles = _pick_file(
        "Particles file",
        ["*.star", "*.cs", "*.mrcs", "*.txt"],
    )
    if not particles:
        _error("Particles file is required.")
        sys.exit(1)
    cmd_parts.append(particles)

    is_star_or_cs = particles.lower().endswith((".star", ".cs"))

    # ── Step 2: Output directory ─────────────────────────────────────────
    _heading("Step 2: Output directory")
    _info("You can use a simple output directory, or use the project system")
    _info("for organized, auto-numbered job directories.")
    use_project = _prompt_yesno("Use project system (recommended for multi-step workflows)?", default=False)
    if use_project:
        project_dir = _prompt("Project directory", default="recovar_project")
        cmd_parts.extend(["--project", project_dir])
        _info(f"Output will be auto-numbered under {project_dir}/Pipeline/job_NNNN/")
    else:
        outdir = _prompt("Output directory", default="recovar_output")
        cmd_parts.extend(["-o", outdir])

    # ── Step 2b: Data directory & strip-prefix (for .star / .cs) ────────
    if is_star_or_cs:
        _heading("Step 2b: Data directory (optional)")
        _info("If your MRC files are in a different location than recorded")
        _info("in the .star/.cs file, specify the base directory here.")
        _info("RECOVAR auto-resolves .mrc/.mrcs extensions and flat directories.")
        _info("Run 'recovar check_paths <particles> --datadir <dir>' to diagnose.")
        datadir = _prompt("Data directory (Enter to auto-detect from particles file)", required=False)
        if datadir:
            cmd_parts.extend(["--datadir", datadir])

        _info("")
        _info("If paths in your metadata contain a prefix that doesn't exist")
        _info("on this system (e.g. 'J3/imported'), strip it here.")
        strip_prefix = _prompt("Strip prefix (Enter to skip)", required=False)
        if strip_prefix:
            cmd_parts.extend(["--strip-prefix", strip_prefix])

    # ── Step 3: Mask ─────────────────────────────────────────────────────
    _heading("Step 3: Solvent mask")
    _info("A mask improves SNR. You can provide a .mrc file, or use")
    _info("'from_halfmaps' to auto-generate, or 'sphere' for a loose mask.")

    mask = _pick_file(
        "Mask",
        ["*mask*.mrc", "*.mrc"],
        allow_special=[
            ("from_halfmaps", "Auto-generate from halfmaps"),
            ("sphere", "Loose spherical mask"),
            ("none", "No mask"),
        ],
    )
    cmd_parts.extend(["--mask", mask])

    # ── Step 4: Downsampling ─────────────────────────────────────────────
    _heading("Step 4: Downsampling")
    _info("By default, images are downsampled to 256px (skipped if already near that size).")
    _info("Fourier cropping preserves frequency content up to the new Nyquist.")

    downsample = _prompt("Downsample to box size (Enter for default 256, 'no' to disable)", required=False)
    if downsample:
        if downsample.lower() in ("no", "none", "off", "false"):
            cmd_parts.append("--no-downsample")
        else:
            try:
                ds_int = int(downsample)
                if ds_int % 2 != 0:
                    _warn("Box size must be even. Rounding down.")
                    ds_int = ds_int - 1
                cmd_parts.extend(["--downsample", str(ds_int)])
            except ValueError:
                _warn(f"Invalid number '{downsample}', using default (256).")

    # ── Step 5: Poses & CTF (only if not .star/.cs) ──────────────────────
    if not is_star_or_cs:
        _heading("Step 5: Poses and CTF")
        _info("Your particles file is not .star or .cs, so poses and CTF")
        _info("must be provided as pickle files.")

        poses = _pick_file("Poses file (.pkl)", ["*poses*.pkl", "*.pkl"])
        if poses:
            cmd_parts.extend(["--poses", poses])

        ctf = _pick_file("CTF file (.pkl)", ["*ctf*.pkl", "*.pkl"])
        if ctf:
            cmd_parts.extend(["--ctf", ctf])

    # ── Step 6: Focus mask (optional) ────────────────────────────────────
    _heading("Step 6: Focus mask (optional)")
    _info("A focus mask restricts heterogeneity analysis to a specific region.")

    focus_mask = _pick_file(
        "Focus mask (or Enter to skip)",
        ["*focus*mask*.mrc", "*mask*.mrc"],
        required=False,
    )
    if focus_mask:
        cmd_parts.extend(["--focus-mask", focus_mask])

    # ── Step 7: Additional options ───────────────────────────────────────
    _heading("Step 7: Additional options")

    # Lazy loading
    if _prompt_yesno("Use lazy loading (for large datasets)?", default=False):
        cmd_parts.append("--lazy")

    # Contrast correction
    if _prompt_yesno("Correct amplitude contrast?", default=False):
        cmd_parts.append("--correct-contrast")

    # Quick mean-only run
    if _prompt_yesno("Quick mean-only run (to verify setup)?", default=False):
        cmd_parts.append("--only-mean")

    # Tilt series
    if _prompt_yesno("Is this tilt series / cryo-ET data?", default=False):
        cmd_parts.append("--tilt-series")
        tilt_ctf = _prompt_choice(
            "CTF model for tilt series",
            [
                ("relion5", "RELION 5 per-tilt CTF (recommended)"),
                ("cryoem", "Standard cryo-EM CTF"),
                ("warp", "Warp (Windows)"),
            ],
            default="relion5",
        )
        cmd_parts.extend(["--tilt-series-ctf", tilt_ctf])

    # GPU memory limit
    gpu_gb = _prompt("GPU memory limit in GB (Enter for all available)", required=False)
    if gpu_gb:
        try:
            int(gpu_gb)
            cmd_parts.extend(["--gpu-budget-gb", gpu_gb])
        except ValueError:
            _warn(f"Invalid number '{gpu_gb}', using all available GPU memory.")

    # zdim
    zdim_custom = _prompt(
        "Latent dimensions (comma-separated, or Enter for default 1,2,4,10,20)",
        required=False,
    )
    if zdim_custom:
        cmd_parts.extend(["--zdim", zdim_custom])

    # ── Summary ──────────────────────────────────────────────────────────
    _heading("Command summary")

    # Group flag-value pairs so they stay on the same line
    display_parts = []
    i = 0
    while i < len(cmd_parts):
        part = cmd_parts[i]
        # If this is a flag followed by a value, keep them together
        if part.startswith("-") and i + 1 < len(cmd_parts) and not cmd_parts[i + 1].startswith("-"):
            display_parts.append(f"{part} {cmd_parts[i + 1]}")
            i += 2
        else:
            display_parts.append(part)
            i += 1

    cmd_str = " \\\n    ".join(display_parts)
    print()
    print(f"  {_GREEN}$ {cmd_str}{_RESET}")
    print()

    if args.dry_run:
        _info("(dry run — not executing)")
        return

    if _prompt_yesno("Execute this command now?", default=True):
        print()
        _info("Launching RECOVAR pipeline...")
        print()
        from recovar.utils.subprocess_helpers import recovar_subprocess_env

        result = subprocess.run(cmd_parts, env=recovar_subprocess_env())
        sys.exit(result.returncode)
    else:
        _info("Command printed above — copy and run when ready.")
        print()
        print(f"  {_DIM}{' '.join(cmd_parts)}{_RESET}")
        print()


if __name__ == "__main__":
    main()
