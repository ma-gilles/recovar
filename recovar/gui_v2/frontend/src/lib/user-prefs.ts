/**
 * Per-project SLURM defaults stored in localStorage.
 * Used by SlurmSettings to remember the user's last-used values for each project.
 */

import type { SlurmOpts } from "../components/job-form/SlurmSettings";

const STORAGE_KEY_PREFIX = "recovar:slurm-defaults:";

export function getSlurmDefaults(projectPath: string): SlurmOpts | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_PREFIX + projectPath);
    if (!raw) return null;
    return JSON.parse(raw) as SlurmOpts;
  } catch {
    return null;
  }
}

export function saveSlurmDefaults(projectPath: string, opts: SlurmOpts): void {
  try {
    localStorage.setItem(STORAGE_KEY_PREFIX + projectPath, JSON.stringify(opts));
  } catch {
    // localStorage may be unavailable or full — ignore
  }
}
