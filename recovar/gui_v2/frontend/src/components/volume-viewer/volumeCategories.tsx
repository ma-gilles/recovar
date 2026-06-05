import { useState } from "react";
import { Box, ChevronRight, ChevronDown, Pin, Wand2 } from "lucide-react";
import { clsx } from "clsx";
import type { VolumeEntry } from "../../lib/api/client";

/**
 * Format a job `type` for display. Job types arrive as PascalCase
 * (e.g. "ComputeState") or snake_case; split camelCase boundaries and
 * underscores into spaces so they render as "Compute State" rather than
 * "Computestate".
 */
export function formatJobType(type: string): string {
  return type
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/_/g, " ");
}

/**
 * Maps PascalCase job type names to the URL-friendly snake_case slugs used
 * by the /jobs/new route. Single source of truth for both suggested-next
 * links and clone params.
 */
export const TYPE_URL_MAP: Record<string, string> = {
  Pipeline: "pipeline",
  Analyze: "analyze",
  ComputeState: "compute_state",
  ComputeTrajectory: "compute_trajectory",
  Density: "density",
  StableStates: "stable_states",
  Postprocess: "postprocess",
  Downsample: "downsample",
  ReconstructState: "reconstruct_state",
  ReconstructTrajectory: "reconstruct_trajectory",
};

/**
 * Resolve a job type to its /jobs/new URL slug, falling back to a
 * camelCase→snake_case conversion for unknown types.
 */
export function jobTypeToUrlSlug(type: string): string {
  return TYPE_URL_MAP[type] ?? type.replace(/([a-z])([A-Z])/g, "$1_$2").toLowerCase();
}

// ---------------------------------------------------------------------------
// Volume filtering & display helpers (shared by jobs/$jobId and explore/$jobId)
// ---------------------------------------------------------------------------

/** Patterns matching "uninteresting" volumes hidden by default. */
export const HIDDEN_PATTERNS = [/_half[0-9]/, /_unfil/, /halfmap/, /unfiltered/i, /^sampling\.mrc$/i];

/** Returns true if a volume name matches the hidden-by-default patterns. */
export function isHiddenVolume(name: string): boolean {
  const lower = name.toLowerCase();
  return HIDDEN_PATTERNS.some((pat) => pat.test(lower));
}

/**
 * Build a display name for a volume.  If `needsDisambiguation` is true
 * (i.e. another volume in the same list has an identical filename),
 * prepend the parent directory.
 */
export function volumeDisplayName(v: VolumeEntry, needsDisambiguation: boolean): string {
  if (!needsDisambiguation) return v.name;
  const parts = v.path.replace(/\\/g, "/").split("/");
  if (parts.length >= 2) {
    return `${parts[parts.length - 2]}/${v.name}`;
  }
  return v.name;
}

/** Human-readable labels for volume categories. */
export const CATEGORY_LABELS: Record<string, string> = {
  mean: "Mean Reconstruction",
  eigen: "Eigenvolumes",
  variance: "Variance Map",
  halfmap: "Half-maps (raw)",
  mask: "Masks",
  kmeans_center: "K-means Centers",
  trajectory: "Trajectory Volumes",
  reconstruction: "Reconstructed States",
  density: "Density / Deconvolved",
  other: "Other",
};

/** Canonical ordering for category groups. */
export const CATEGORY_ORDER: string[] = [
  "mean",
  "eigen",
  "variance",
  "kmeans_center",
  "trajectory",
  "reconstruction",
  "density",
  "mask",
  "other",
  "halfmap",
];

/** Categories collapsed by default. */
export const COLLAPSED_BY_DEFAULT = new Set(["halfmap", "other"]);

/**
 * Natural sort comparator: splits on numeric boundaries so that
 * "vol_2" sorts before "vol_10".
 */
export function naturalCompare(a: string, b: string): number {
  const re = /(\d+)/g;
  const aParts = a.split(re);
  const bParts = b.split(re);
  const len = Math.min(aParts.length, bParts.length);
  for (let i = 0; i < len; i++) {
    const aNum = Number(aParts[i]);
    const bNum = Number(bParts[i]);
    if (!isNaN(aNum) && !isNaN(bNum)) {
      if (aNum !== bNum) return aNum - bNum;
    } else {
      const cmp = (aParts[i] ?? "").localeCompare(bParts[i] ?? "");
      if (cmp !== 0) return cmp;
    }
  }
  return aParts.length - bParts.length;
}

export function VolumeCategoryGroup({
  cat,
  vols,
  selectedVolume,
  onSelect,
  ambiguousNames,
  defaultCollapsed,
  pinnedPaths,
  onPin,
  onUnpin,
  pinDisabled,
  onMakeMask,
}: {
  cat: string;
  vols: VolumeEntry[];
  selectedVolume: string | null;
  onSelect: (path: string) => void;
  ambiguousNames: Set<string>;
  defaultCollapsed: boolean;
  pinnedPaths?: Set<string>;
  onPin?: (path: string, name: string) => void;
  onUnpin?: (path: string) => void;
  pinDisabled?: boolean;
  onMakeMask?: (path: string, name: string) => void;
}): React.JSX.Element {
  const [open, setOpen] = useState(!defaultCollapsed);

  return (
    <div>
      <button
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="flex w-full items-center gap-1.5 py-1.5 text-left text-xs font-medium uppercase tracking-wider text-zinc-500 hover:text-zinc-300 outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-1 focus-visible:ring-offset-zinc-950 rounded"
      >
        {open ? (
          <ChevronDown className="h-3.5 w-3.5 shrink-0" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 shrink-0" />
        )}
        {CATEGORY_LABELS[cat] ?? cat}
        <span className="font-normal normal-case tracking-normal text-zinc-600">
          ({vols.length})
        </span>
      </button>
      {open && (
        <div className="ml-5 space-y-px">
          {vols.map((v) => {
            const displayName = volumeDisplayName(v, ambiguousNames.has(v.name));
            const active = selectedVolume === v.path;
            return (
              <div
                key={v.path}
                className={clsx(
                  "flex items-center gap-2 rounded px-2 py-1 text-sm",
                  active
                    ? "bg-blue-500/15 text-blue-300"
                    : "text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100"
                )}
                title={v.path}
              >
                <button
                  className="flex flex-1 items-center gap-2 text-left min-w-0"
                  onClick={() => onSelect(v.path)}
                >
                  <Box className="h-3.5 w-3.5 shrink-0 text-sky-400" />
                  <span className="truncate">{displayName}</span>
                </button>
                <span className="shrink-0 text-xs text-zinc-600">
                  {(v.size_bytes / 1e6).toFixed(1)} MB
                </span>
                {onPin && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      const isPinned = pinnedPaths?.has(v.path);
                      if (isPinned) onUnpin?.(v.path);
                      else onPin(v.path, v.name);
                    }}
                    className={clsx(
                      "shrink-0",
                      pinnedPaths?.has(v.path)
                        ? "text-blue-400"
                        : "text-zinc-600 hover:text-zinc-300"
                    )}
                    disabled={pinDisabled && !pinnedPaths?.has(v.path)}
                    aria-label={pinnedPaths?.has(v.path) ? `Unpin ${displayName}` : `Pin ${displayName}`}
                  >
                    <Pin className="h-3 w-3" />
                  </button>
                )}
                {onMakeMask && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onMakeMask(v.path, v.name);
                    }}
                    className="shrink-0 rounded p-0.5 text-emerald-500 hover:bg-emerald-500/15 hover:text-emerald-300"
                    aria-label={`Create mask from ${displayName}`}
                    title="Create mask from this volume"
                  >
                    <Wand2 className="h-4 w-4" />
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
