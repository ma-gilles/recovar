/** Maximum simultaneously pinned volumes in the viewer. */
export const MAX_PINNED_VOLUMES = 4;

/**
 * Selectable view-resolution (downsample target box sizes) for the 3D viewer,
 * in addition to "Auto" (server default) and "Full" (original resolution).
 * Single source of truth — keep in sync with backend MAX_SERVE_DIM.
 */
export const VIEW_DIMS = [256, 128, 64];

/**
 * Default 3D view resolution: a downsampled box so volumes render fast over
 * SSH tunnels out of the box. The server clamps this up to the original size
 * for smaller volumes, so it is always safe.
 */
export const DEFAULT_VIEW_DIM = 128;

/** Subsample scatter plot above this particle count. */
export const SUBSAMPLE_THRESHOLD = 2_000_000;

/** Display subsample size for scatter plots exceeding SUBSAMPLE_THRESHOLD. */
export const DISPLAY_SUBSAMPLE_SIZE = 1_000_000;

/**
 * Directory prefixes that are ephemeral (wiped on reboot).
 * Used to warn users when creating projects in temporary locations.
 */
const EPHEMERAL_PREFIXES = ["/tmp/", "/tmp", "/dev/shm/", "/dev/shm"];

/**
 * Returns true if the given path is under an ephemeral/temporary directory
 * that may be cleared when the machine reboots (or by tmp-cleaners).
 */
export function isEphemeralPath(path: string): boolean {
  const normalized = path.replace(/\/+$/, ""); // strip trailing slashes
  return EPHEMERAL_PREFIXES.some(
    (prefix) => normalized === prefix.replace(/\/+$/, "") || normalized.startsWith(prefix.endsWith("/") ? prefix : prefix + "/")
  );
}

/** Warning message for ephemeral project paths. */
export const EPHEMERAL_PATH_WARNING =
  "This directory is temporary and may be cleared when the machine reboots. Choose a path on scratch or a persistent filesystem.";
