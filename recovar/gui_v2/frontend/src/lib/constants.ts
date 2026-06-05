/** Maximum simultaneously pinned volumes in the viewer. */
export const MAX_PINNED_VOLUMES = 4;

/**
 * Selectable view-resolution (downsample target box sizes) for the 3D viewer,
 * in addition to "Auto" (server default) and "Full" (original resolution).
 * Single source of truth — keep in sync with backend MAX_SERVE_DIM.
 */
export const VIEW_DIMS = [256, 128, 64];

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
 * that will be deleted when the server restarts.
 */
export function isEphemeralPath(path: string): boolean {
  const normalized = path.replace(/\/+$/, ""); // strip trailing slashes
  return EPHEMERAL_PREFIXES.some(
    (prefix) => normalized === prefix.replace(/\/+$/, "") || normalized.startsWith(prefix.endsWith("/") ? prefix : prefix + "/")
  );
}

/** Warning message for ephemeral project paths. */
export const EPHEMERAL_PATH_WARNING =
  "This directory is temporary and will be deleted when the server restarts. Choose a path on scratch or a persistent filesystem.";
