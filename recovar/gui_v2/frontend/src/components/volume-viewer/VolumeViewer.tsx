import React, { useCallback, useEffect, useMemo, useState, Suspense } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  RotateCcw,
  Pin,
  Layers,
  Eye,
  EyeOff,
} from "lucide-react";
import { clsx } from "clsx";
import { getVolumeInfo, type VolumeEntry } from "../../lib/api/client";
import { Button } from "../ui/button";
import { Spinner } from "../ui/spinner";
import { MAX_PINNED_VOLUMES, VIEW_DIMS, DEFAULT_VIEW_DIM } from "../../lib/constants";
import { VtkErrorBoundary } from "./VtkErrorBoundary";
import type { DownsampleInfo } from "./VtkViewer";

// Lazy-load VtkViewer so vtk.js (~3-4 MB) is split into a separate chunk
// and only fetched when the user opens the 3D view.
const VtkViewer = React.lazy(() =>
  import("./VtkViewer").then((m) => ({ default: m.VtkViewer }))
);

import {
  TrajectoryPlayer,
  getTrajectoryVolumes,
  isTrajectoryVolume,
} from "./TrajectoryPlayer";

const VOLUME_COLOR_HEX = ["#38bdf8", "#fb7185", "#34d399", "#fbbf24"];

export interface PinnedVolume {
  path: string;
  name: string;
  threshold: number;
  opacity: number;
  visible: boolean;
  colorIndex: number;
  category?: string;
}

interface VolumeViewerProps {
  volumes?: VolumeEntry[];
  initialVolumePath?: string;
  /** Hide the built-in volume list (when the parent already provides one) */
  hideVolumeList?: boolean;
  /** Controlled pinning state — when provided, VolumeViewer does not manage its own pin state */
  pinnedVolumes?: PinnedVolume[];
  onPinnedVolumesChange?: (pinned: PinnedVolume[]) => void;
}

/**
 * Volume viewer using server-rendered orthogonal slices.
 * Full vtk.js isosurface rendering requires the vtk.js package (installed separately).
 * This provides the slice view mode and controls as a fallback/default.
 */
export function VolumeViewer({
  volumes,
  initialVolumePath,
  hideVolumeList,
  pinnedVolumes: controlledPinned,
  onPinnedVolumesChange,
}: VolumeViewerProps): React.JSX.Element {
  const [internalPinned, setInternalPinned] = useState<PinnedVolume[]>([]);
  // Use controlled state if provided, otherwise internal
  const pinnedVolumes = controlledPinned ?? internalPinned;
  const setPinnedVolumes = useCallback(
    (next: PinnedVolume[] | ((prev: PinnedVolume[]) => PinnedVolume[])) => {
      if (onPinnedVolumesChange) {
        // Controlled: resolve the functional-updater form against current
        // value before handing a plain array to the parent callback.
        onPinnedVolumesChange(typeof next === "function" ? next(pinnedVolumes) : next);
      } else {
        setInternalPinned(next);
      }
    },
    [onPinnedVolumesChange, pinnedVolumes],
  );
  const [activeVolume, setActiveVolume] = useState<string | null>(initialVolumePath ?? null);

  // Sync activeVolume when the parent changes initialVolumePath (e.g. user
  // clicks a different volume in an external list while hideVolumeList is set).
  useEffect(() => {
    if (initialVolumePath != null) {
      setActiveVolume(initialVolumePath);
    }
  }, [initialVolumePath]);

  const [axis, setAxis] = useState<0 | 1 | 2>(2); // Z axis default
  const [sliceIdx, setSliceIdx] = useState(0);
  const [viewMode, setViewMode] = useState<"slice" | "3d">("3d");
  // Live slider value (drives the label, updates every onChange) and the
  // debounced value that actually drives marching-cubes re-meshing. Debouncing
  // avoids a synchronous main-thread re-mesh + render on every slider tick.
  const [activeSigma, setActiveSigma] = useState(3.0);
  const [committedSigma, setCommittedSigma] = useState(3.0);
  const [maxSlice, setMaxSlice] = useState(128);
  const [downsampleInfo, setDownsampleInfo] = useState<DownsampleInfo | null>(null);
  // Selected view resolution (downsample target). Defaults to 128 so volumes
  // load fast over slow/SSH links; users can switch to 256/Full for detail.
  const [viewDim, setViewDim] = useState<number | null>(DEFAULT_VIEW_DIM);

  // Debounce the sigma slider: commit the value to the renderer ~140ms after
  // the user stops dragging so re-meshing runs once instead of per tick.
  useEffect(() => {
    const id = window.setTimeout(() => setCommittedSigma(activeSigma), 140);
    return () => window.clearTimeout(id);
  }, [activeSigma]);

  const activeCategory = volumes?.find((v) => v.path === activeVolume)?.category;

  // Trajectory detection
  const trajectoryVolumes = useMemo(
    () => (volumes ? getTrajectoryVolumes(volumes) : []),
    [volumes]
  );
  const [trajectoryActive, setTrajectoryActive] = useState(false);
  const isInTrajectory =
    !!activeVolume && isTrajectoryVolume(activeVolume, trajectoryVolumes);
  const showTrajectory = isInTrajectory && trajectoryVolumes.length >= 2;

  // Compute prefetch paths: the next few frames in the trajectory
  const prefetchPaths = useMemo(() => {
    if (!showTrajectory || !activeVolume) return undefined;
    const idx = trajectoryVolumes.findIndex((v) => v.path === activeVolume);
    if (idx < 0) return undefined;
    const ahead: string[] = [];
    // Prefetch next 3 frames
    for (let i = 1; i <= 3; i++) {
      const nextIdx = (idx + i) % trajectoryVolumes.length;
      ahead.push(trajectoryVolumes[nextIdx].path);
    }
    return ahead;
  }, [showTrajectory, activeVolume, trajectoryVolumes]);

  // Handle trajectory frame changes
  const handleTrajectoryFrame = useCallback((path: string) => {
    setActiveVolume(path);
    setTrajectoryActive(true);
  }, []);

  // Load volume info for the active volume
  const { data: volInfo } = useQuery({
    queryKey: ["volume-info", activeVolume],
    queryFn: () => getVolumeInfo(activeVolume!),
    enabled: !!activeVolume,
  });

  // Largest box dimension of the active volume (for disabling resolution
  // options that exceed it and for resolving the "Full" option).
  const maxVolDim = volInfo ? Math.max(...volInfo.shape) : null;

  useEffect(() => {
    if (volInfo) {
      setMaxSlice(volInfo.shape[axis] - 1);
      setSliceIdx(Math.floor(volInfo.shape[axis] / 2));
    }
  }, [volInfo, axis]);

  // Load slice image
  const sliceUrl = activeVolume
    ? `/api/volumes/slice?path=${encodeURIComponent(activeVolume)}&axis=${axis}&idx=${sliceIdx}`
    : null;

  const handleDownsampleInfo = useCallback((info: DownsampleInfo | null) => {
    setDownsampleInfo(info);
  }, []);

  const loadVolume = useCallback(
    (path: string, _name: string) => {
      setActiveVolume(path);
      setSliceIdx(0);
      setTrajectoryActive(false);
      setDownsampleInfo(null); // clear until new volume reports back
      // Keep the fast downsampled default for every volume (the server clamps
      // it up to the original size when the new volume is smaller, so it is
      // always safe). Resetting to Auto here would silently make each
      // subsequent volume load at full server resolution — slower over SSH.
      setViewDim(DEFAULT_VIEW_DIM);
    },
    []
  );

  const pinVolume = useCallback(
    (path: string, name: string) => {
      if (pinnedVolumes.length >= MAX_PINNED_VOLUMES) return;
      if (pinnedVolumes.some((v) => v.path === path)) return;
      const category = volumes?.find((v) => v.path === path)?.category;
      setPinnedVolumes((prev) => [
        ...prev,
        {
          path,
          name,
          threshold: 3.0,
          opacity: 0.8,
          visible: true,
          colorIndex: prev.length,
          category,
        },
      ]);
    },
    [pinnedVolumes, volumes]
  );

  const unpinVolume = useCallback((path: string) => {
    setPinnedVolumes((prev) => prev.filter((v) => v.path !== path));
  }, []);

  const updatePinned = useCallback(
    (path: string, updates: Partial<PinnedVolume>) => {
      setPinnedVolumes((prev) =>
        prev.map((v) => (v.path === path ? { ...v, ...updates } : v))
      );
    },
    []
  );

  return (
    <div className="flex gap-4">
      {/* Viewer panel */}
      <div className="flex-1 space-y-3">
        {/* Toolbar */}
        <div className="flex items-center gap-2">
          <div className="flex gap-1 rounded-md border border-zinc-700 p-0.5">
            <button
              onClick={() => setViewMode("slice")}
              aria-pressed={viewMode === "slice"}
              className={clsx(
                "rounded px-2 py-1 text-xs focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-1 focus-visible:ring-offset-zinc-950 outline-none",
                viewMode === "slice" ? "bg-zinc-700 text-zinc-50" : "text-zinc-400"
              )}
            >
              <Layers className="inline h-3 w-3" /> Slice
            </button>
            <button
              onClick={() => setViewMode("3d")}
              aria-pressed={viewMode === "3d"}
              className={clsx(
                "rounded px-2 py-1 text-xs focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-1 focus-visible:ring-offset-zinc-950 outline-none",
                viewMode === "3d" ? "bg-zinc-700 text-zinc-50" : "text-zinc-400"
              )}
            >
              3D
            </button>
          </div>

          {viewMode === "slice" && (
            <>
              <div className="flex gap-1 rounded-md border border-zinc-700 p-0.5">
                {(["X", "Y", "Z"] as const).map((label, i) => (
                  <button
                    key={label}
                    onClick={() => setAxis(i as 0 | 1 | 2)}
                    aria-label={`${label} axis`}
                    aria-pressed={axis === i}
                    className={clsx(
                      "rounded px-2 py-1 text-xs focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-1 focus-visible:ring-offset-zinc-950 outline-none",
                      axis === i ? "bg-zinc-700 text-zinc-50" : "text-zinc-400"
                    )}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <input
                type="range"
                min={0}
                max={maxSlice}
                value={sliceIdx}
                onChange={(e) => setSliceIdx(parseInt(e.target.value))}
                className="w-32"
              />
              <span className="text-xs text-zinc-500">
                {sliceIdx} / {maxSlice}
              </span>
            </>
          )}

          {viewMode === "3d" && activeVolume && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-zinc-400">Sigma</span>
              <input
                type="range"
                min={0}
                max={10}
                step={0.1}
                value={activeSigma}
                onChange={(e) => setActiveSigma(parseFloat(e.target.value))}
                className="w-40"
              />
              <span className="text-xs text-zinc-300 w-10">{activeSigma.toFixed(1)}σ</span>
              <span className="text-xs text-zinc-400">Resolution</span>
              <select
                value={
                  viewDim == null
                    ? "auto"
                    : maxVolDim != null && viewDim >= maxVolDim
                      ? "full"
                      : VIEW_DIMS.includes(viewDim)
                        ? String(viewDim)
                        : "full"
                }
                onChange={(e) => {
                  const v = e.target.value;
                  if (v === "auto") {
                    // Auto: server default (MAX_SERVE_DIM).
                    setViewDim(null);
                  } else if (v === "full") {
                    // Full: request a target >= the original box size so the
                    // server serves it via the fast path (max_dim <= target).
                    // Round up to even because the server only accepts an even
                    // `dim`; an odd box (e.g. 255) would otherwise 400.
                    setViewDim(maxVolDim != null ? maxVolDim + (maxVolDim % 2) : null);
                  } else {
                    setViewDim(parseInt(v));
                  }
                }}
                className="rounded border border-zinc-700 bg-zinc-900 px-1.5 py-1 text-xs text-zinc-300 outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
                aria-label="View resolution"
              >
                <option value="auto">Auto</option>
                {VIEW_DIMS.map((d) => (
                  <option key={d} value={d} disabled={maxVolDim != null && d > maxVolDim}>
                    {d}
                  </option>
                ))}
                <option value="full">Full</option>
              </select>
            </div>
          )}

          {/* Reset view only recenters the slice, so hide it in 3D mode where
              camera control lives in the viewport (drag to rotate). */}
          {viewMode === "slice" && (
            <div className="ml-auto">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setSliceIdx(Math.floor(maxSlice / 2))}
                aria-label="Reset view"
              >
                <RotateCcw className="h-3.5 w-3.5" />
              </Button>
            </div>
          )}
        </div>

        {/* Viewport */}
        <div className="flex items-center justify-center rounded-lg border border-zinc-800 bg-black" style={{ minHeight: 400 }}>
          {!activeVolume ? (
            <p className="text-sm text-zinc-500">Select a volume to view</p>
          ) : viewMode === "slice" && sliceUrl ? (
            <img
              src={sliceUrl}
              alt={`Slice ${axis}:${sliceIdx}`}
              className="max-h-[500px] max-w-full"
              style={{ imageRendering: "pixelated" }}
            />
          ) : viewMode === "3d" ? (
            <VtkErrorBoundary onWebGLFail={() => setViewMode("slice")}>
              <Suspense fallback={<Spinner label="Loading 3D viewer..." />}>
                <VtkViewer
                  activeVolume={activeVolume}
                  pinnedVolumes={pinnedVolumes}
                  activeSigma={committedSigma}
                  activeCategory={activeCategory}
                  preserveCamera={trajectoryActive}
                  prefetchPaths={prefetchPaths}
                  onDownsampleInfo={handleDownsampleInfo}
                  viewDim={viewDim}
                />
              </Suspense>
            </VtkErrorBoundary>
          ) : (
            <Spinner label="Loading..." />
          )}
        </div>

        {/* Trajectory playback bar */}
        {showTrajectory && viewMode === "3d" && activeVolume && (
          <TrajectoryPlayer
            volumes={trajectoryVolumes}
            currentFrame={activeVolume}
            onFrameChange={handleTrajectoryFrame}
          />
        )}

        {/* Volume info */}
        {volInfo && (
          <div className="flex items-center gap-4 text-xs text-zinc-500">
            <span>Shape: {volInfo.shape.join(" x ")}</span>
            <span>Voxel: {volInfo.voxel_size.toFixed(2)} A</span>
            <span>
              Range: [{volInfo.min.toFixed(3)}, {volInfo.max.toFixed(3)}]
            </span>
            {downsampleInfo && (
              <span className="rounded bg-amber-900/40 px-1.5 py-0.5 text-amber-400">
                Viewing at {downsampleInfo.servedDim}&sup3; (original: {downsampleInfo.originalDim}&sup3;)
              </span>
            )}
          </div>
        )}
      </div>

      {/* Volume list / pin panel — hidden when parent provides its own list */}
      {(!hideVolumeList || pinnedVolumes.length > 0) && (
        <div className="w-56 space-y-3">
          {/* Pinned volumes */}
          {pinnedVolumes.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-xs font-medium uppercase tracking-wider text-zinc-500">
                Pinned ({pinnedVolumes.length}/{MAX_PINNED_VOLUMES})
              </h4>
              {pinnedVolumes.map((pv) => (
                <div key={pv.path} className="rounded-md border border-zinc-800 bg-zinc-900 p-2 space-y-1">
                  <div className="flex items-center gap-1">
                    <span
                      className="h-2 w-2 rounded-full"
                      style={{ backgroundColor: VOLUME_COLOR_HEX[pv.colorIndex % VOLUME_COLOR_HEX.length] }}
                    />
                    <span className="flex-1 truncate text-xs">{pv.name}</span>
                    <button
                      onClick={() => updatePinned(pv.path, { visible: !pv.visible })}
                      className="text-zinc-500 hover:text-zinc-300"
                      aria-label={pv.visible ? `Hide ${pv.name}` : `Show ${pv.name}`}
                    >
                      {pv.visible ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}
                    </button>
                    <button
                      onClick={() => unpinVolume(pv.path)}
                      className="text-zinc-500 hover:text-red-400"
                      aria-label={`Unpin ${pv.name}`}
                    >
                      <Pin className="h-3 w-3" />
                    </button>
                  </div>
                  <div className="space-y-0.5">
                    <div className="flex items-center gap-1 text-xs text-zinc-500">
                      <span className="w-12">Sigma</span>
                      <input
                        type="range"
                        min={0}
                        max={10}
                        step={0.1}
                        value={pv.threshold}
                        onChange={(e) => updatePinned(pv.path, { threshold: parseFloat(e.target.value) })}
                        className="flex-1"
                      />
                      <span className="w-8 text-right">{pv.threshold.toFixed(1)}</span>
                    </div>
                    <div className="flex items-center gap-1 text-xs text-zinc-500">
                      <span className="w-12">Opacity</span>
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={pv.opacity}
                        onChange={(e) => updatePinned(pv.path, { opacity: parseFloat(e.target.value) })}
                        className="flex-1"
                      />
                      <span className="w-8 text-right">{pv.opacity.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Volume list — only shown when not hidden by parent */}
          {!hideVolumeList && volumes && volumes.length > 0 && (
            <div className="space-y-1">
              <h4 className="text-xs font-medium uppercase tracking-wider text-zinc-500">
                Available Volumes
              </h4>
              {volumes.map((v) => (
                <div
                  key={v.path}
                  className={clsx(
                    "flex items-center gap-1 rounded px-2 py-1 text-xs cursor-pointer",
                    activeVolume === v.path
                      ? "bg-zinc-700 text-zinc-50"
                      : "text-zinc-400 hover:bg-zinc-800"
                  )}
                >
                  <button
                    className="flex-1 truncate text-left"
                    onClick={() => loadVolume(v.path, v.name)}
                  >
                    {v.name}
                  </button>
                  <button
                    onClick={() => pinVolume(v.path, v.name)}
                    className={clsx(
                      "shrink-0",
                      pinnedVolumes.some((p) => p.path === v.path)
                        ? "text-blue-400"
                        : "text-zinc-600 hover:text-zinc-300"
                    )}
                    disabled={
                      pinnedVolumes.length >= MAX_PINNED_VOLUMES &&
                      !pinnedVolumes.some((p) => p.path === v.path)
                    }
                    aria-label={`Pin ${v.name}`}
                  >
                    <Pin className="h-3 w-3" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
