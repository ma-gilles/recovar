import { useCallback, useEffect, useState } from "react";
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
import { MAX_PINNED_VOLUMES } from "../../lib/constants";
import { VtkViewer, VtkErrorBoundary } from "./VtkViewer";

const VOLUME_COLOR_HEX = ["#38bdf8", "#fb7185", "#34d399", "#fbbf24"];

interface PinnedVolume {
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
}

/**
 * Volume viewer using server-rendered orthogonal slices.
 * Full vtk.js isosurface rendering requires the vtk.js package (installed separately).
 * This provides the slice view mode and controls as a fallback/default.
 */
export function VolumeViewer({ volumes, initialVolumePath }: VolumeViewerProps): React.JSX.Element {
  const [pinnedVolumes, setPinnedVolumes] = useState<PinnedVolume[]>([]);
  const [activeVolume, setActiveVolume] = useState<string | null>(initialVolumePath ?? null);
  const [axis, setAxis] = useState<0 | 1 | 2>(2); // Z axis default
  const [sliceIdx, setSliceIdx] = useState(0);
  const [viewMode, setViewMode] = useState<"slice" | "3d">("3d");
  const [activeSigma, setActiveSigma] = useState(3.0);
  const [maxSlice, setMaxSlice] = useState(128);

  const activeCategory = volumes?.find((v) => v.path === activeVolume)?.category;

  // Load volume info for the active volume
  const { data: volInfo } = useQuery({
    queryKey: ["volume-info", activeVolume],
    queryFn: () => getVolumeInfo(activeVolume!),
    enabled: !!activeVolume,
  });

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

  const loadVolume = useCallback(
    (path: string, _name: string) => {
      setActiveVolume(path);
      setSliceIdx(0);
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
              className={clsx(
                "rounded px-2 py-1 text-xs",
                viewMode === "slice" ? "bg-zinc-700 text-zinc-50" : "text-zinc-400"
              )}
            >
              <Layers className="inline h-3 w-3" /> Slice
            </button>
            <button
              onClick={() => setViewMode("3d")}
              className={clsx(
                "rounded px-2 py-1 text-xs",
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
                    className={clsx(
                      "rounded px-2 py-1 text-xs",
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
              <span className="text-xs text-zinc-400">Threshold</span>
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
            </div>
          )}

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
              <VtkViewer
                activeVolume={activeVolume}
                pinnedVolumes={pinnedVolumes}
                activeSigma={activeSigma}
                activeCategory={activeCategory}
              />
            </VtkErrorBoundary>
          ) : (
            <Spinner label="Loading..." />
          )}
        </div>

        {/* Volume info */}
        {volInfo && (
          <div className="flex gap-4 text-xs text-zinc-500">
            <span>Shape: {volInfo.shape.join(" x ")}</span>
            <span>Voxel: {volInfo.voxel_size.toFixed(2)} A</span>
            <span>
              Range: [{volInfo.min.toFixed(3)}, {volInfo.max.toFixed(3)}]
            </span>
          </div>
        )}
      </div>

      {/* Volume list / pin panel */}
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
                    style={{ backgroundColor: VOLUME_COLOR_HEX[pv.colorIndex] }}
                  />
                  <span className="flex-1 truncate text-xs">{pv.name}</span>
                  <button
                    onClick={() => updatePinned(pv.path, { visible: !pv.visible })}
                    className="text-zinc-500 hover:text-zinc-300"
                  >
                    {pv.visible ? <Eye className="h-3 w-3" /> : <EyeOff className="h-3 w-3" />}
                  </button>
                  <button
                    onClick={() => unpinVolume(pv.path)}
                    className="text-zinc-500 hover:text-red-400"
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

        {/* Volume list */}
        {volumes && volumes.length > 0 && (
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
                >
                  <Pin className="h-3 w-3" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
