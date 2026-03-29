/**
 * VtkViewer — 3D isosurface renderer using vtk.js marching cubes.
 *
 * Fetches raw MRC binary from /api/volumes/raw, parses the 1024-byte header,
 * builds vtkImageData, and renders isosurfaces with vtkImageMarchingCubes.
 *
 * Supports up to 4 simultaneous volumes (pinned) with distinct colors.
 */

import { useEffect, useRef, useCallback, useState } from "react";

// vtk.js imports — use deep paths for tree-shaking
// eslint-disable-next-line @typescript-eslint/no-explicit-any
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
// eslint-disable-next-line @typescript-eslint/no-explicit-any
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
// eslint-disable-next-line @typescript-eslint/no-explicit-any
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
// eslint-disable-next-line @typescript-eslint/no-explicit-any
import vtkImageMarchingCubes from "@kitware/vtk.js/Filters/General/ImageMarchingCubes";
// eslint-disable-next-line @typescript-eslint/no-explicit-any
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData";
// eslint-disable-next-line @typescript-eslint/no-explicit-any
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";

import { Spinner } from "../ui/spinner";

// ---- Types ----

interface VolumeData {
  nx: number;
  ny: number;
  nz: number;
  scalars: Float32Array;
  mean: number;
  std: number;
}

interface VtkPipelineEntry {
  marchingCubes: ReturnType<typeof vtkImageMarchingCubes.newInstance>;
  mapper: ReturnType<typeof vtkMapper.newInstance>;
  actor: ReturnType<typeof vtkActor.newInstance>;
  imageData: ReturnType<typeof vtkImageData.newInstance>;
  volumeData: VolumeData;
}

export interface PinnedVolumeState {
  path: string;
  name: string;
  threshold: number;
  opacity: number;
  visible: boolean;
  colorIndex: number;
}

interface VtkViewerProps {
  /** Active single-volume path (used when no pinned volumes). */
  activeVolume: string | null;
  /** Pinned volumes with per-volume controls. */
  pinnedVolumes: PinnedVolumeState[];
}

// Design system palette: sky-400, rose-400, emerald-400, amber-400
const VOLUME_COLORS: [number, number, number][] = [
  [0.220, 0.741, 0.973],   // sky-400   #38bdf8
  [0.984, 0.443, 0.522],   // rose-400  #fb7185
  [0.204, 0.827, 0.600],   // emerald-400 #34d399
  [0.984, 0.749, 0.141],   // amber-400 #fbbf24
];

const BG_COLOR: [number, number, number, number] = [0.09, 0.09, 0.11, 1.0]; // zinc-950-ish

// ---- MRC parsing ----

function parseMrc(buffer: ArrayBuffer): VolumeData {
  const headerView = new DataView(buffer);
  // NX, NY, NZ are the first 3 int32 values (bytes 0-11)
  const nx = headerView.getInt32(0, true);
  const ny = headerView.getInt32(4, true);
  const nz = headerView.getInt32(8, true);

  const headerSize = 1024;
  const dataBytes = buffer.slice(headerSize);
  // MRC stores float32 data in column-major (Fortran) order: X varies fastest
  const scalars = new Float32Array(dataBytes);

  if (scalars.length !== nx * ny * nz) {
    throw new Error(
      `MRC data size mismatch: expected ${nx * ny * nz} floats, got ${scalars.length}`
    );
  }

  // Compute mean and std for sigma-based thresholding
  let sum = 0;
  let sum2 = 0;
  for (let i = 0; i < scalars.length; i++) {
    sum += scalars[i];
    sum2 += scalars[i] * scalars[i];
  }
  const mean = sum / scalars.length;
  const variance = sum2 / scalars.length - mean * mean;
  const std = Math.sqrt(Math.max(0, variance));

  return { nx, ny, nz, scalars, mean, std };
}

function buildVtkImageData(vol: VolumeData): ReturnType<typeof vtkImageData.newInstance> {
  const imageData = vtkImageData.newInstance();
  // MRC column-major: X varies fastest, so dimensions are [nx, ny, nz]
  imageData.setDimensions(nx_ny_nz(vol));
  imageData.setSpacing([1.0, 1.0, 1.0]);
  imageData.setOrigin([0, 0, 0]);

  const dataArray = vtkDataArray.newInstance({
    name: "Scalars",
    values: vol.scalars,
    numberOfComponents: 1,
  });
  imageData.getPointData().setScalars(dataArray);
  return imageData;
}

function nx_ny_nz(vol: VolumeData): [number, number, number] {
  return [vol.nx, vol.ny, vol.nz];
}

// ---- Component ----

export function VtkViewer({ activeVolume, pinnedVolumes }: VtkViewerProps): React.JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const renderContextRef = useRef<any>(null);
  const pipelinesRef = useRef<Map<string, VtkPipelineEntry>>(new Map());
  const [loading, setLoading] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);
  // Track the last rendered state to avoid redundant re-renders
  const renderedStateRef = useRef<string>("");

  // Initialize vtk.js render window
  useEffect(() => {
    if (!containerRef.current) return;

    const grw = vtkGenericRenderWindow.newInstance({
      background: BG_COLOR as unknown as [number, number, number],
    });
    grw.setContainer(containerRef.current);

    // Size the render window to fill the container
    const rect = containerRef.current.getBoundingClientRect();
    const apiRW = grw.getApiSpecificRenderWindow();
    apiRW.setSize(Math.round(rect.width), Math.round(rect.height));

    renderContextRef.current = grw;

    // Handle resize
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          apiRW.setSize(Math.round(width), Math.round(height));
          grw.resize();
          grw.getRenderWindow().render();
        }
      }
    });
    observer.observe(containerRef.current);

    return () => {
      observer.disconnect();
      // Clean up all pipelines
      for (const pipeline of pipelinesRef.current.values()) {
        pipeline.actor.delete();
        pipeline.mapper.delete();
        pipeline.marchingCubes.delete();
        pipeline.imageData.delete();
      }
      pipelinesRef.current.clear();
      grw.delete();
      renderContextRef.current = null;
    };
  }, []);

  // Fetch and create pipeline for a volume
  const ensurePipeline = useCallback(async (path: string): Promise<VtkPipelineEntry | null> => {
    // Already loaded
    if (pipelinesRef.current.has(path)) {
      return pipelinesRef.current.get(path)!;
    }

    setLoading((prev) => new Set(prev).add(path));
    setError(null);

    try {
      const resp = await fetch(`/api/volumes/raw?path=${encodeURIComponent(path)}`);
      if (!resp.ok) {
        throw new Error(`Failed to fetch volume: ${resp.status} ${resp.statusText}`);
      }
      const buffer = await resp.arrayBuffer();
      const volumeData = parseMrc(buffer);
      const imageData = buildVtkImageData(volumeData);

      const marchingCubes = vtkImageMarchingCubes.newInstance({
        contourValue: volumeData.mean + 3.0 * volumeData.std,
        computeNormals: true,
        mergePoints: true,
      });
      marchingCubes.setInputData(imageData);

      const mapper = vtkMapper.newInstance();
      mapper.setInputConnection(marchingCubes.getOutputPort());

      const actor = vtkActor.newInstance();
      actor.setMapper(mapper);

      const pipeline: VtkPipelineEntry = {
        marchingCubes,
        mapper,
        actor,
        imageData,
        volumeData,
      };

      pipelinesRef.current.set(path, pipeline);
      return pipeline;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      return null;
    } finally {
      setLoading((prev) => {
        const next = new Set(prev);
        next.delete(path);
        return next;
      });
    }
  }, []);

  // Determine which volumes to show
  const volumesToShow = pinnedVolumes.length > 0
    ? pinnedVolumes
    : activeVolume
      ? [{ path: activeVolume, name: "", threshold: 3.0, opacity: 0.8, visible: true, colorIndex: 0 }]
      : [];

  // Sync pipelines with volumesToShow
  useEffect(() => {
    const grw = renderContextRef.current;
    if (!grw) return;

    const renderer = grw.getRenderer();

    // Build a state key to avoid redundant updates
    const stateKey = volumesToShow.map(
      (v) => `${v.path}|${v.threshold}|${v.opacity}|${v.visible}|${v.colorIndex}`
    ).join(";");
    if (stateKey === renderedStateRef.current) return;

    let cancelled = false;

    async function syncPipelines(): Promise<void> {
      // Ensure all needed pipelines are loaded
      const entries: (VtkPipelineEntry | null)[] = await Promise.all(
        volumesToShow.map((v) => ensurePipeline(v.path))
      );
      if (cancelled) return;

      // Remove all actors first
      renderer.removeAllViewProps();

      // Add actors for visible volumes
      for (let i = 0; i < volumesToShow.length; i++) {
        const v = volumesToShow[i];
        const pipeline = entries[i];
        if (!pipeline || !v.visible) continue;

        // Update contour value from sigma threshold
        const contourValue = pipeline.volumeData.mean + v.threshold * pipeline.volumeData.std;
        pipeline.marchingCubes.setContourValue(contourValue);

        // Set color and lighting
        const color = VOLUME_COLORS[v.colorIndex % VOLUME_COLORS.length];
        const prop = pipeline.actor.getProperty();
        prop.setColor(...color);
        prop.setOpacity(v.opacity);
        prop.setAmbient(0.2);
        prop.setDiffuse(0.7);
        prop.setSpecular(0.3);
        prop.setSpecularPower(20);

        renderer.addActor(pipeline.actor);
      }

      renderer.resetCamera();
      grw.getRenderWindow().render();
      renderedStateRef.current = stateKey;
    }

    syncPipelines();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [volumesToShow, ensurePipeline]);

  const isLoading = loading.size > 0;
  const hasVolumes = volumesToShow.length > 0;

  return (
    <div className="relative w-full" style={{ minHeight: 400 }}>
      {/* VTK render container */}
      <div
        ref={containerRef}
        className="w-full rounded-lg"
        style={{ height: 400 }}
      />

      {/* Overlay: loading spinner */}
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/60 rounded-lg">
          <Spinner label="Loading volume..." />
        </div>
      )}

      {/* Overlay: no volume selected */}
      {!hasVolumes && !isLoading && (
        <div className="absolute inset-0 flex items-center justify-center rounded-lg">
          <p className="text-sm text-zinc-500">Select a volume to view in 3D</p>
        </div>
      )}

      {/* Overlay: error */}
      {error && (
        <div className="absolute bottom-2 left-2 right-2 rounded bg-red-900/80 px-3 py-2 text-xs text-red-200">
          {error}
        </div>
      )}
    </div>
  );
}
