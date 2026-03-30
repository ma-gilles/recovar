/**
 * VtkViewer — 3D isosurface renderer using vtk.js marching cubes.
 *
 * Fetches raw MRC binary from /api/volumes/raw, parses the 1024-byte header,
 * builds vtkImageData, and renders isosurfaces with vtkImageMarchingCubes.
 *
 * Supports up to 4 simultaneous volumes (pinned) with distinct colors.
 */

import React, { useEffect, useRef, useCallback, useState } from "react";

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

// Side-effect import: register ALL OpenGL view-node factories needed for
// geometry rendering.  The Geometry profile registers Camera, Renderer,
// Actor, PolyDataMapper, Texture, and others.  Without Camera in particular,
// the PolyDataMapper's render pass fails with "Cannot read properties of
// undefined (reading 'getKeyMatrices')" because the OpenGL camera view-node
// is never created.  Without Renderer/Actor/Mapper the traversal fails with
// "Cannot read properties of undefined (reading 'traverse')".
import "@kitware/vtk.js/Rendering/OpenGL/Profiles/Geometry";

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
  category?: string;
}

interface VtkViewerProps {
  /** Active single-volume path (used when no pinned volumes). */
  activeVolume: string | null;
  /** Sigma threshold for the active volume (when no pinned volumes). */
  activeSigma?: number;
  /** Pinned volumes with per-volume controls. */
  pinnedVolumes: PinnedVolumeState[];
  /** Category of the active volume (e.g. "eigen" for eigenvolumes). */
  activeCategory?: string;
  /**
   * When true, the camera is NOT reset when the volume changes.
   * Used during trajectory playback so the view stays stable.
   */
  preserveCamera?: boolean;
  /**
   * Paths to pre-fetch (e.g. upcoming trajectory frames).
   * These are fetched in the background and cached for instant display.
   */
  prefetchPaths?: string[];
}

// Design system palette: sky-400, rose-400, emerald-400, amber-400
const VOLUME_COLORS: [number, number, number][] = [
  [0.220, 0.741, 0.973],   // sky-400   #38bdf8
  [0.984, 0.443, 0.522],   // rose-400  #fb7185
  [0.204, 0.827, 0.600],   // emerald-400 #34d399
  [0.984, 0.749, 0.141],   // amber-400 #fbbf24
];

const BG_COLOR: [number, number, number, number] = [0.09, 0.09, 0.11, 1.0]; // zinc-950-ish

// Dual-surface colors for eigenvolumes (positive/negative lobes)
const EIGEN_POSITIVE_COLOR: [number, number, number] = [0.3, 0.5, 1.0]; // blue
const EIGEN_NEGATIVE_COLOR: [number, number, number] = [1.0, 0.3, 0.3]; // red

// ---- MRC parsing ----

function parseMrc(buffer: ArrayBuffer): VolumeData {
  if (buffer.byteLength < 1024) {
    throw new Error(
      `MRC file too small: got ${buffer.byteLength} bytes, need at least 1024 for the header`
    );
  }

  const headerView = new DataView(buffer);
  // NX, NY, NZ are the first 3 int32 values (bytes 0-11)
  const nx = headerView.getInt32(0, true);
  const ny = headerView.getInt32(4, true);
  const nz = headerView.getInt32(8, true);
  // MODE at bytes 12-15: 2 = float32
  const mode = headerView.getInt32(12, true);

  // Validate header sanity
  if (nx <= 0 || ny <= 0 || nz <= 0 || nx > 4096 || ny > 4096 || nz > 4096) {
    throw new Error(
      `MRC header dimensions look invalid: NX=${nx}, NY=${ny}, NZ=${nz}. ` +
      `Expected positive values <= 4096.`
    );
  }

  if (mode !== 2) {
    throw new Error(
      `MRC MODE=${mode} is not supported. Only MODE=2 (float32) is supported for 3D viewing. ` +
      `MODE values: 0=int8, 1=int16, 2=float32, 6=uint16, 12=float16.`
    );
  }

  // Check for extended header (NSYMBT at bytes 92-95)
  const nsymbt = headerView.getInt32(92, true);
  const headerSize = 1024 + Math.max(0, nsymbt);

  if (buffer.byteLength < headerSize + nx * ny * nz * 4) {
    throw new Error(
      `MRC file truncated: expected ${headerSize + nx * ny * nz * 4} bytes ` +
      `(header=${headerSize}, data=${nx}x${ny}x${nz} float32), got ${buffer.byteLength}`
    );
  }

  const dataBytes = buffer.slice(headerSize);
  // MRC stores float32 data in column-major (Fortran) order: X varies fastest
  const scalars = new Float32Array(dataBytes);

  if (scalars.length < nx * ny * nz) {
    throw new Error(
      `MRC data size mismatch: expected ${nx * ny * nz} floats, got ${scalars.length}`
    );
  }

  // If there's extra data beyond NX*NY*NZ (e.g. symmetry records), take only what we need
  const trimmedScalars = scalars.length > nx * ny * nz
    ? scalars.slice(0, nx * ny * nz)
    : scalars;

  // Compute mean and std for sigma-based thresholding
  let sum = 0;
  let sum2 = 0;
  for (let i = 0; i < trimmedScalars.length; i++) {
    sum += trimmedScalars[i];
    sum2 += trimmedScalars[i] * trimmedScalars[i];
  }
  const mean = sum / trimmedScalars.length;
  const variance = sum2 / trimmedScalars.length - mean * mean;
  const std = Math.sqrt(Math.max(0, variance));

  console.log(
    `MRC parsed: ${nx}x${ny}x${nz}, mode=${mode}, ` +
    `mean=${mean.toFixed(4)}, std=${std.toFixed(4)}, ` +
    `header=${headerSize}B, total=${buffer.byteLength}B`
  );

  return { nx, ny, nz, scalars: trimmedScalars, mean, std };
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

export function VtkViewer({ activeVolume, activeSigma = 3.0, pinnedVolumes, activeCategory, preserveCamera = false, prefetchPaths }: VtkViewerProps): React.JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const renderContextRef = useRef<any>(null);
  const pipelinesRef = useRef<Map<string, VtkPipelineEntry>>(new Map());
  const [loading, setLoading] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);
  const [webglFailed, setWebglFailed] = useState(false);
  // Track the last rendered state to avoid redundant re-renders
  const renderedStateRef = useRef<string>("");
  // Track whether the camera has been initialized at least once
  const cameraInitializedRef = useRef(false);

  // Initialize vtk.js render window.
  //
  // IMPORTANT: vtkGenericRenderWindow.newInstance() internally calls
  // interactor.initialize() which triggers a render pass (traverseAllPasses).
  // That render pass requires the OpenGL scene-graph view-node factories to
  // be registered (via side-effect imports above).  Without those imports the
  // factory cannot create view nodes for Renderer/Actor/Mapper and the
  // traversal fails with "Cannot read properties of undefined".
  useEffect(() => {
    if (!containerRef.current) return;

    try {
      // Create the generic render window.  The constructor triggers an
      // internal render on a detached canvas — that is harmless as long as
      // the OpenGL view-node factories have been registered (see imports).
      const grw = vtkGenericRenderWindow.newInstance({
        background: BG_COLOR as unknown as [number, number, number],
        listenWindowResize: false, // we use ResizeObserver instead
      });

      // Attach the vtk.js canvas to the DOM container.
      grw.setContainer(containerRef.current);

      // Size the render window to fill the container.  grw.resize() reads
      // the container bounding rect and accounts for devicePixelRatio.
      grw.resize();

      renderContextRef.current = grw;

      // Handle resize
      const observer = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const { width, height } = entry.contentRect;
          if (width > 0 && height > 0) {
            grw.resize();
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
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error("VtkViewer: WebGL initialization failed:", msg);
      setWebglFailed(true);
      setError(
        "3D rendering requires WebGL. Your browser does not support it. Using slice view instead."
      );
      return undefined;
    }
  }, []);

  // Track paths that failed to load to prevent infinite retry loops.
  // When a fetch fails, setError() triggers re-render which creates a new
  // volumesToShow array, firing the effect again.  Without this guard, the
  // fetch retries thousands of times until the browser runs out of resources.
  const failedPathsRef = useRef(new Set<string>());

  // Track in-flight fetches to prevent concurrent duplicate requests.
  // Multiple effect invocations can call ensurePipeline before the first
  // fetch completes, bypassing the failedPathsRef check.
  const inflightRef = useRef(new Map<string, Promise<VtkPipelineEntry | null>>());

  // Fetch and create pipeline for a volume
  const ensurePipeline = useCallback(async (path: string): Promise<VtkPipelineEntry | null> => {
    // Already loaded
    if (pipelinesRef.current.has(path)) {
      return pipelinesRef.current.get(path)!;
    }

    // Already failed — don't retry (avoids infinite loop)
    if (failedPathsRef.current.has(path)) {
      return null;
    }

    // Already being fetched — wait for the existing request
    if (inflightRef.current.has(path)) {
      return inflightRef.current.get(path)!;
    }

    // Wrap the fetch in a promise tracked by inflightRef BEFORE any async
    // work starts, so concurrent callers see it immediately.
    const doFetch = async (): Promise<VtkPipelineEntry | null> => {
    setLoading((prev) => new Set(prev).add(path));
    setError(null);

    try {
      // Step 1: Fetch the raw MRC bytes.
      //
      // We read via response.body (ReadableStream) and manually accumulate
      // chunks instead of using resp.arrayBuffer().  The single-shot
      // arrayBuffer() call can fail with ERR_INSUFFICIENT_RESOURCES on
      // Chromium when the response is large (>~4 MB) and served over an
      // SSH tunnel, because the browser tries to allocate the full buffer
      // before the Content-Length is known.  Streaming avoids this.
      console.log(`VtkViewer: fetching volume: ${path}`);
      const resp = await fetch(`/api/volumes/raw?path=${encodeURIComponent(path)}`, {
        cache: "force-cache",
      });
      if (!resp.ok) {
        const body = await resp.text().catch(() => "");
        throw new Error(`HTTP ${resp.status} ${resp.statusText}: ${body || "Failed to fetch volume"}`);
      }

      let buffer: ArrayBuffer;
      if (resp.body) {
        // Stream-based read: accumulate chunks to avoid single large allocation
        const reader = resp.body.getReader();
        const chunks: Uint8Array[] = [];
        let totalLength = 0;
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          totalLength += value.byteLength;
        }
        const merged = new Uint8Array(totalLength);
        let offset = 0;
        for (const chunk of chunks) {
          merged.set(chunk, offset);
          offset += chunk.byteLength;
        }
        buffer = merged.buffer;
      } else {
        // Fallback for browsers without ReadableStream support
        buffer = await resp.arrayBuffer();
      }
      console.log(`VtkViewer: received ${buffer.byteLength} bytes`);

      if (buffer.byteLength === 0) {
        throw new Error("Server returned empty response (0 bytes)");
      }

      // Step 2: Parse the MRC header and data
      let volumeData: VolumeData;
      try {
        volumeData = parseMrc(buffer);
      } catch (parseErr) {
        const parseMsg = parseErr instanceof Error ? parseErr.message : String(parseErr);
        throw new Error(`MRC parsing failed: ${parseMsg}`);
      }

      // Step 3: Build vtk.js image data
      let imageData: ReturnType<typeof vtkImageData.newInstance>;
      try {
        imageData = buildVtkImageData(volumeData);
      } catch (vtkErr) {
        const vtkMsg = vtkErr instanceof Error ? vtkErr.message : String(vtkErr);
        throw new Error(`vtk.js image data creation failed: ${vtkMsg}`);
      }

      // Step 4: Build the marching cubes pipeline
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
      console.log(`VtkViewer: pipeline ready for ${path} (${volumeData.nx}x${volumeData.ny}x${volumeData.nz})`);
      return pipeline;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error("VtkViewer: pipeline error:", msg, err);
      failedPathsRef.current.add(path);
      setError(msg);
      return null;
    } finally {
      setLoading((prev) => {
        const next = new Set(prev);
        next.delete(path);
        return next;
      });
    }

    };

    const promise = doFetch();
    inflightRef.current.set(path, promise);
    try {
      return await promise;
    } finally {
      inflightRef.current.delete(path);
    }
  }, []);

  // Determine which volumes to show
  const volumesToShow = pinnedVolumes.length > 0
    ? pinnedVolumes
    : activeVolume
      ? [{ path: activeVolume, name: "", threshold: activeSigma, opacity: 0.8, visible: true, colorIndex: 0, category: activeCategory }]
      : [];

  const hasEigenVolume = volumesToShow.some((v) => v.visible && v.category === "eigen");

  // Sync pipelines with volumesToShow
  useEffect(() => {
    const grw = renderContextRef.current;
    if (!grw) return;

    const renderer = grw.getRenderer();

    // Build a state key to avoid redundant updates
    const stateKey = volumesToShow.map(
      (v) => `${v.path}|${v.threshold}|${v.opacity}|${v.visible}|${v.colorIndex}|${v.category ?? ""}`
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

        if (v.category === "eigen") {
          // Eigenvolume: render dual surfaces (positive blue, negative red)
          const posContour = pipeline.volumeData.mean + v.threshold * pipeline.volumeData.std;
          const negContour = pipeline.volumeData.mean - v.threshold * pipeline.volumeData.std;

          // Positive surface (blue)
          pipeline.marchingCubes.setContourValue(posContour);
          const posProp = pipeline.actor.getProperty();
          posProp.setColor(...EIGEN_POSITIVE_COLOR);
          posProp.setOpacity(v.opacity);
          posProp.setAmbient(0.2);
          posProp.setDiffuse(0.7);
          posProp.setSpecular(0.3);
          posProp.setSpecularPower(20);
          renderer.addActor(pipeline.actor);

          // Negative surface (red) — separate pipeline
          const negMC = vtkImageMarchingCubes.newInstance({
            contourValue: negContour,
            computeNormals: true,
            mergePoints: true,
          });
          negMC.setInputData(pipeline.imageData);

          const negMapper = vtkMapper.newInstance();
          negMapper.setInputConnection(negMC.getOutputPort());

          const negActor = vtkActor.newInstance();
          negActor.setMapper(negMapper);
          const negProp = negActor.getProperty();
          negProp.setColor(...EIGEN_NEGATIVE_COLOR);
          negProp.setOpacity(v.opacity);
          negProp.setAmbient(0.2);
          negProp.setDiffuse(0.7);
          negProp.setSpecular(0.3);
          negProp.setSpecularPower(20);
          renderer.addActor(negActor);
        } else {
          // Standard volume: single isosurface
          const contourValue = pipeline.volumeData.mean + v.threshold * pipeline.volumeData.std;
          pipeline.marchingCubes.setContourValue(contourValue);

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
      }

      // Only reset the camera on first load or when not in trajectory mode.
      // During trajectory playback (preserveCamera=true), keep the user's
      // current camera orientation so the animation feels stable.
      if (!preserveCamera || !cameraInitializedRef.current) {
        renderer.resetCamera();
        cameraInitializedRef.current = true;
      }
      try {
        grw.getRenderWindow().render();
      } catch (renderErr) {
        const renderMsg = renderErr instanceof Error ? renderErr.message : String(renderErr);
        console.error("VtkViewer: render() failed:", renderMsg, renderErr);
        setError(`3D render failed: ${renderMsg}`);
        return;
      }
      renderedStateRef.current = stateKey;
    }

    syncPipelines();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [volumesToShow, ensurePipeline, preserveCamera]);

  // Pre-fetch upcoming trajectory frames in the background
  useEffect(() => {
    if (!prefetchPaths || prefetchPaths.length === 0) return;
    for (const path of prefetchPaths) {
      // Fire-and-forget: ensurePipeline caches internally
      ensurePipeline(path);
    }
  }, [prefetchPaths, ensurePipeline]);

  const isLoading = loading.size > 0;
  const hasVolumes = volumesToShow.length > 0;

  // If WebGL failed, show a user-friendly message instead of the canvas.
  if (webglFailed) {
    return (
      <div className="flex items-center justify-center rounded-lg border border-zinc-800 bg-zinc-900 p-8" style={{ minHeight: 400 }}>
        <p className="text-sm text-amber-400">
          3D rendering requires WebGL. Your browser does not support it. Using slice view instead.
        </p>
      </div>
    );
  }

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

      {/* Overlay: eigen legend */}
      {hasEigenVolume && !isLoading && (
        <div className="absolute bottom-2 left-2 flex gap-3 rounded bg-black/70 px-3 py-1.5 text-xs text-zinc-300">
          <span className="flex items-center gap-1">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: `rgb(${EIGEN_POSITIVE_COLOR.map((c) => Math.round(c * 255)).join(",")})` }} />
            Positive
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: `rgb(${EIGEN_NEGATIVE_COLOR.map((c) => Math.round(c * 255)).join(",")})` }} />
            Negative
          </span>
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


// ---------------------------------------------------------------------------
// ErrorBoundary — safety net around VtkViewer for unhandled render errors.
// ---------------------------------------------------------------------------

interface VtkErrorBoundaryProps {
  onWebGLFail: () => void;
  children: React.ReactNode;
}

interface VtkErrorBoundaryState {
  hasError: boolean;
}

export class VtkErrorBoundary extends React.Component<VtkErrorBoundaryProps, VtkErrorBoundaryState> {
  constructor(props: VtkErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(_error: Error): VtkErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo): void {
    console.error("VtkErrorBoundary caught error:", error, info);
    this.props.onWebGLFail();
  }

  render(): React.ReactNode {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center rounded-lg border border-zinc-800 bg-zinc-900 p-8" style={{ minHeight: 400 }}>
          <p className="text-sm text-amber-400">
            3D rendering requires WebGL. Your browser does not support it. Using slice view instead.
          </p>
        </div>
      );
    }
    return this.props.children;
  }
}
