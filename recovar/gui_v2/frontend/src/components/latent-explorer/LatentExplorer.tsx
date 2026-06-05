import { useState, useCallback, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { clsx } from "clsx";
import { Download, Target, Route, Crosshair, FileSpreadsheet, Play } from "lucide-react";
import {
  fetchEmbeddings,
  fetchAvailableEmbeddings,
  projectPCA,
  fetchRelatedDensityJobs,
  fetchDensityValues,
} from "../../lib/api/embeddings";
import { createSubset, exportSubsetStar, submitJob } from "../../lib/api/client";
import { Button } from "../ui/button";
import { Select } from "../ui/select";
import { Spinner } from "../ui/spinner";
import { ScatterPanel } from "./ScatterPanel";
import { HistogramPanel } from "./HistogramPanel";
import { SelectionToolbar, type SelectionTool } from "./SelectionToolbar";
import { SubsetProvenanceButton, SubsetProvenanceSummary } from "./SubsetProvenance";

/**
 * Grid-based density estimation for 2D point data.
 * Bins points into a grid, smooths with a 3x3 kernel, then maps
 * each point's cell density to [0,1] for color mapping.
 * @param points Interleaved xy coords: [x0,y0,x1,y1,...]
 * @returns Float32Array of per-point density in [0,1]
 */
function computeGridDensity(points: Float32Array): Float32Array {
  const n = points.length / 2;
  if (n === 0) return new Float32Array(0);

  // Choose grid resolution: ~64 bins per axis (good balance of detail vs speed)
  const GRID = 64;

  // Compute bounds
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = points[i * 2], y = points[i * 2 + 1];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  const invBinX = GRID / rangeX;
  const invBinY = GRID / rangeY;

  // Bin the points into a 2D grid
  const grid = new Float32Array(GRID * GRID);
  const cellForPoint = new Uint16Array(n * 2); // store (gx, gy) per point
  for (let i = 0; i < n; i++) {
    const gx = Math.min(Math.floor((points[i * 2] - minX) * invBinX), GRID - 1);
    const gy = Math.min(Math.floor((points[i * 2 + 1] - minY) * invBinY), GRID - 1);
    grid[gy * GRID + gx]++;
    cellForPoint[i * 2] = gx;
    cellForPoint[i * 2 + 1] = gy;
  }

  // Smooth with a 3x3 averaging kernel for softer density
  const smoothed = new Float32Array(GRID * GRID);
  for (let gy = 0; gy < GRID; gy++) {
    for (let gx = 0; gx < GRID; gx++) {
      let sum = 0;
      let count = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = gx + dx, ny = gy + dy;
          if (nx >= 0 && nx < GRID && ny >= 0 && ny < GRID) {
            sum += grid[ny * GRID + nx];
            count++;
          }
        }
      }
      smoothed[gy * GRID + gx] = sum / count;
    }
  }

  // Find the max smoothed density for normalization
  let maxDensity = 0;
  for (let i = 0; i < smoothed.length; i++) {
    if (smoothed[i] > maxDensity) maxDensity = smoothed[i];
  }
  if (maxDensity === 0) maxDensity = 1;

  // Map each point to its normalized density
  const result = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const gx = cellForPoint[i * 2];
    const gy = cellForPoint[i * 2 + 1];
    result[i] = smoothed[gy * GRID + gx] / maxDensity;
  }

  return result;
}

interface LatentExplorerProps {
  jobId: string;
  projectId: string;
  resultDir: string;
  /** Path to the original particles .star file (from the pipeline job params). */
  particlesStar?: string | null;
  /** The zdim the analyze job was run with (for accurate UMAP title). */
  analyzeZdim?: number | null;
}

interface MarkerPoint {
  index: number;
  coords: number[]; // full zdim coords
  label: string;
}

export function LatentExplorer({ jobId, projectId, resultDir, particlesStar, analyzeZdim }: LatentExplorerProps): React.JSX.Element {
  const [zdim, setZdim] = useState<number | null>(null);
  const [pcaAxisX, setPcaAxisX] = useState(0);
  const [pcaAxisY, setPcaAxisY] = useState(1);
  const [colorBy, setColorBy] = useState<"none" | "kmeans" | "density" | "deconvolved">("kmeans");
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(new Set());
  const [selectionTool, setSelectionTool] = useState<SelectionTool | null>(null);
  const [markers, setMarkers] = useState<MarkerPoint[]>([]);
  const [showComputeDialog, setShowComputeDialog] = useState(false);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [useDensityGuided, setUseDensityGuided] = useState(false);
  const [liveSelectionCount, setLiveSelectionCount] = useState<number | null>(null);
  // Which .star export is in flight: false = "Export .star", true = "Export All Except".
  const [starExportInvert, setStarExportInvert] = useState<boolean | null>(null);
  const queryClient = useQueryClient();

  // Available zdims
  const { data: available } = useQuery({
    queryKey: ["embeddings-available", jobId],
    queryFn: () => fetchAvailableEmbeddings(jobId),
  });

  // Auto-select zdim: prefer 4 if available, otherwise first
  const effectiveZdim = zdim ?? (available?.zdims.includes(4) ? 4 : available?.zdims[0]) ?? null;

  // The zdim at which UMAP/k-means were computed (from the analyze job)
  const umapSourceZdim = analyzeZdim ?? null;

  // Load embedding data
  const { data: embeddings, isLoading, error: embeddingError } = useQuery({
    queryKey: ["embeddings", jobId, effectiveZdim],
    queryFn: () => fetchEmbeddings(jobId, effectiveZdim!),
    enabled: effectiveZdim !== null,
  });

  // Related density jobs for this analyze job
  const { data: relatedDensityJobs } = useQuery({
    queryKey: ["related-density", jobId],
    queryFn: () => fetchRelatedDensityJobs(jobId),
    enabled: !!jobId,
  });

  const activeDensityJob = relatedDensityJobs?.[0] ?? null;

  // Deconvolved density values from a linked density job
  const { data: deconvolvedDensity } = useQuery({
    queryKey: ["density-values", jobId, effectiveZdim, activeDensityJob?.id],
    queryFn: () => fetchDensityValues(jobId, effectiveZdim!, activeDensityJob!.id),
    enabled: effectiveZdim !== null && !!activeDensityJob,
  });

  // PCA projection
  const pcaPoints = useMemo(() => {
    if (!embeddings) return new Float32Array(0);
    return projectPCA(embeddings.pcaCoords, embeddings.meta.zdim, pcaAxisX, pcaAxisY);
  }, [embeddings, pcaAxisX, pcaAxisY]);

  // UMAP points
  const umapPoints = embeddings?.umapCoords ?? new Float32Array(0);

  // K-means labels for coloring
  const labels = colorBy === "kmeans" ? embeddings?.kmeansLabels : null;

  // Grid-based density estimation for density coloring
  const pcaDensity = useMemo(() => {
    if (colorBy !== "density" || !pcaPoints || pcaPoints.length < 4) return null;
    return computeGridDensity(pcaPoints);
  }, [colorBy, pcaPoints]);

  const umapDensity = useMemo(() => {
    if (colorBy !== "density" || !umapPoints || umapPoints.length < 4) return null;
    return computeGridDensity(umapPoints);
  }, [colorBy, umapPoints]);

  // Density values to pass to scatter panels (grid estimation vs deconvolved)
  const pcaDensityForScatter = useMemo(() => {
    if (colorBy === "density") return pcaDensity;
    if (colorBy === "deconvolved" && deconvolvedDensity) return deconvolvedDensity.particleDensity;
    return null;
  }, [colorBy, pcaDensity, deconvolvedDensity]);

  const umapDensityForScatter = useMemo(() => {
    if (colorBy === "density") return umapDensity;
    if (colorBy === "deconvolved" && deconvolvedDensity) return deconvolvedDensity.particleDensity;
    return null;
  }, [colorBy, umapDensity, deconvolvedDensity]);

  // Marker positions for both plots
  const markerPositions = useMemo(() => {
    if (!embeddings || markers.length === 0) return null;
    // PCA marker positions
    const pcaMarkers = new Float32Array(markers.length * 2);
    for (let i = 0; i < markers.length; i++) {
      pcaMarkers[i * 2] = markers[i].coords[pcaAxisX] ?? 0;
      pcaMarkers[i * 2 + 1] = markers[i].coords[pcaAxisY] ?? 0;
    }
    return pcaMarkers;
  }, [markers, pcaAxisX, pcaAxisY, embeddings]);

  // UMAP marker positions — need to look up from particle index
  const umapMarkerPositions = useMemo(() => {
    if (!embeddings?.umapCoords || markers.length === 0) return null;
    const out = new Float32Array(markers.length * 2);
    for (let i = 0; i < markers.length; i++) {
      const idx = markers[i].index;
      out[i * 2] = embeddings.umapCoords[idx * 2];
      out[i * 2 + 1] = embeddings.umapCoords[idx * 2 + 1];
    }
    return out;
  }, [markers, embeddings]);

  // K-means center positions projected onto PCA and UMAP
  const pcaCenterPositions = useMemo(() => {
    if (!embeddings?.kmeansCenters || !embeddings.meta.zdim) return null;
    const nClusters = embeddings.meta.n_clusters ?? 0;
    if (nClusters === 0) return null;
    const out = new Float32Array(nClusters * 2);
    for (let c = 0; c < nClusters; c++) {
      out[c * 2] = embeddings.kmeansCenters[c * embeddings.meta.zdim + pcaAxisX] ?? 0;
      out[c * 2 + 1] = embeddings.kmeansCenters[c * embeddings.meta.zdim + pcaAxisY] ?? 0;
    }
    return out;
  }, [embeddings, pcaAxisX, pcaAxisY]);

  const umapCenterPositions = useMemo(() => {
    if (!embeddings?.umapCoords || !embeddings?.kmeansLabels || !embeddings?.kmeansCenters) return null;
    const nClusters = embeddings.meta.n_clusters ?? 0;
    if (nClusters === 0) return null;
    // For UMAP, find the particle closest to each k-means center and use its UMAP position
    const nParticles = embeddings.kmeansLabels.length;
    const zdim = embeddings.meta.zdim;
    const out = new Float32Array(nClusters * 2);
    for (let c = 0; c < nClusters; c++) {
      let bestIdx = 0;
      let bestDist = Infinity;
      for (let i = 0; i < nParticles; i++) {
        if (embeddings.kmeansLabels[i] !== c) continue;
        let dist = 0;
        for (let d = 0; d < zdim; d++) {
          const diff = embeddings.pcaCoords[i * zdim + d] - embeddings.kmeansCenters[c * zdim + d];
          dist += diff * diff;
        }
        if (dist < bestDist) {
          bestDist = dist;
          bestIdx = i;
        }
      }
      out[c * 2] = embeddings.umapCoords[bestIdx * 2];
      out[c * 2 + 1] = embeddings.umapCoords[bestIdx * 2 + 1];
    }
    return out;
  }, [embeddings]);

  // Click handler — select k-means center or particle
  const handlePointClick = useCallback(
    (index: number, _coords: [number, number]) => {
      if (!embeddings) return;
      const fullCoords: number[] = [];
      for (let d = 0; d < embeddings.meta.zdim; d++) {
        fullCoords.push(embeddings.pcaCoords[index * embeddings.meta.zdim + d]);
      }

      // Check if this is a k-means center
      let label = `Particle ${index}`;
      if (embeddings.kmeansLabels && embeddings.kmeansCenters) {
        const clusterLabel = embeddings.kmeansLabels[index];
        // Check if index is near a k-means center
        const centerStart = clusterLabel * embeddings.meta.zdim;
        let isCenterPoint = true;
        for (let d = 0; d < embeddings.meta.zdim; d++) {
          if (
            Math.abs(fullCoords[d] - embeddings.kmeansCenters[centerStart + d]) > 0.01
          ) {
            isCenterPoint = false;
            break;
          }
        }
        if (isCenterPoint) {
          label = `Cluster ${clusterLabel}`;
        }
      }

      const newMarker: MarkerPoint = { index, coords: fullCoords, label };

      if (markers.length === 0) {
        setMarkers([newMarker]);
        setShowComputeDialog(true);
      } else if (markers.length === 1) {
        setMarkers([markers[0], newMarker]);
        setShowComputeDialog(true);
      } else {
        // Replace all, start fresh
        setMarkers([newMarker]);
        setShowComputeDialog(true);
      }
    },
    [embeddings, markers]
  );

  // Selection handler (lasso, rectangle, polygon all produce the same output)
  const handleSelect = useCallback((indices: number[]) => {
    setSelectedIndices(new Set(indices));
  }, []);

  const handleClearSelection = useCallback(() => {
    setSelectedIndices(new Set());
  }, []);

  // Subset export mutation
  const subsetMutation = useMutation({
    mutationFn: (params: { name: string; indices: number[]; invert: boolean }) => {
      const finalIndices = params.invert
        ? Array.from({ length: embeddings!.meta.n_particles }, (_, i) => i).filter(
            (i) => !new Set(params.indices).has(i)
          )
        : params.indices;
      return createSubset({
        project_id: projectId,
        name: params.name,
        source_job_id: jobId,
        zdim: effectiveZdim ?? undefined,
        method: { type: selectionTool ?? "lasso", plot: "pca", axes: [pcaAxisX, pcaAxisY], vertices: [] },
        indices: finalIndices,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["subsets", projectId] });
      queryClient.invalidateQueries({ queryKey: ["project", projectId] });
    },
  });

  // Combined: create subset + export .star in one click
  const oneClickStarMutation = useMutation({
    mutationFn: async (params: { indices: number[]; invert: boolean }) => {
      const finalIndices = params.invert
        ? Array.from({ length: embeddings!.meta.n_particles }, (_, i) => i).filter(
            (i) => !new Set(params.indices).has(i)
          )
        : params.indices;
      const subset = await createSubset({
        project_id: projectId,
        name: `subset_${Date.now()}`,
        source_job_id: jobId,
        zdim: effectiveZdim ?? undefined,
        method: { type: selectionTool ?? "lasso", plot: "pca", axes: [pcaAxisX, pcaAxisY], vertices: [] },
        indices: finalIndices,
      });
      const star = await exportSubsetStar(subset.id, particlesStar!);
      return { subset, star };
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["subsets", projectId] });
      queryClient.invalidateQueries({ queryKey: ["project", projectId] });
    },
    onSettled: () => {
      setStarExportInvert(null);
    },
  });

  // Compute state/trajectory
  const computeMutation = useMutation({
    mutationFn: (params: { type: string; z_start?: number[]; z_end?: number[]; latent_points?: number[]; density?: string }) =>
      submitJob(projectId, params.type, {
        result_dir: resultDir,
        zdim: effectiveZdim,
        ...params,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["project", projectId] });
    },
  });

  if (!available) {
    return (
      <div className="flex items-center justify-center py-20">
        <Spinner label="Loading embeddings..." />
      </div>
    );
  }

  if (available.zdims.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-4 py-20 text-center">
        <p className="text-zinc-400">
          No analysis results found. Run <strong>Analyze</strong> on this pipeline output first.
        </p>
        <Link
          to="/jobs/new"
          search={{
            type: "analyze",
            result_dir: resultDir,
            density: undefined,
            input: undefined,
            particles: undefined,
            params: undefined,
          }}
          className="inline-flex items-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500"
        >
          Run Analyze
        </Link>
      </div>
    );
  }

  const axisOptions = Array.from(
    { length: effectiveZdim ?? 0 },
    (_, i) => i
  );

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-1">
          <span className="text-xs text-zinc-500">zdim:</span>
          <Select
            className="w-20"
            value={effectiveZdim?.toString() ?? ""}
            onChange={(e) => setZdim(parseInt(e.target.value))}
          >
            {available.zdims.map((z) => (
              <option key={z} value={z}>
                {z}
              </option>
            ))}
          </Select>
        </div>

        <div className="flex items-center gap-1">
          <span className="text-xs text-zinc-500">X:</span>
          <Select
            className="w-20"
            value={pcaAxisX.toString()}
            onChange={(e) => setPcaAxisX(parseInt(e.target.value))}
          >
            {axisOptions.map((i) => (
              <option key={i} value={i}>
                PC{i + 1}
              </option>
            ))}
          </Select>
          <span className="text-xs text-zinc-500">Y:</span>
          <Select
            className="w-20"
            value={pcaAxisY.toString()}
            onChange={(e) => setPcaAxisY(parseInt(e.target.value))}
          >
            {axisOptions.map((i) => (
              <option key={i} value={i}>
                PC{i + 1}
              </option>
            ))}
          </Select>
        </div>

        <div className="flex items-center gap-1">
          <span className="text-xs text-zinc-500">Color:</span>
          <Select
            className="w-32"
            value={colorBy}
            onChange={(e) => setColorBy(e.target.value as typeof colorBy)}
          >
            <option value="none">None</option>
            <option value="kmeans">K-means</option>
            <option value="density">Point density</option>
            {activeDensityJob && (
              <option value="deconvolved">
                Deconvolved ({activeDensityJob.output_dir.split("/").pop()})
              </option>
            )}
          </Select>
        </div>

        {embeddings && (
          <span className="text-xs text-zinc-500">
            {embeddings.meta.n_particles.toLocaleString()} particles
          </span>
        )}
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-20">
          <Spinner label="Loading embedding data..." />
        </div>
      ) : embeddingError ? (
        <div className="rounded-md border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">
          Failed to load embedding data for zdim={effectiveZdim}: {String(embeddingError instanceof Error ? embeddingError.message : embeddingError)}
        </div>
      ) : embeddings ? (
        <>
          {/* Selection toolbar */}
          {effectiveZdim !== 1 && (
            <SelectionToolbar
              activeTool={selectionTool}
              onToolChange={setSelectionTool}
              onClearSelection={handleClearSelection}
              hasSelection={selectedIndices.size > 0}
              liveSelectionCount={liveSelectionCount}
            />
          )}

          {/* Scatter or histogram panels */}
          <div className={clsx(
            "grid grid-cols-1 gap-4",
            umapPoints.length > 0 && "lg:grid-cols-2"
          )}>
            {effectiveZdim === 1 ? (
              <HistogramPanel
                values={pcaPoints}
                labels={labels}
                xLabel="PC1"
                title="PCA Projection (1D)"
                selectedIndices={selectedIndices}
                onPointClick={handlePointClick}
              />
            ) : (
              <ScatterPanel
                points={pcaPoints}
                labels={labels}
                densityValues={pcaDensityForScatter}
                markers={markerPositions}
                centerPositions={pcaCenterPositions}
                centerDensityValues={colorBy === "deconvolved" ? deconvolvedDensity?.centerDensity ?? undefined : undefined}
                xLabel={`PC${pcaAxisX + 1}`}
                yLabel={`PC${pcaAxisY + 1}`}
                title="PCA Projection"
                onSelect={handleSelect}
                onPointClick={handlePointClick}
                selectedIndices={selectedIndices}
                panelId="pca"
                activeTool={selectionTool}
                hoveredIndex={hoveredIndex}
                onHover={setHoveredIndex}
                onLiveSelectionCount={setLiveSelectionCount}
              />
            )}
            {umapPoints.length > 0 && (
              <ScatterPanel
                points={umapPoints}
                labels={labels}
                densityValues={umapDensityForScatter}
                markers={umapMarkerPositions}
                centerPositions={umapCenterPositions}
                centerDensityValues={colorBy === "deconvolved" ? deconvolvedDensity?.centerDensity ?? undefined : undefined}
                xLabel="UMAP 1"
                yLabel="UMAP 2"
                title={umapSourceZdim != null && umapSourceZdim !== effectiveZdim
                  ? `UMAP Projection (from zdim=${umapSourceZdim} analysis)`
                  : `UMAP Projection (zdim=${effectiveZdim})`}
                onSelect={handleSelect}
                onPointClick={handlePointClick}
                selectedIndices={selectedIndices}
                panelId="umap"
                activeTool={selectionTool}
                hoveredIndex={hoveredIndex}
                onHover={setHoveredIndex}
                onLiveSelectionCount={setLiveSelectionCount}
              />
            )}
          </div>

          {umapPoints.length === 0 && (
            <div className="rounded-md border border-zinc-800 bg-black/50 px-4 py-3">
              <p className="text-sm text-zinc-500">
                No UMAP embedding available.{" "}
                <Link
                  to="/jobs/new"
                  search={{
                    type: "analyze",
                    result_dir: resultDir,
                    density: undefined,
                    input: undefined,
                    particles: undefined,
                    params: undefined,
                  }}
                  className="text-blue-400 hover:text-blue-300 hover:underline"
                >
                  Run Analyze to generate UMAP embedding
                </Link>
              </p>
            </div>
          )}

          {colorBy === "kmeans" && embeddings && !embeddings.kmeansLabels && (
            <div className="rounded-md border border-zinc-800 bg-black/50 px-4 py-3">
              <p className="text-sm text-zinc-500">
                No k-means clustering available.{" "}
                <Link
                  to="/jobs/new"
                  search={{
                    type: "analyze",
                    result_dir: resultDir,
                    density: undefined,
                    input: undefined,
                    particles: undefined,
                    params: undefined,
                  }}
                  className="text-blue-400 hover:text-blue-300 hover:underline"
                >
                  Run Analyze to generate clusters
                </Link>
              </p>
            </div>
          )}

          {/* Selection actions */}
          {selectedIndices.size > 0 && (
            <div className="rounded-md border border-blue-500/30 bg-blue-500/10 px-4 py-3 space-y-2">
              <div className="flex items-center gap-3">
                <span className="text-sm text-blue-300">
                  {selectedIndices.size.toLocaleString()} particles selected
                </span>
                <div className="ml-auto flex gap-2">
                  {/* Primary action: export filtered .star in one click */}
                  {particlesStar && particlesStar.endsWith(".star") && (
                    <Button
                      size="sm"
                      onClick={() => {
                        setStarExportInvert(false);
                        oneClickStarMutation.mutate({
                          indices: Array.from(selectedIndices),
                          invert: false,
                        });
                      }}
                      loading={oneClickStarMutation.isPending && starExportInvert === false}
                      disabled={oneClickStarMutation.isPending}
                    >
                      <FileSpreadsheet className="h-3.5 w-3.5" />
                      Export .star
                    </Button>
                  )}
                  {particlesStar && particlesStar.endsWith(".star") && (
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => {
                        setStarExportInvert(true);
                        oneClickStarMutation.mutate({
                          indices: Array.from(selectedIndices),
                          invert: true,
                        });
                      }}
                      loading={oneClickStarMutation.isPending && starExportInvert === true}
                      disabled={oneClickStarMutation.isPending}
                    >
                      Export All Except
                    </Button>
                  )}
                  {/* Fallback: .ind export when no .star available */}
                  {(!particlesStar || !particlesStar.endsWith(".star")) && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        subsetMutation.mutate({
                          name: `subset_${Date.now()}`,
                          indices: Array.from(selectedIndices),
                          invert: false,
                        });
                      }}
                      loading={subsetMutation.isPending}
                    >
                      <Download className="h-3.5 w-3.5" />
                      Export .ind
                    </Button>
                  )}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setSelectedIndices(new Set())}
                  >
                    Clear
                  </Button>
                </div>
              </div>
              {!particlesStar && (
                <p className="text-xs text-zinc-500">
                  No source .star file found. Export creates an index file (.ind) with selected particle indices.
                </p>
              )}
            </div>
          )}

          {/* One-click .star export result */}
          {oneClickStarMutation.isSuccess && (
            <div className="rounded-md border border-emerald-500/30 bg-emerald-500/10 px-4 py-3 space-y-2">
              <div className="flex items-center gap-2 text-sm text-emerald-400">
                <FileSpreadsheet className="h-4 w-4" />
                Exported {oneClickStarMutation.data.star.n_particles.toLocaleString()} particles
              </div>
              <p className="text-xs text-zinc-400 font-mono truncate" title={oneClickStarMutation.data.star.path}>
                {oneClickStarMutation.data.star.path}
              </p>
              <div className="flex items-center gap-3 pt-1">
                <Link
                  to="/jobs/new"
                  search={{
                    type: "pipeline",
                    particles: oneClickStarMutation.data.star.path,
                    result_dir: undefined,
                    density: undefined,
                    input: undefined,
                    params: undefined,
                  }}
                  className="inline-flex items-center gap-1.5 text-sm font-medium text-blue-400 hover:text-blue-300"
                >
                  <Play className="h-3.5 w-3.5" />
                  Rerun pipeline with this subset
                </Link>
                <SubsetProvenanceButton
                  subsetId={oneClickStarMutation.data.subset.id}
                />
              </div>
            </div>
          )}
          {oneClickStarMutation.isError && (
            <div className="rounded-md border border-red-500/30 bg-red-500/10 px-4 py-2 text-xs text-red-400">
              Export failed: {(oneClickStarMutation.error as Error).message}
            </div>
          )}

          {/* Fallback .ind export result */}
          {subsetMutation.isSuccess && !oneClickStarMutation.isSuccess && (
            <div className="rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-400">
              <SubsetProvenanceSummary
                name={(subsetMutation.data as { name: string }).name}
                nParticles={(subsetMutation.data as { n_particles: number }).n_particles}
                zdim={effectiveZdim}
                selectionTool={selectionTool}
                sourceJobId={jobId}
              />
            </div>
          )}

          {/* Compute dialog */}
          {showComputeDialog && markers.length > 0 && (
            <div className="rounded-lg border border-zinc-700 bg-zinc-900 p-4 space-y-3">
              <div className="flex items-center gap-2">
                <Crosshair className="h-4 w-4 text-blue-400" />
                <span className="text-sm font-medium">
                  {markers.length === 1 ? "Compute Volume" : "Compute Actions"}
                </span>
              </div>

              {markers.map((m, i) => (
                <div key={i} className="text-xs text-zinc-400">
                  <span className="font-medium text-zinc-300">{m.label}:</span>{" "}
                  <span className="font-mono">
                    [{m.coords.map((c) => c.toFixed(4)).join(", ")}]
                  </span>
                  {deconvolvedDensity && (
                    <span className="ml-2 text-amber-400">
                      density: {deconvolvedDensity.particleDensity[m.index]?.toFixed(4) ?? "?"}
                    </span>
                  )}
                </div>
              ))}

              {markers.length === 2 && activeDensityJob && (
                <label className="flex items-center gap-2 text-xs text-zinc-400">
                  <input
                    type="checkbox"
                    checked={useDensityGuided}
                    onChange={(e) => setUseDensityGuided(e.target.checked)}
                    className="rounded border-zinc-600 bg-zinc-800 text-blue-500 focus:ring-blue-500"
                  />
                  <span>Use density-guided path</span>
                  <span className="truncate font-mono text-zinc-600" title={activeDensityJob.density_pkl_path}>
                    ({activeDensityJob.output_dir.split("/").pop()})
                  </span>
                </label>
              )}

              <div className="flex gap-2">
                {markers.length >= 1 && (
                  <Button
                    size="sm"
                    onClick={() => {
                      computeMutation.mutate({
                        type: "compute_state",
                        latent_points: markers[0].coords,
                      });
                      setShowComputeDialog(false);
                    }}
                    loading={computeMutation.isPending}
                  >
                    <Target className="h-3.5 w-3.5" />
                    Compute State
                  </Button>
                )}
                {markers.length === 2 && (
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => {
                      computeMutation.mutate({
                        type: "compute_trajectory",
                        z_start: markers[0].coords,
                        z_end: markers[1].coords,
                        ...(useDensityGuided && activeDensityJob
                          ? { density: activeDensityJob.density_pkl_path }
                          : {}),
                      });
                      setShowComputeDialog(false);
                    }}
                  >
                    <Route className="h-3.5 w-3.5" />
                    Compute Trajectory
                  </Button>
                )}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setMarkers([]);
                    setShowComputeDialog(false);
                  }}
                >
                  Clear
                </Button>
              </div>

              {computeMutation.isSuccess && (
                <div className="text-xs text-emerald-400">
                  Job submitted: {(computeMutation.data as { id: string }).id}
                </div>
              )}
              {computeMutation.isError && (
                <div className="text-xs text-red-400">
                  {(computeMutation.error as Error).message}
                </div>
              )}
            </div>
          )}
        </>
      ) : null}
    </div>
  );
}
