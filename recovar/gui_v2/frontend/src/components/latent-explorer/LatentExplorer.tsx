import { useState, useCallback, useMemo } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Download, Target, Route, Crosshair } from "lucide-react";
import {
  fetchEmbeddings,
  fetchAvailableEmbeddings,
  projectPCA,
} from "../../lib/api/embeddings";
import { createSubset, submitJob } from "../../lib/api/client";
import { Button } from "../ui/button";
import { Select } from "../ui/select";
import { Spinner } from "../ui/spinner";
import { ScatterPanel } from "./ScatterPanel";
import { HistogramPanel } from "./HistogramPanel";
import { SUBSAMPLE_THRESHOLD, DISPLAY_SUBSAMPLE_SIZE } from "../../lib/constants";

interface LatentExplorerProps {
  jobId: string;
  projectId: string;
  resultDir: string;
}

interface MarkerPoint {
  index: number;
  coords: number[]; // full zdim coords
  label: string;
}

export function LatentExplorer({ jobId, projectId, resultDir }: LatentExplorerProps): React.JSX.Element {
  const [zdim, setZdim] = useState<number | null>(null);
  const [pcaAxisX, setPcaAxisX] = useState(0);
  const [pcaAxisY, setPcaAxisY] = useState(1);
  const [colorBy, setColorBy] = useState<"none" | "kmeans">("kmeans");
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(new Set());
  const [markers, setMarkers] = useState<MarkerPoint[]>([]);
  const [showComputeDialog, setShowComputeDialog] = useState(false);

  // Available zdims
  const { data: available } = useQuery({
    queryKey: ["embeddings-available", jobId],
    queryFn: () => fetchAvailableEmbeddings(jobId),
  });

  // Auto-select first zdim
  const effectiveZdim = zdim ?? available?.zdims[0] ?? null;

  // Load embedding data
  const { data: embeddings, isLoading } = useQuery({
    queryKey: ["embeddings", jobId, effectiveZdim],
    queryFn: () => fetchEmbeddings(jobId, effectiveZdim!),
    enabled: effectiveZdim !== null,
  });

  // Subsample for display if needed
  const displayIndices = useMemo(() => {
    if (!embeddings) return null;
    const n = embeddings.meta.n_particles;
    if (n <= SUBSAMPLE_THRESHOLD) return null;
    // Random subsample
    const indices = new Uint32Array(DISPLAY_SUBSAMPLE_SIZE);
    for (let i = 0; i < DISPLAY_SUBSAMPLE_SIZE; i++) {
      indices[i] = Math.floor(Math.random() * n);
    }
    return indices;
  }, [embeddings]);

  // PCA projection
  const pcaPoints = useMemo(() => {
    if (!embeddings) return new Float32Array(0);
    return projectPCA(embeddings.pcaCoords, embeddings.meta.zdim, pcaAxisX, pcaAxisY);
  }, [embeddings, pcaAxisX, pcaAxisY]);

  // UMAP points
  const umapPoints = embeddings?.umapCoords ?? new Float32Array(0);

  // K-means labels for coloring
  const labels = colorBy === "kmeans" ? embeddings?.kmeansLabels : null;

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

  // Lasso handler
  const handleLasso = useCallback((indices: number[]) => {
    setSelectedIndices(new Set(indices));
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
        method: { type: "lasso" as const, plot: "pca", axes: [pcaAxisX, pcaAxisY], vertices: [] },
        indices: finalIndices,
      });
    },
  });

  // Compute state/trajectory
  const computeMutation = useMutation({
    mutationFn: (params: { type: string; z_start?: number[]; z_end?: number[]; latent_points?: number[] }) =>
      submitJob(projectId, params.type, {
        result_dir: resultDir,
        zdim: effectiveZdim,
        ...params,
      }),
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
        <a
          href="/jobs/new"
          className="inline-flex items-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500"
        >
          Run Analyze
        </a>
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
            className="w-24"
            value={colorBy}
            onChange={(e) => setColorBy(e.target.value as "none" | "kmeans")}
          >
            <option value="none">None</option>
            <option value="kmeans">K-means</option>
          </Select>
        </div>

        {embeddings && (
          <span className="text-xs text-zinc-500">
            {embeddings.meta.n_particles.toLocaleString()} particles
            {displayIndices && ` (showing ${DISPLAY_SUBSAMPLE_SIZE.toLocaleString()})`}
          </span>
        )}
      </div>

      {displayIndices && (
        <div className="rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-1.5 text-xs text-amber-400">
          Showing {(DISPLAY_SUBSAMPLE_SIZE / 1e6).toFixed(0)}M of{" "}
          {(embeddings!.meta.n_particles / 1e6).toFixed(1)}M particles. Full dataset used for
          selection export.
        </div>
      )}

      {isLoading ? (
        <div className="flex items-center justify-center py-20">
          <Spinner label="Loading embedding data..." />
        </div>
      ) : embeddings ? (
        <>
          {/* Scatter or histogram panels */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {effectiveZdim === 1 ? (
              <HistogramPanel
                values={pcaPoints}
                labels={labels}
                xLabel="PC1"
                title="PCA (1D)"
                selectedIndices={selectedIndices}
                onPointClick={handlePointClick}
              />
            ) : (
              <ScatterPanel
                points={pcaPoints}
                labels={labels}
                markers={markerPositions}
                xLabel={`PC${pcaAxisX + 1}`}
                yLabel={`PC${pcaAxisY + 1}`}
                title="PCA"
                onLasso={handleLasso}
                onPointClick={handlePointClick}
                selectedIndices={selectedIndices}
                panelId="pca"
              />
            )}
            {umapPoints.length > 0 && (
              <ScatterPanel
                points={umapPoints}
                labels={labels}
                markers={umapMarkerPositions}
                xLabel="UMAP 1"
                yLabel="UMAP 2"
                title="UMAP"
                onLasso={handleLasso}
                onPointClick={handlePointClick}
                selectedIndices={selectedIndices}
                panelId="umap"
              />
            )}
          </div>

          {/* Selection actions */}
          {selectedIndices.size > 0 && (
            <div className="flex items-center gap-3 rounded-md border border-blue-500/30 bg-blue-500/10 px-4 py-3">
              <span className="text-sm text-blue-300">
                {selectedIndices.size.toLocaleString()} particles selected
              </span>
              <div className="ml-auto flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    const name = `subset_${Date.now()}`;
                    subsetMutation.mutate({
                      name,
                      indices: Array.from(selectedIndices),
                      invert: false,
                    });
                  }}
                  loading={subsetMutation.isPending}
                >
                  <Download className="h-3.5 w-3.5" />
                  Export as Subset
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    const name = `subset_inv_${Date.now()}`;
                    subsetMutation.mutate({
                      name,
                      indices: Array.from(selectedIndices),
                      invert: true,
                    });
                  }}
                >
                  Export All Except
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSelectedIndices(new Set())}
                >
                  Clear
                </Button>
              </div>
            </div>
          )}

          {subsetMutation.isSuccess && (
            <div className="rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-1.5 text-xs text-emerald-400">
              Subset exported: {(subsetMutation.data as { name: string }).name} (
              {(subsetMutation.data as { n_particles: number }).n_particles.toLocaleString()}{" "}
              particles)
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
                </div>
              ))}

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
