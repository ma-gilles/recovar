/**
 * Embedding data loader — parses the binary ArrayBuffer format
 * from GET /api/jobs/:id/embeddings?zdim=N
 */

export interface EmbeddingMeta {
  n_particles: number;
  zdim: number;
  has_umap: boolean;
  has_kmeans: boolean;
  n_clusters: number;
}

export interface EmbeddingData {
  meta: EmbeddingMeta;
  pcaCoords: Float32Array; // n x zdim, row-major
  umapCoords: Float32Array | null; // n x 2
  kmeansLabels: Int32Array | null; // n x 1
  kmeansCenters: Float32Array | null; // k x zdim
}

export async function fetchEmbeddings(jobId: string, zdim: number): Promise<EmbeddingData> {
  const resp = await fetch(`/api/jobs/${jobId}/embeddings?zdim=${zdim}`);
  if (!resp.ok) {
    throw new Error(`Failed to load embeddings: ${resp.status}`);
  }

  const metaHeader = resp.headers.get("X-Embedding-Meta");
  if (!metaHeader) {
    throw new Error("Missing X-Embedding-Meta header");
  }
  const meta: EmbeddingMeta = JSON.parse(metaHeader);

  const buffer = await resp.arrayBuffer();
  let offset = 0;
  const n = meta.n_particles;

  // PCA coords: n x zdim float32
  const pcaSize = n * meta.zdim;
  const pcaCoords = new Float32Array(buffer, offset, pcaSize);
  offset += pcaSize * 4;

  // UMAP coords: n x 2 float32 (if present)
  let umapCoords: Float32Array | null = null;
  if (meta.has_umap) {
    umapCoords = new Float32Array(buffer, offset, n * 2);
    offset += n * 2 * 4;
  }

  // K-means labels: n x 1 int32 (if present)
  let kmeansLabels: Int32Array | null = null;
  if (meta.has_kmeans) {
    kmeansLabels = new Int32Array(buffer, offset, n);
    offset += n * 4;
  }

  // K-means centers: k x zdim float32 (if present)
  let kmeansCenters: Float32Array | null = null;
  if (meta.has_kmeans && meta.n_clusters > 0) {
    kmeansCenters = new Float32Array(buffer, offset, meta.n_clusters * meta.zdim);
  }

  return { meta, pcaCoords, umapCoords, kmeansLabels, kmeansCenters };
}

export async function fetchAvailableEmbeddings(
  jobId: string
): Promise<{ zdims: number[]; has_umap: Record<number, boolean> }> {
  const resp = await fetch(`/api/jobs/${jobId}/embeddings/available`);
  if (!resp.ok) throw new Error(`${resp.status}`);
  return resp.json();
}

/**
 * Extract 2D PCA projection for axes (axisX, axisY) from the full zdim coords.
 * Returns interleaved [x0, y0, x1, y1, ...] for scatter plot.
 */
export function projectPCA(
  pcaCoords: Float32Array,
  zdim: number,
  axisX: number,
  axisY: number
): Float32Array {
  const n = pcaCoords.length / zdim;
  const out = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    out[i * 2] = pcaCoords[i * zdim + axisX];
    out[i * 2 + 1] = pcaCoords[i * zdim + axisY];
  }
  return out;
}

export interface RelatedDensityJob {
  id: string;
  output_dir: string;
  pca_dim: number | null;
  density_pkl_path: string;
  created: string;
}

export async function fetchRelatedDensityJobs(jobId: string): Promise<RelatedDensityJob[]> {
  const resp = await fetch(`/api/jobs/${jobId}/related-density`);
  if (!resp.ok) return [];
  return resp.json();
}

export interface DensityData {
  particleDensity: Float32Array;
  centerDensity: Float32Array | null;
  meta: { n_particles: number; n_clusters: number; pca_dim: number; alpha: number; density_job_id: string; };
}

export async function fetchDensityValues(jobId: string, zdim: number, densityJobId: string): Promise<DensityData> {
  const resp = await fetch(`/api/jobs/${jobId}/embeddings/density?zdim=${zdim}&density_job_id=${densityJobId}`);
  if (!resp.ok) throw new Error(`Failed to load density: ${resp.status}`);

  const metaHeader = resp.headers.get("X-Density-Meta");
  if (!metaHeader) throw new Error("Missing X-Density-Meta header");
  const meta = JSON.parse(metaHeader);

  const buffer = await resp.arrayBuffer();
  let offset = 0;

  const particleDensity = new Float32Array(buffer, offset, meta.n_particles);
  offset += meta.n_particles * 4;

  let centerDensity: Float32Array | null = null;
  if (meta.n_clusters > 0) {
    centerDensity = new Float32Array(buffer, offset, meta.n_clusters);
  }

  return { particleDensity, centerDensity, meta };
}
