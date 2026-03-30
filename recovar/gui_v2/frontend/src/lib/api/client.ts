/**
 * REST API client for the recovar GUI backend.
 *
 * All REST calls go through these typed functions.
 * WebSocket and binary endpoints (volumes, embeddings) are in separate files.
 */

const BASE = "/api";

/**
 * Custom error class that preserves the HTTP status code so callers
 * (e.g. TanStack Query retry logic) can distinguish 4xx from 5xx.
 */
export class ApiError extends Error {
  status: number;
  constructor(status: number, body: string) {
    super(`${status}: ${body}`);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(
  path: string,
  options?: RequestInit
): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!resp.ok) {
    const body = await resp.text();
    throw new ApiError(resp.status, body);
  }
  if (resp.status === 204) return undefined as T;
  return resp.json();
}

// --- Types ---

export interface Project {
  id: string;
  path: string;
  name: string;
  created: string;
}

export interface ProjectDetail extends Project {
  jobs: JobSummary[];
  disk_usage_bytes: number;
  disk_usage_total: number;
}

export interface JobSummary {
  id: string;
  type: string;
  status: string;
  output_dir: string;
  created: string;
  completed?: string | null;
  slurm_id?: string | null;
  error?: string | null;
}

export interface JobDetail extends JobSummary {
  project_id: string;
  params?: Record<string, unknown> | null;
  handle?: string | null;
  parent_jobs?: string[] | null;
  execution_mode: string;
  execution_summary: string;
}

export interface VolumeEntry {
  name: string;
  path: string;
  category: string;
  size_bytes: number;
}

export interface PlotEntry {
  name: string;
  path: string;
}

export interface SuggestedNext {
  type: string;
  label: string;
  prefilled_params: Record<string, unknown>;
}

export interface FileEntry {
  name: string;
  path: string;
  is_dir: boolean;
  size: number;
  modified: string;
  type: string;
}

export interface SubsetEntry {
  id: string;
  name: string;
  n_particles: number;
  source_job_id?: string | null;
  method?: Record<string, unknown> | null;
  created: string;
}

export interface SystemInfo {
  slurm_available: boolean;
  executor_mode: string;
  recovar_version: string;
  gpu_count: number;
  hostname: string;
  disk?: { path: string; total: number; used: number; free: number } | null;
}

export interface SlurmDefaults {
  partition: string;
  account: string;
  gpus: number;
  cpus: number;
  memory: string;
  time: string;
}

export interface SbatchScript {
  script: string;
  source: string;
}

// --- Projects ---

export function createProject(path: string, name: string): Promise<Project> {
  return request("/projects", {
    method: "POST",
    body: JSON.stringify({ path, name }),
  });
}

export function getProject(id: string): Promise<ProjectDetail> {
  return request(`/projects/${id}`);
}

export function scanProject(
  projectId: string,
  scanPath: string
): Promise<{ imported: { id: string; type: string; status: string; output_dir: string; legacy: boolean }[]; hint?: string | null }> {
  return request(`/projects/${projectId}/scan`, {
    method: "POST",
    body: JSON.stringify({ scan_path: scanPath }),
  });
}

// --- Jobs ---

export function submitJob(
  projectId: string,
  type: string,
  params: Record<string, unknown>
): Promise<{ id: string; type: string; status: string; created: string; handle?: string }> {
  return request("/jobs", {
    method: "POST",
    body: JSON.stringify({ project_id: projectId, type, params }),
  });
}

export function getJob(id: string): Promise<JobDetail> {
  return request(`/jobs/${id}`);
}

export function cancelJob(id: string): Promise<{ status: string }> {
  return request(`/jobs/${id}/cancel`, { method: "POST" });
}

export interface ReconcileResult {
  id: string;
  previous_status: string;
  new_status: string;
  changed: boolean;
  error?: string | null;
}

export function reconcileJob(id: string): Promise<ReconcileResult> {
  return request(`/jobs/${id}/reconcile`, { method: "POST" });
}

export function deleteJob(id: string): Promise<void> {
  return request(`/jobs/${id}`, { method: "DELETE" });
}

export function getJobVolumes(id: string): Promise<VolumeEntry[]> {
  return request(`/jobs/${id}/volumes`);
}

export function getJobPlots(id: string): Promise<PlotEntry[]> {
  return request(`/jobs/${id}/plots`);
}

export function getSuggestedNext(id: string): Promise<SuggestedNext[]> {
  return request(`/jobs/${id}/suggested-next`);
}

// --- Files ---

export function browseFiles(path: string): Promise<FileEntry[]> {
  return request(`/files/browse?path=${encodeURIComponent(path)}`);
}

export function validateStar(
  path: string
): Promise<{ valid: boolean | null; n_particles?: number; box_size?: number; columns?: string[]; error?: string }> {
  return request("/files/validate-star", {
    method: "POST",
    body: JSON.stringify({ path }),
  });
}

export function validateMrc(
  path: string
): Promise<{ valid: boolean | null; shape?: number[]; voxel_size?: number; error?: string }> {
  return request("/files/validate-mrc", {
    method: "POST",
    body: JSON.stringify({ path }),
  });
}

// --- Volumes ---

export function getVolumeInfo(
  path: string
): Promise<{ shape: number[]; voxel_size: number; min: number; max: number; mean: number }> {
  return request(`/volumes/info?path=${encodeURIComponent(path)}`);
}

// --- Subsets ---

export function createSubset(data: {
  project_id: string;
  name: string;
  source_job_id?: string;
  zdim?: number;
  method?: Record<string, unknown>;
  indices: number[];
}): Promise<{ id: string; name: string; path: string; n_particles: number }> {
  return request("/subsets", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export function listSubsets(projectId: string): Promise<SubsetEntry[]> {
  return request(`/subsets?project_id=${projectId}`);
}

export function deleteSubset(id: string): Promise<void> {
  return request(`/subsets/${id}`, { method: "DELETE" });
}

export function exportSubsetStar(
  subsetId: string,
  particlesStar: string
): Promise<{ path: string; n_particles: number }> {
  return request(`/subsets/${subsetId}/export-star`, {
    method: "POST",
    body: JSON.stringify({ particles_star: particlesStar }),
  });
}

// --- System ---

export function getSystemInfo(): Promise<SystemInfo> {
  return request("/system/info");
}

export function getSlurmDefaults(): Promise<SlurmDefaults> {
  return request("/system/slurm-defaults");
}

// --- Jobs (extended) ---

export function getJobSbatchScript(id: string): Promise<SbatchScript> {
  return request(`/jobs/${id}/sbatch-script`);
}
