import { useState, useEffect } from "react";
import { useParams, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Box } from "lucide-react";
import { clsx } from "clsx";
import { getJob, getProject, getJobVolumes, type JobDetail, type VolumeEntry } from "../../lib/api/client";
import { useProject } from "../../lib/project-context";
import { Spinner } from "../../components/ui/spinner";
import { LatentExplorer } from "../../components/latent-explorer/LatentExplorer";
import { VolumeViewer } from "../../components/volume-viewer/VolumeViewer";

/**
 * Resolve the particles .star file path from the job's parent pipeline job.
 * The analyze job's parent_jobs[0] is the pipeline job, whose params.particles
 * contains the original .star file path.
 */
async function resolveParticlesStar(job: JobDetail): Promise<string | null> {
  // If this job itself has particles (unlikely for analyze, but check)
  const directParticles = (job.params as Record<string, unknown> | null)?.particles;
  if (typeof directParticles === "string" && directParticles.endsWith(".star")) {
    return directParticles;
  }
  // Look up parent pipeline job
  if (job.parent_jobs && job.parent_jobs.length > 0) {
    try {
      const parentJob = await getJob(job.parent_jobs[0]);
      const parentParticles = (parentJob.params as Record<string, unknown> | null)?.particles;
      if (typeof parentParticles === "string" && parentParticles.endsWith(".star")) {
        return parentParticles;
      }
    } catch {
      // Ignore — parent job may not exist
    }
  }
  return null;
}

type ViewMode = "explorer" | "volumes";

export function ExplorePage(): React.JSX.Element {
  const { jobId } = useParams({ from: "/explore/$jobId" });
  const [viewMode, setViewMode] = useState<ViewMode>("explorer");

  const { data: job, isLoading: jobLoading } = useQuery<JobDetail>({
    queryKey: ["job", jobId],
    queryFn: () => getJob(jobId),
  });

  const { data: volumes } = useQuery<VolumeEntry[]>({
    queryKey: ["job-volumes", jobId],
    queryFn: () => getJobVolumes(jobId),
    enabled: !!job,
  });

  // Resolve the original particles .star file path from the parent pipeline job
  const { data: particlesStar } = useQuery<string | null>({
    queryKey: ["particles-star", jobId],
    queryFn: () => resolveParticlesStar(job!),
    enabled: !!job,
  });

  const { project: activeProject, setProject } = useProject();

  // Auto-restore project context when navigating directly to an explore page
  useEffect(() => {
    if (activeProject || !job?.project_id) return;
    let cancelled = false;
    getProject(job.project_id).then((projectData) => {
      if (!cancelled) {
        setProject({ id: projectData.id, path: projectData.path, name: projectData.name });
      }
    }).catch(() => {
      // Ignore errors — project may have been deleted
    });
    return () => { cancelled = true; };
  }, [activeProject, job?.project_id, setProject]);

  const projectId = activeProject?.id ?? "";

  if (jobLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Spinner label="Loading..." />
      </div>
    );
  }

  if (!job) {
    return (
      <div className="py-20 text-center">
        <p className="text-zinc-400">Job not found.</p>
        <Link to="/" className="mt-2 text-blue-400 hover:underline">
          Back to dashboard
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Link
            to="/jobs/$jobId"
            params={{ jobId }}
            className="text-zinc-400 hover:text-zinc-50"
          >
            <ArrowLeft className="h-4 w-4" />
          </Link>
          <h1 className="text-xl font-semibold">Explore</h1>
          <span className="text-sm text-zinc-500">
            {job.output_dir?.split("/").pop()}
          </span>
        </div>

        <div className="flex gap-1 rounded-md border border-zinc-700 p-0.5">
          <button
            onClick={() => setViewMode("explorer")}
            className={clsx(
              "flex items-center gap-1 rounded px-3 py-1.5 text-xs",
              viewMode === "explorer"
                ? "bg-zinc-700 text-zinc-50"
                : "text-zinc-400 hover:text-zinc-200"
            )}
          >
            Latent Space
          </button>
          <button
            onClick={() => setViewMode("volumes")}
            className={clsx(
              "flex items-center gap-1 rounded px-3 py-1.5 text-xs",
              viewMode === "volumes"
                ? "bg-zinc-700 text-zinc-50"
                : "text-zinc-400 hover:text-zinc-200"
            )}
          >
            <Box className="h-3 w-3" />
            Volumes
          </button>
        </div>
      </div>

      {/* Content */}
      {viewMode === "explorer" ? (
        <LatentExplorer
          jobId={jobId}
          projectId={projectId}
          resultDir={job.output_dir ?? ""}
          particlesStar={particlesStar}
        />
      ) : (
        <VolumeViewer volumes={volumes} />
      )}
    </div>
  );
}
