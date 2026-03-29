import { useState } from "react";
import { useParams, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Box } from "lucide-react";
import { clsx } from "clsx";
import { getJob, getJobVolumes, type JobDetail, type VolumeEntry } from "../../lib/api/client";
import { useProject } from "../../lib/project-context";
import { Spinner } from "../../components/ui/spinner";
import { LatentExplorer } from "../../components/latent-explorer/LatentExplorer";
import { VolumeViewer } from "../../components/volume-viewer/VolumeViewer";

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

  const { project: activeProject } = useProject();
  const projectId = activeProject?.id ?? "";

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
        />
      ) : (
        <VolumeViewer volumes={volumes} />
      )}
    </div>
  );
}
