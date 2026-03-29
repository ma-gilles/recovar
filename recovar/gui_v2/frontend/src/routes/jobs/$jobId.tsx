import { useState, useCallback } from "react";
import { useParams, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowLeft,
  Clock,
  Terminal,
  Settings,
  Box,
  Image,
  ChevronRight,
  Copy,
  XCircle,
} from "lucide-react";
import { clsx } from "clsx";
import {
  getJob,
  getJobVolumes,
  getJobPlots,
  getSuggestedNext,
  cancelJob,
  type JobDetail,
  type VolumeEntry,
  type PlotEntry,
  type SuggestedNext,
} from "../../lib/api/client";
import { VolumeViewer } from "../../components/volume-viewer/VolumeViewer";
import { StatusBadge } from "../../components/ui/badge";
import { Button } from "../../components/ui/button";
import { Spinner } from "../../components/ui/spinner";
import { LogViewer } from "../../components/log-viewer/LogViewer";

const tabs = [
  { id: "overview", label: "Overview", icon: Clock },
  { id: "logs", label: "Logs", icon: Terminal },
  { id: "params", label: "Parameters", icon: Settings },
  { id: "volumes", label: "Volumes", icon: Box },
  { id: "plots", label: "Plots", icon: Image },
] as const;

type TabId = (typeof tabs)[number]["id"];

function OverviewTab({ job, suggestions }: { job: JobDetail; suggestions?: SuggestedNext[] }): React.JSX.Element {
  const duration =
    job.completed && job.created
      ? Math.round(
          (new Date(job.completed).getTime() - new Date(job.created).getTime()) / 1000
        )
      : null;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <div className="space-y-1">
          <span className="text-xs text-zinc-500">Status</span>
          <div>
            <StatusBadge status={job.status} />
          </div>
        </div>
        <div className="space-y-1">
          <span className="text-xs text-zinc-500">Type</span>
          <p className="text-sm capitalize">{job.type.replace("_", " ")}</p>
        </div>
        <div className="space-y-1">
          <span className="text-xs text-zinc-500">Created</span>
          <p className="text-sm">{new Date(job.created).toLocaleString()}</p>
        </div>
        {duration !== null && (
          <div className="space-y-1">
            <span className="text-xs text-zinc-500">Duration</span>
            <p className="text-sm">
              {duration < 60
                ? `${duration}s`
                : duration < 3600
                  ? `${Math.floor(duration / 60)}m ${duration % 60}s`
                  : `${Math.floor(duration / 3600)}h ${Math.floor((duration % 3600) / 60)}m`}
            </p>
          </div>
        )}
      </div>

      {job.slurm_id && (
        <div className="space-y-1">
          <span className="text-xs text-zinc-500">SLURM Job ID</span>
          <p className="font-mono text-sm">{job.slurm_id}</p>
        </div>
      )}

      {job.error && (
        <div className="rounded-md border border-red-500/30 bg-red-500/10 p-4">
          <p className="text-sm font-medium text-red-400">Error</p>
          <pre className="mt-1 whitespace-pre-wrap font-mono text-xs text-red-300">{job.error}</pre>
        </div>
      )}

      {job.output_dir && (
        <div className="space-y-1">
          <span className="text-xs text-zinc-500">Output Directory</span>
          <p className="font-mono text-sm text-zinc-300">{job.output_dir}</p>
        </div>
      )}

      {suggestions && suggestions.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-zinc-400">Suggested Next Steps</h3>
          <div className="flex flex-wrap gap-2">
            {suggestions.map((s) => (
              <Link
                key={s.type}
                to="/jobs/new"
                search={{
                  type: s.type.toLowerCase().replace(/([a-z])([A-Z])/g, "$1_$2").toLowerCase(),
                  result_dir: (s.prefilled_params?.result_dir as string) || undefined,
                  density: (s.prefilled_params?.density as string) || undefined,
                  input: (s.prefilled_params?.input as string) || undefined,
                  particles: (s.prefilled_params?.particles as string) || undefined,
                  params: undefined,
                }}
                className="inline-flex items-center gap-1 rounded-md border border-zinc-700 px-3 py-1.5 text-sm hover:bg-zinc-800"
              >
                {s.label}
                <ChevronRight className="h-3.5 w-3.5" />
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function buildCloneSearchParams(job: JobDetail): {
  type: string | undefined;
  result_dir: string | undefined;
  density: string | undefined;
  input: string | undefined;
  particles: string | undefined;
  params: string | undefined;
} {
  const typeMap: Record<string, string> = {
    Pipeline: "pipeline",
    Analyze: "analyze",
    ComputeState: "compute_state",
    ComputeTrajectory: "compute_trajectory",
    Density: "density",
    StableStates: "stable_states",
    Postprocess: "postprocess",
    Downsample: "downsample",
  };
  const type = typeMap[job.type] ?? job.type.toLowerCase().replace(/([a-z])([A-Z])/g, "$1_$2").toLowerCase();
  const p = job.params ?? {};
  return {
    type,
    result_dir: p.result_dir ? String(p.result_dir) : undefined,
    density: p.density ? String(p.density) : undefined,
    input: p.input ? String(p.input) : undefined,
    particles: p.particles ? String(p.particles) : undefined,
    params: JSON.stringify(p),
  };
}

function ParamsTab({ job }: { job: JobDetail }): React.JSX.Element {
  const [showCli, setShowCli] = useState(false);

  return (
    <div className="space-y-4">
      {/* Clone/CLI buttons at the top for easy access */}
      <div className="flex gap-2">
        <Button variant="outline" size="sm" onClick={() => setShowCli(!showCli)}>
          {showCli ? "Hide" : "Show"} CLI Command
        </Button>
        <Link to="/jobs/new" search={buildCloneSearchParams(job)}>
          <Button variant="outline" size="sm">
            <Copy className="h-3.5 w-3.5" />
            Clone Job
          </Button>
        </Link>
      </div>

      {showCli && job.params && (
        <pre className="rounded-md bg-zinc-900 p-3 font-mono text-xs text-zinc-300">
          recovar {job.type}{" "}
          {Object.entries(job.params)
            .map(([k, v]) => `--${k.replace(/_/g, "-")} ${typeof v === "boolean" ? "" : String(v)}`)
            .join(" ")}
        </pre>
      )}

      {job.params && Object.keys(job.params).length > 0 ? (
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-800 text-left text-xs text-zinc-500">
              <th className="pb-2 pr-4">Parameter</th>
              <th className="pb-2">Value</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(job.params).map(([key, value]) => {
              const display = typeof value === "object" ? JSON.stringify(value) : String(value);
              const isLong = display.length > 60;
              return (
                <tr key={key} className="border-b border-zinc-800/50">
                  <td className="py-2 pr-4 font-mono text-zinc-300 align-top">{key}</td>
                  <td className="py-2 font-mono text-zinc-400" style={{ wordBreak: "break-all", overflowWrap: "anywhere" }}>
                    <span>{display}</span>
                    {isLong && (
                      <button
                        className="ml-2 inline-flex items-center rounded px-1.5 py-0.5 text-[10px] text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
                        onClick={() => navigator.clipboard.writeText(display)}
                        title="Copy value"
                      >
                        <Copy className="h-3 w-3" />
                      </button>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      ) : (
        <p className="text-sm text-zinc-500">No parameters recorded.</p>
      )}

    </div>
  );
}

function VolumesTab({ jobId }: { jobId: string }): React.JSX.Element {
  const [selectedVolume, setSelectedVolume] = useState<string | null>(null);
  const { data: volumes, isLoading } = useQuery<VolumeEntry[]>({
    queryKey: ["job-volumes", jobId],
    queryFn: () => getJobVolumes(jobId),
  });

  if (isLoading) return <Spinner label="Loading volumes..." />;
  if (!volumes || volumes.length === 0) {
    return <p className="text-sm text-zinc-500">No volumes in this job output.</p>;
  }

  // Group by category
  const groups: Record<string, VolumeEntry[]> = {};
  for (const v of volumes) {
    (groups[v.category] ??= []).push(v);
  }

  return (
    <div className="space-y-4">
      {Object.entries(groups).map(([cat, vols]) => (
        <div key={cat}>
          <h4 className="mb-2 text-xs font-medium uppercase tracking-wider text-zinc-500">
            {cat} ({vols.length})
          </h4>
          <div className="grid grid-cols-2 gap-2 md:grid-cols-3 lg:grid-cols-4">
            {vols.map((v) => {
              // Show parent directory for disambiguation (e.g., "diagnostics/mask.mrc")
              const pathParts = v.path.replace(/\\/g, "/").split("/");
              const displayName = pathParts.length >= 2
                ? `${pathParts[pathParts.length - 2]}/${v.name}`
                : v.name;
              return (
                <button
                  key={v.path}
                  onClick={() => setSelectedVolume(selectedVolume === v.path ? null : v.path)}
                  className={clsx(
                    "rounded-md border bg-zinc-900 p-3 text-left hover:border-blue-500/50 hover:bg-zinc-800",
                    selectedVolume === v.path ? "border-blue-500" : "border-zinc-800"
                  )}
                >
                  <Box className="mb-1 h-8 w-8 text-sky-400" />
                  <p className="truncate text-sm" title={v.path}>{displayName}</p>
                  <p className="text-xs text-zinc-500">
                    {(v.size_bytes / 1e6).toFixed(1)} MB
                  </p>
                </button>
              );
            })}
          </div>
        </div>
      ))}

      {selectedVolume && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-950 p-4">
          <VolumeViewer volumes={volumes} initialVolumePath={selectedVolume} />
        </div>
      )}
    </div>
  );
}

function PlotsTab({ jobId }: { jobId: string }): React.JSX.Element {
  const { data: plots, isLoading } = useQuery<PlotEntry[]>({
    queryKey: ["job-plots", jobId],
    queryFn: () => getJobPlots(jobId),
  });

  const [enlarged, setEnlarged] = useState<string | null>(null);

  if (isLoading) return <Spinner label="Loading plots..." />;
  if (!plots || plots.length === 0) {
    return <p className="text-sm text-zinc-500">No plots in this job output.</p>;
  }

  return (
    <>
      <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
        {plots.map((p) => (
          <button
            key={p.path}
            onClick={() => setEnlarged(p.path)}
            className="rounded-md border border-zinc-800 bg-zinc-900 p-2 hover:border-blue-500/50"
          >
            <img
              src={`/api/files/serve?path=${encodeURIComponent(p.path)}`}
              alt={p.name}
              className="w-full rounded"
              loading="lazy"
            />
            <p className="mt-1 truncate text-xs text-zinc-400">{p.name}</p>
          </button>
        ))}
      </div>

      {/* Enlarged modal */}
      {enlarged && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80"
          onClick={() => setEnlarged(null)}
        >
          <img
            src={`/api/files/serve?path=${encodeURIComponent(enlarged)}`}
            alt="Enlarged plot"
            className="max-h-[90vh] max-w-[90vw] rounded-lg"
          />
        </div>
      )}
    </>
  );
}

export function JobDetailPage(): React.JSX.Element {
  const { jobId } = useParams({ from: "/jobs/$jobId" });
  const [activeTab, setActiveTab] = useState<TabId>("overview");

  const { data: job, isLoading, refetch } = useQuery<JobDetail>({
    queryKey: ["job", jobId],
    queryFn: () => getJob(jobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "running" || status === "queued") return 5000;
      return false;
    },
  });

  const { data: suggestions } = useQuery<SuggestedNext[]>({
    queryKey: ["job-suggestions", jobId],
    queryFn: () => getSuggestedNext(jobId),
    enabled: job?.status === "completed",
  });

  const handleStatusChange = useCallback(
    (_status: string) => {
      refetch();
    },
    [refetch]
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Spinner label="Loading job..." />
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

  const isTerminal = job.status === "completed" || job.status === "failed" || job.status === "cancelled";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Link to="/" className="text-zinc-400 hover:text-zinc-50">
            <ArrowLeft className="h-4 w-4" />
          </Link>
          <h1 className="text-xl font-semibold capitalize">
            {job.type.replace("_", " ")}
          </h1>
          <StatusBadge status={job.status} />
        </div>
        <div className="flex items-center gap-2">
          {!isTerminal && (
            <Button
              variant="destructive"
              size="sm"
              onClick={() => cancelJob(jobId).then(() => refetch())}
            >
              <XCircle className="h-3.5 w-3.5" />
              Cancel
            </Button>
          )}
          {isTerminal && job.type.toLowerCase() === "pipeline" && (
            <Link to="/explore/$jobId" params={{ jobId }}>
              <Button variant="outline" size="sm">
                Explore
              </Button>
            </Link>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-zinc-800">
        <div className="flex gap-1">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={clsx(
                  "flex items-center gap-1.5 border-b-2 px-4 py-2 text-sm transition-colors",
                  activeTab === tab.id
                    ? "border-blue-500 text-zinc-50"
                    : "border-transparent text-zinc-400 hover:text-zinc-200"
                )}
              >
                <Icon className="h-3.5 w-3.5" />
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Tab content */}
      <div>
        {activeTab === "overview" && <OverviewTab job={job} suggestions={suggestions} />}
        {activeTab === "logs" && (
          <LogViewer jobId={jobId} jobStatus={job.status} onStatusChange={handleStatusChange} />
        )}
        {activeTab === "params" && <ParamsTab job={job} />}
        {activeTab === "volumes" && <VolumesTab jobId={jobId} />}
        {activeTab === "plots" && <PlotsTab jobId={jobId} />}
      </div>
    </div>
  );
}
