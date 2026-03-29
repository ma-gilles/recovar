import { useState, useCallback, useMemo } from "react";
import { useParams, Link } from "@tanstack/react-router";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft,
  Clock,
  Terminal,
  Settings,
  Box,
  Image,
  ChevronRight,
  ChevronDown,
  Copy,
  XCircle,
  Eye,
  EyeOff,
  RefreshCw,
} from "lucide-react";
import { clsx } from "clsx";
import {
  getJob,
  getJobVolumes,
  getJobPlots,
  getSuggestedNext,
  getJobSbatchScript,
  cancelJob,
  reconcileJob,
  type JobDetail,
  type VolumeEntry,
  type PlotEntry,
  type SuggestedNext,
  type SbatchScript,
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

function OverviewTab({
  job,
  suggestions,
  onReconcile,
  isReconciling,
}: {
  job: JobDetail;
  suggestions?: SuggestedNext[];
  onReconcile?: () => void;
  isReconciling?: boolean;
}): React.JSX.Element {
  const duration =
    job.completed && job.created
      ? Math.round(
          (new Date(job.completed).getTime() - new Date(job.created).getTime()) / 1000
        )
      : null;

  const isActive = job.status === "running" || job.status === "queued";

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <div className="space-y-1">
          <span className="text-xs text-zinc-500">Status</span>
          <div className="flex items-center gap-2">
            <StatusBadge status={job.status} />
            {isActive && onReconcile && (
              <button
                onClick={onReconcile}
                disabled={isReconciling}
                className="inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50"
                title="Check actual SLURM status"
              >
                <RefreshCw className={clsx("h-3 w-3", isReconciling && "animate-spin")} />
                Refresh
              </button>
            )}
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

      <div className="space-y-1">
        <span className="text-xs text-zinc-500">Execution</span>
        <p className="text-sm">{job.execution_summary}</p>
      </div>

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
  const { data: sbatchData } = useQuery<SbatchScript>({
    queryKey: ["job-sbatch", job.id],
    queryFn: () => getJobSbatchScript(job.id),
    enabled: showCli,
  });

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

      {showCli && (
        <div className="space-y-1">
          {sbatchData?.source === "file" && (
            <p className="text-xs text-zinc-500">
              From submit.sh (actual sbatch script used)
            </p>
          )}
          <pre className="max-h-96 overflow-auto rounded-md bg-zinc-900 p-3 font-mono text-xs text-zinc-300">
            {sbatchData?.script ?? "Loading..."}
          </pre>
        </div>
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

// ---------------------------------------------------------------------------
// Volume filtering & display helpers
// ---------------------------------------------------------------------------

/** Patterns matching "uninteresting" volumes hidden by default. */
const HIDDEN_PATTERNS = [/_half[0-9]/, /_unfil/, /halfmap/, /unfiltered/i];

/** Returns true if a volume name matches the hidden-by-default patterns. */
function isHiddenVolume(name: string): boolean {
  const lower = name.toLowerCase();
  return HIDDEN_PATTERNS.some((pat) => pat.test(lower));
}

/**
 * Build a display name for a volume.  If `needsDisambiguation` is true
 * (i.e. another volume in the same list has an identical filename),
 * prepend the parent directory.
 */
function volumeDisplayName(v: VolumeEntry, needsDisambiguation: boolean): string {
  if (!needsDisambiguation) return v.name;
  const parts = v.path.replace(/\\/g, "/").split("/");
  if (parts.length >= 2) {
    return `${parts[parts.length - 2]}/${v.name}`;
  }
  return v.name;
}

/** Default number of items shown in a collapsed category. */
const COLLAPSED_LIMIT = 5;

function VolumeCategoryGroup({
  cat,
  vols,
  selectedVolume,
  onSelect,
  ambiguousNames,
}: {
  cat: string;
  vols: VolumeEntry[];
  selectedVolume: string | null;
  onSelect: (path: string) => void;
  ambiguousNames: Set<string>;
}): React.JSX.Element {
  const [expanded, setExpanded] = useState(vols.length <= COLLAPSED_LIMIT);
  const visible = expanded ? vols : vols.slice(0, COLLAPSED_LIMIT);
  const remaining = vols.length - COLLAPSED_LIMIT;

  return (
    <div>
      <h4 className="mb-2 text-xs font-medium uppercase tracking-wider text-zinc-500">
        {cat} ({vols.length})
      </h4>
      <div className="grid grid-cols-2 gap-2 md:grid-cols-3 lg:grid-cols-4">
        {visible.map((v) => {
          const displayName = volumeDisplayName(v, ambiguousNames.has(v.name));
          return (
            <button
              key={v.path}
              onClick={() => onSelect(v.path)}
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
      {!expanded && remaining > 0 && (
        <button
          onClick={() => setExpanded(true)}
          className="mt-2 flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300"
        >
          <ChevronDown className="h-3 w-3" />
          Show all {vols.length} volumes
        </button>
      )}
      {expanded && vols.length > COLLAPSED_LIMIT && (
        <button
          onClick={() => setExpanded(false)}
          className="mt-2 flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300"
        >
          <ChevronRight className="h-3 w-3" />
          Collapse
        </button>
      )}
    </div>
  );
}

function VolumesTab({ jobId }: { jobId: string }): React.JSX.Element {
  const [selectedVolume, setSelectedVolume] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const { data: volumes, isLoading } = useQuery<VolumeEntry[]>({
    queryKey: ["job-volumes", jobId],
    queryFn: () => getJobVolumes(jobId),
  });

  // Compute filtered volumes and groups
  const { filteredVolumes, groups, hiddenCount, ambiguousNames } = useMemo(() => {
    if (!volumes) return { filteredVolumes: [], groups: {} as Record<string, VolumeEntry[]>, hiddenCount: 0, ambiguousNames: new Set<string>() };

    const hidden = volumes.filter((v) => isHiddenVolume(v.name));
    const filtered = showAll ? volumes : volumes.filter((v) => !isHiddenVolume(v.name));

    // Detect duplicate filenames for disambiguation
    const nameCounts = new Map<string, number>();
    for (const v of filtered) {
      nameCounts.set(v.name, (nameCounts.get(v.name) ?? 0) + 1);
    }
    const ambiguous = new Set<string>();
    for (const [name, count] of nameCounts) {
      if (count > 1) ambiguous.add(name);
    }

    // Group by category
    const grps: Record<string, VolumeEntry[]> = {};
    for (const v of filtered) {
      (grps[v.category] ??= []).push(v);
    }

    return { filteredVolumes: filtered, groups: grps, hiddenCount: hidden.length, ambiguousNames: ambiguous };
  }, [volumes, showAll]);

  const handleSelect = useCallback((path: string) => {
    setSelectedVolume((prev) => (prev === path ? null : path));
  }, []);

  if (isLoading) return <Spinner label="Loading volumes..." />;
  if (!volumes || volumes.length === 0) {
    return <p className="text-sm text-zinc-500">No volumes in this job output.</p>;
  }

  return (
    <div className="flex flex-col" style={{ height: "calc(100vh - 200px)", minHeight: 600 }}>
      {/* TOP HALF: Volume viewer (fixed) */}
      <div className="shrink-0 rounded-lg border border-zinc-800 bg-zinc-950 p-4" style={{ minHeight: 480 }}>
        {selectedVolume ? (
          <VolumeViewer volumes={filteredVolumes} initialVolumePath={selectedVolume} />
        ) : (
          <div className="flex items-center justify-center" style={{ height: 400 }}>
            <p className="text-sm text-zinc-500">Click a volume below to view it</p>
          </div>
        )}
      </div>

      {/* Show all / hide toggle */}
      <div className="flex items-center gap-3 py-3">
        <span className="text-xs text-zinc-500">
          {filteredVolumes.length} volume{filteredVolumes.length !== 1 ? "s" : ""}
        </span>
        {hiddenCount > 0 && (
          <button
            onClick={() => setShowAll(!showAll)}
            className="flex items-center gap-1 text-xs text-zinc-400 hover:text-zinc-200"
          >
            {showAll ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
            {showAll ? "Hide" : "Show"} {hiddenCount} halfmap/unfiltered volume{hiddenCount !== 1 ? "s" : ""}
          </button>
        )}
      </div>

      {/* BOTTOM HALF: Scrollable volume grid */}
      <div className="flex-1 overflow-auto space-y-4 pr-1">
        {Object.entries(groups).map(([cat, vols]) => (
          <VolumeCategoryGroup
            key={cat}
            cat={cat}
            vols={vols}
            selectedVolume={selectedVolume}
            onSelect={handleSelect}
            ambiguousNames={ambiguousNames}
          />
        ))}
      </div>
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
  const queryClient = useQueryClient();

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

  const reconcileMutation = useMutation({
    mutationFn: () => reconcileJob(jobId),
    onSuccess: () => {
      // Refresh the job detail and the project sidebar
      refetch();
      queryClient.invalidateQueries({ queryKey: ["project"] });
    },
  });

  const handleReconcile = useCallback(() => {
    reconcileMutation.mutate();
  }, [reconcileMutation]);

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
        {activeTab === "overview" && (
          <OverviewTab
            job={job}
            suggestions={suggestions}
            onReconcile={handleReconcile}
            isReconciling={reconcileMutation.isPending}
          />
        )}
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
