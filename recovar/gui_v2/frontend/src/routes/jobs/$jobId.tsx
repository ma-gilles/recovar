import { useState, useCallback, useMemo, useEffect, useRef } from "react";
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
  Copy,
  XCircle,
  X,
  Eye,
  EyeOff,
  RefreshCw,
  ZoomIn,
} from "lucide-react";
import { clsx } from "clsx";
import {
  getJob,
  getProject,
  getJobVolumes,
  getJobPlots,
  getSuggestedNext,
  getJobSbatchScript,
  cancelJob,
  reconcileJob,
  getChartData,
  type JobDetail,
  type ProjectDetail,
  type VolumeEntry,
  type PlotEntry,
  type SuggestedNext,
  type SbatchScript,
  type ChartData,
} from "../../lib/api/client";
import Plot from "react-plotly.js";
import { useProject } from "../../lib/project-context";
import { VolumeViewer, type PinnedVolume } from "../../components/volume-viewer/VolumeViewer";
import {
  VolumeCategoryGroup,
  CATEGORY_ORDER,
  COLLAPSED_BY_DEFAULT,
  isHiddenVolume,
  naturalCompare,
  formatJobType,
  jobTypeToUrlSlug,
} from "../../components/volume-viewer/volumeCategories";
import { MaskWizard } from "../../components/mask-wizard/MaskWizard";
import { MAX_PINNED_VOLUMES } from "../../lib/constants";
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
  onTabChange,
}: {
  job: JobDetail;
  suggestions?: SuggestedNext[];
  onReconcile?: () => void;
  isReconciling?: boolean;
  onTabChange?: (tab: TabId) => void;
}): React.JSX.Element {
  const duration =
    job.completed && job.created
      ? Math.round(
          (new Date(job.completed).getTime() - new Date(job.created).getTime()) / 1000
        )
      : null;

  const isActive = job.status === "running" || job.status === "queued";

  const hoursQueued =
    job.status === "queued" && job.created
      ? (Date.now() - new Date(job.created).getTime()) / 3_600_000
      : 0;

  const { data: plots } = useQuery<PlotEntry[]>({
    queryKey: ["job-plots", job.id],
    queryFn: () => getJobPlots(job.id),
    enabled: job.status === "completed",
  });

  const { data: volumes } = useQuery<VolumeEntry[]>({
    queryKey: ["job-volumes", job.id],
    queryFn: () => getJobVolumes(job.id),
    enabled: job.status === "completed",
  });

  const previewPlots = plots?.slice(0, 3) ?? [];
  const volumeCount = volumes?.length ?? 0;
  const hasPreview = previewPlots.length > 0 || volumeCount > 0;

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
                <RefreshCw className={clsx("h-3 w-3", isReconciling && "motion-safe:animate-spin")} />
                Refresh
              </button>
            )}
          </div>
        </div>
        <div className="space-y-1">
          <span className="text-xs text-zinc-500">Type</span>
          <p className="text-sm capitalize">{formatJobType(job.type)}</p>
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

      {hoursQueued >= 1 && (
        <div className="rounded-md border border-yellow-700 bg-yellow-900/30 p-4">
          <p className="text-sm text-yellow-300">
            Job has been queued for {Math.floor(hoursQueued)} {Math.floor(hoursQueued) === 1 ? "hour" : "hours"}. The cluster may be busy.
          </p>
        </div>
      )}

      {job.output_dir && (
        <div className="space-y-1">
          <span className="text-xs text-zinc-500">Output Directory</span>
          <div className="flex items-center gap-1">
            <p className="font-mono text-sm text-zinc-300">{job.output_dir}</p>
            <button
              className="inline-flex items-center rounded px-1.5 py-0.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
              onClick={() => navigator.clipboard?.writeText(job.output_dir!).catch(() => undefined)}
              title="Copy path"
              aria-label="Copy output directory path"
            >
              <Copy className="h-3 w-3" />
            </button>
          </div>
        </div>
      )}

      {/* Clone / Suggested Next Steps — shown for terminal jobs */}
      {(job.status === "completed" || job.status === "failed" || job.status === "cancelled") && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-zinc-400">
            {job.status === "failed" ? "Retry" : "Next Steps"}
          </h3>
          <div className="flex flex-wrap gap-2">
            <Link
              to="/jobs/new"
              search={buildCloneSearchParams(job)}
              className="inline-flex items-center gap-1.5 rounded-md border border-blue-600/50 bg-blue-600/10 px-3 py-1.5 text-sm text-blue-300 hover:bg-blue-600/20"
            >
              <Copy className="h-3.5 w-3.5" />
              {job.status === "failed" ? "Retry with same settings" : "Clone Job"}
            </Link>
            {suggestions?.map((s) => (
              <Link
                key={s.type}
                to="/jobs/new"
                search={{
                  type: jobTypeToUrlSlug(s.type),
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

      {hasPreview && (
        <div className="space-y-3">
          <h3 className="text-sm font-medium text-zinc-400">Quick Preview</h3>
          {previewPlots.length > 0 && (
            <div className="flex gap-3">
              {previewPlots.map((p) => (
                <button
                  key={p.path}
                  onClick={() => onTabChange?.("plots")}
                  className="group overflow-hidden rounded-md border border-zinc-800 bg-zinc-900 hover:border-zinc-600"
                  title={`${p.name} — click to view all plots`}
                >
                  <img
                    src={`/api/files/serve?path=${encodeURIComponent(p.path)}`}
                    alt={p.name}
                    className="h-[150px] w-auto object-contain"
                    loading="lazy"
                  />
                  <p className="truncate px-2 py-1 text-xs text-zinc-500 group-hover:text-zinc-300">
                    {p.name}
                  </p>
                </button>
              ))}
            </div>
          )}
          {volumeCount > 0 && (
            <button
              onClick={() => onTabChange?.("volumes")}
              className="inline-flex items-center gap-2 rounded-md border border-zinc-700 bg-zinc-800/50 px-3 py-1.5 text-sm text-zinc-300 hover:bg-zinc-700 hover:text-zinc-100"
            >
              <Box className="h-4 w-4 text-sky-400" />
              {volumeCount} volume{volumeCount !== 1 ? "s" : ""} available
            </button>
          )}
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
  const type = jobTypeToUrlSlug(job.type);
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
          <pre className="max-h-96 overflow-auto whitespace-pre-wrap break-all rounded-md bg-zinc-900 p-3 font-mono text-xs text-zinc-300">
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
                        onClick={() => navigator.clipboard?.writeText(display).catch(() => undefined)}
                        title="Copy value"
                        aria-label={`Copy ${key} value`}
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

function VolumesTab({ jobId, projectId }: { jobId: string; projectId: string }): React.JSX.Element {
  const [selectedVolume, setSelectedVolume] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const [pinnedVolumes, setPinnedVolumes] = useState<PinnedVolume[]>([]);
  const [maskSource, setMaskSource] = useState<{ path: string; name: string } | null>(null);
  const { data: volumes, isLoading } = useQuery<VolumeEntry[]>({
    queryKey: ["job-volumes", jobId],
    queryFn: () => getJobVolumes(jobId),
  });

  // Compute filtered volumes and ordered groups
  const { filteredVolumes, orderedGroups, hiddenCount, ambiguousNames } = useMemo(() => {
    if (!volumes) return { filteredVolumes: [], orderedGroups: [] as Array<[string, VolumeEntry[]]>, hiddenCount: 0, ambiguousNames: new Set<string>() };

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

    // Natural sort within each category
    for (const vols of Object.values(grps)) {
      vols.sort((a, b) => naturalCompare(a.name, b.name));
    }

    // Order groups by CATEGORY_ORDER, then any unknown categories alphabetically
    const ordered: Array<[string, VolumeEntry[]]> = [];
    for (const cat of CATEGORY_ORDER) {
      if (grps[cat]) ordered.push([cat, grps[cat]]);
    }
    for (const cat of Object.keys(grps).sort()) {
      if (!CATEGORY_ORDER.includes(cat)) {
        ordered.push([cat, grps[cat]]);
      }
    }

    return { filteredVolumes: filtered, orderedGroups: ordered, hiddenCount: hidden.length, ambiguousNames: ambiguous };
  }, [volumes, showAll]);

  const handleSelect = useCallback((path: string) => {
    setSelectedVolume((prev) => (prev === path ? null : path));
  }, []);

  const handlePin = useCallback(
    (path: string, name: string) => {
      setPinnedVolumes((prev) => {
        if (prev.length >= MAX_PINNED_VOLUMES) return prev;
        if (prev.some((v) => v.path === path)) return prev;
        const category = volumes?.find((v) => v.path === path)?.category;
        return [...prev, { path, name, threshold: 3.0, opacity: 0.8, visible: true, colorIndex: prev.length, category }];
      });
    },
    [volumes]
  );

  const handleUnpin = useCallback((path: string) => {
    setPinnedVolumes((prev) => prev.filter((v) => v.path !== path));
  }, []);

  const pinnedPaths = useMemo(() => new Set(pinnedVolumes.map((p) => p.path)), [pinnedVolumes]);

  if (isLoading) return <Spinner label="Loading volumes..." />;
  if (!volumes || volumes.length === 0) {
    return <p className="text-sm text-zinc-500">No volumes in this job output.</p>;
  }

  return (
    <div className="flex flex-col" style={{ height: "calc(100vh - 200px)", minHeight: 600 }}>
      {/* TOP HALF: Volume viewer (fixed) */}
      <div className="shrink-0 rounded-lg border border-zinc-800 bg-zinc-950 p-4" style={selectedVolume ? { minHeight: 480 } : undefined}>
        {selectedVolume ? (
          <VolumeViewer
            volumes={filteredVolumes}
            initialVolumePath={selectedVolume}
            hideVolumeList
            pinnedVolumes={pinnedVolumes}
            onPinnedVolumesChange={setPinnedVolumes}
          />
        ) : (
          <div className="flex items-center justify-center rounded-lg border border-zinc-800 bg-zinc-900/50" style={{ height: 120 }}>
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
            aria-label={showAll ? `Hide ${hiddenCount} halfmap/unfiltered volumes` : `Show ${hiddenCount} halfmap/unfiltered volumes`}
          >
            {showAll ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
            {showAll ? "Hide" : "Show"} {hiddenCount} halfmap/unfiltered volume{hiddenCount !== 1 ? "s" : ""}
          </button>
        )}
      </div>

      {/* BOTTOM HALF: Scrollable volume list */}
      <div className="flex-1 overflow-auto space-y-1 pr-1">
        {orderedGroups.map(([cat, vols]) => (
          <VolumeCategoryGroup
            key={cat}
            cat={cat}
            vols={vols}
            selectedVolume={selectedVolume}
            onSelect={handleSelect}
            ambiguousNames={ambiguousNames}
            defaultCollapsed={COLLAPSED_BY_DEFAULT.has(cat) || vols.length > 10}
            pinnedPaths={pinnedPaths}
            onPin={handlePin}
            onUnpin={handleUnpin}
            pinDisabled={pinnedVolumes.length >= MAX_PINNED_VOLUMES}
            onMakeMask={(path, name) => setMaskSource({ path, name })}
          />
        ))}
      </div>

      <MaskWizard
        open={maskSource !== null}
        onClose={() => setMaskSource(null)}
        sourcePath={maskSource?.path ?? ""}
        sourceName={maskSource?.name ?? ""}
        projectId={projectId}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Interactive Plotly charts for diagnostic plots
// ---------------------------------------------------------------------------

const DARK_LAYOUT: Record<string, unknown> = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(24,24,27,1)", // zinc-900
  font: { color: "#a1a1aa", size: 12 }, // zinc-400
  xaxis: { gridcolor: "#3f3f46", zerolinecolor: "#52525b" }, // zinc-700, zinc-600
  yaxis: { gridcolor: "#3f3f46", zerolinecolor: "#52525b" },
  margin: { t: 30, r: 20, b: 40, l: 50 },
  autosize: true,
};

/** Map plot filenames to chart data endpoint names. */
function detectChartName(filename: string): string | null {
  const lower = filename.toLowerCase();
  if (lower.includes("fsc")) return "fsc";
  if (lower.includes("eigenvalue")) return "eigenvalues";
  if (lower.includes("histogram")) return "histogram";
  return null;
}

/** Derive a human-readable caption from a plot filename. */
function plotCaption(filename: string): string {
  return filename
    .replace(/\.[^.]+$/, "")       // strip extension
    .replace(/[_-]+/g, " ")        // underscores/hyphens to spaces
    .replace(/\b\w/g, (c) => c.toUpperCase()); // title-case each word
}

function PlotCell({
  plot,
  jobId,
  onEnlarge,
}: {
  plot: PlotEntry;
  jobId: string;
  onEnlarge: (entry: PlotEntry) => void;
}): React.JSX.Element {
  const chartName = detectChartName(plot.name);
  const [chartFailed, setChartFailed] = useState(false);

  const showInteractive = chartName !== null && !chartFailed;

  return (
    <div className="group overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900 transition-all duration-200 hover:border-zinc-600 hover:shadow-lg hover:shadow-black/30">
      {showInteractive ? (
        <div className="min-h-[300px] p-3">
          <InteractiveChartWithFallback
            jobId={jobId}
            chartName={chartName}
            onFallback={() => setChartFailed(true)}
          />
        </div>
      ) : (
        <button
          onClick={() => onEnlarge(plot)}
          className="relative w-full cursor-pointer bg-zinc-950 p-4 transition-transform duration-200 group-hover:scale-[1.02]"
        >
          <img
            src={`/api/files/serve?path=${encodeURIComponent(plot.path)}`}
            alt={plot.name}
            className="mx-auto w-full rounded-md object-contain"
            style={{ minHeight: 200 }}
            loading="lazy"
          />
          <div className="absolute inset-0 flex items-center justify-center bg-black/0 transition-colors duration-200 group-hover:bg-black/20">
            <ZoomIn className="h-8 w-8 text-white opacity-0 drop-shadow-lg transition-opacity duration-200 group-hover:opacity-70" />
          </div>
        </button>
      )}
      <div className="border-t border-zinc-800 px-3 py-2">
        <p className="truncate text-sm font-medium text-zinc-300" title={plot.name}>
          {plotCaption(plot.name)}
        </p>
        <p className="truncate text-xs text-zinc-500">{plot.name}</p>
      </div>
    </div>
  );
}

function InteractiveChartWithFallback({
  jobId,
  chartName,
  onFallback,
}: {
  jobId: string;
  chartName: string;
  onFallback: () => void;
}): React.JSX.Element | null {
  const { data, isLoading, isError } = useQuery<ChartData | null>({
    queryKey: ["chart-data", jobId, chartName],
    queryFn: () => getChartData(jobId, chartName),
    retry: false,
    staleTime: Infinity,
  });

  // undefined while loading; null when the endpoint returned non-OK (e.g. 404);
  // isError if the request threw. Any of these -> fall back to the static PNG so
  // the "Loading chart..." spinner is never terminal.
  const unavailable =
    isError || data === null || (data !== undefined && data.traces.length === 0);

  useEffect(() => {
    if (unavailable) {
      onFallback();
    }
  }, [unavailable, onFallback]);

  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <Spinner label="Loading chart..." />
      </div>
    );
  }

  if (unavailable || !data) {
    return null;
  }

  const mergedLayout = {
    ...DARK_LAYOUT,
    ...(data.layout ?? {}),
  };

  return (
    <Plot
      data={data.traces as Plotly.Data[]}
      layout={mergedLayout as Partial<Plotly.Layout>}
      config={{ responsive: true, displaylogo: false }}
      useResizeHandler
      style={{ width: "100%", height: "100%" }}
    />
  );
}

function PlotsTab({ jobId }: { jobId: string }): React.JSX.Element {
  const { data: plots, isLoading } = useQuery<PlotEntry[]>({
    queryKey: ["job-plots", jobId],
    queryFn: () => getJobPlots(jobId),
  });

  const [enlarged, setEnlarged] = useState<PlotEntry | null>(null);
  const [showAnnotationVariants, setShowAnnotationVariants] = useState(false);

  // Close modal on Escape key
  useEffect(() => {
    if (!enlarged) return;
    const handler = (e: KeyboardEvent): void => {
      if (e.key === "Escape") setEnlarged(null);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [enlarged]);

  const annotationCount = useMemo(
    () => (plots ?? []).filter((p) => /no_annot/i.test(p.name)).length,
    [plots]
  );

  const visiblePlots = useMemo(
    () =>
      showAnnotationVariants
        ? (plots ?? [])
        : (plots ?? []).filter((p) => !/no_annot/i.test(p.name)),
    [plots, showAnnotationVariants]
  );

  if (isLoading) return <Spinner label="Loading plots..." />;
  if (!plots || plots.length === 0) {
    return <p className="text-sm text-zinc-500">No plots in this job output.</p>;
  }

  return (
    <>
      {annotationCount > 0 && (
        <div className="mb-4 flex items-center gap-3">
          <span className="text-xs text-zinc-500">
            {visiblePlots.length} plot{visiblePlots.length !== 1 ? "s" : ""}
          </span>
          <button
            onClick={() => setShowAnnotationVariants(!showAnnotationVariants)}
            className="flex items-center gap-1 text-xs text-zinc-400 hover:text-zinc-200"
            aria-label={showAnnotationVariants ? `Hide ${annotationCount} annotation variant plots` : `Show ${annotationCount} annotation variant plots`}
          >
            {showAnnotationVariants ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
            {showAnnotationVariants ? "Hide" : "Show"} {annotationCount} annotation variant{annotationCount !== 1 ? "s" : ""}
          </button>
        </div>
      )}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {visiblePlots.map((p) => (
          <PlotCell
            key={p.path}
            plot={p}
            jobId={jobId}
            onEnlarge={setEnlarged}
          />
        ))}
      </div>

      {/* Lightbox modal */}
      {enlarged && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-sm"
          onClick={() => setEnlarged(null)}
        >
          {/* Modal content - stop propagation so clicking image doesn't close */}
          <div
            className="relative flex max-h-[92vh] max-w-[92vw] flex-col items-center"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Close button */}
            <button
              onClick={() => setEnlarged(null)}
              className="absolute -right-3 -top-3 z-10 flex h-8 w-8 items-center justify-center rounded-full bg-zinc-800 text-zinc-300 shadow-lg transition-colors hover:bg-zinc-700 hover:text-white"
              aria-label="Close"
            >
              <X className="h-4 w-4" />
            </button>

            {/* Image */}
            <div className="overflow-auto rounded-lg border border-zinc-700 bg-zinc-950 p-4 shadow-2xl">
              <img
                src={`/api/files/serve?path=${encodeURIComponent(enlarged.path)}`}
                alt={enlarged.name}
                className="max-h-[80vh] max-w-[85vw] object-contain"
              />
            </div>

            {/* Caption below image */}
            <div className="mt-3 text-center">
              <p className="text-sm font-medium text-zinc-200">
                {plotCaption(enlarged.name)}
              </p>
              <p className="mt-0.5 text-xs text-zinc-500">{enlarged.name}</p>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

/**
 * Dropdown that lets the user pick another job of the same type from
 * the active project, then navigates to /compare?jobs=<this>,<that>.
 */
function CompareWithDropdown({
  currentJob,
  project,
}: {
  currentJob: JobDetail;
  project: { id: string } | null;
}): React.JSX.Element | null {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onClick(e: MouseEvent): void {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, [open]);

  const { data: projectDetail } = useQuery<ProjectDetail>({
    queryKey: ["project", project?.id],
    queryFn: () => getProject(project!.id),
    enabled: open && !!project?.id,
  });

  const candidates = useMemo(() => {
    if (!projectDetail) return [] as ProjectDetail["jobs"];
    return projectDetail.jobs.filter(
      (j) => j.id !== currentJob.id && j.type === currentJob.type
    );
  }, [projectDetail, currentJob.id, currentJob.type]);

  return (
    <div ref={ref} className="relative">
      <Button variant="outline" size="sm" onClick={() => setOpen((o) => !o)}>
        Compare with…
      </Button>
      {open && (
        <div className="absolute right-0 top-full z-20 mt-1 max-h-72 w-72 overflow-y-auto rounded-md border border-zinc-700 bg-zinc-900 shadow-xl">
          {!projectDetail ? (
            <p className="px-3 py-2 text-xs text-zinc-500">Loading…</p>
          ) : candidates.length === 0 ? (
            <p className="px-3 py-2 text-xs text-zinc-500">
              No other {currentJob.type} jobs in this project.
            </p>
          ) : (
            candidates.map((j) => (
              <Link
                key={j.id}
                to="/compare"
                search={{ jobs: `${currentJob.id},${j.id}` }}
                onClick={() => setOpen(false)}
                className="block px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-800 hover:text-zinc-50"
                title={j.output_dir}
              >
                <span className="block truncate">
                  {j.output_dir.split("/").slice(-2).join("/")}
                </span>
                <span className="text-[10px] text-zinc-500">{j.status} · {new Date(j.created).toLocaleDateString()}</span>
              </Link>
            ))
          )}
        </div>
      )}
    </div>
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

  const { project, setProject } = useProject();

  // Auto-restore project context when navigating directly to a job page
  useEffect(() => {
    if (project || !job?.project_id) return;
    let cancelled = false;
    getProject(job.project_id).then((projectData) => {
      if (!cancelled) {
        setProject({ id: projectData.id, path: projectData.path, name: projectData.name });
      }
    }).catch(() => {
      // Ignore errors — project may have been deleted
    });
    return () => { cancelled = true; };
  }, [project, job?.project_id, setProject]);

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

  const cancelMutation = useMutation({
    mutationFn: () => cancelJob(jobId),
    onSuccess: () => {
      refetch();
      queryClient.invalidateQueries({ queryKey: ["project"] });
    },
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
            {formatJobType(job.type)}
          </h1>
          <StatusBadge status={job.status} />
        </div>
        <div className="flex items-center gap-2">
          {!isTerminal && (
            <div className="flex flex-col items-end gap-1">
              <Button
                variant="destructive"
                size="sm"
                onClick={() => cancelMutation.mutate()}
                disabled={cancelMutation.isPending}
              >
                <XCircle className="h-3.5 w-3.5" />
                {cancelMutation.isPending ? "Cancelling…" : "Cancel"}
              </Button>
              {cancelMutation.isError && (
                <span className="text-xs text-red-400">Failed to cancel job.</span>
              )}
            </div>
          )}
          {isTerminal && (job.type.toLowerCase() === "pipeline" || job.type.toLowerCase() === "analyze") && (
            <Link to="/explore/$jobId" params={{ jobId }}>
              <Button variant="default" size="sm">
                Explore Latent Space
              </Button>
            </Link>
          )}
          <CompareWithDropdown currentJob={job} project={project} />
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
                  "flex items-center gap-1.5 border-b-2 px-4 py-2 text-sm transition-colors outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-1 focus-visible:ring-offset-zinc-950",
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
            onTabChange={setActiveTab}
          />
        )}
        {activeTab === "logs" && (
          <LogViewer jobId={jobId} jobStatus={job.status} onStatusChange={handleStatusChange} />
        )}
        {activeTab === "params" && <ParamsTab job={job} />}
        {activeTab === "volumes" && <VolumesTab jobId={jobId} projectId={job.project_id} />}
        {activeTab === "plots" && <PlotsTab jobId={jobId} />}
      </div>
    </div>
  );
}
