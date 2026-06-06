import React, { useMemo, Suspense } from "react";
import { Link, useSearch } from "@tanstack/react-router";
import { useQueries, useQuery } from "@tanstack/react-query";
import { ArrowLeft, Box, Diff, GitCompare, LineChart } from "lucide-react";
import Plot from "react-plotly.js";
import {
  getJob,
  getJobVolumes,
  getJobPlots,
  getChartData,
  type ChartData,
  type JobDetail,
  type PlotEntry,
  type VolumeEntry,
} from "../lib/api/client";
import { Spinner } from "../components/ui/spinner";
import { formatJobType } from "../components/volume-viewer/volumeCategories";

// Design system colors matching the volume overlay (sky/rose/emerald/amber).
const JOB_COLORS = ["#38bdf8", "#fb7185", "#34d399", "#fbbf24"];

// Lazy-load VtkViewer (vtk.js is heavy; only fetched when this page loads)
const VtkViewer = React.lazy(() =>
  import("../components/volume-viewer/VtkViewer").then((m) => ({ default: m.VtkViewer }))
);

interface CompareSearch {
  jobs: string;
}

/** Color row by whether the values match. */
function diffClass(differs: boolean): string {
  return differs
    ? "bg-amber-500/10 text-amber-200"
    : "text-zinc-400";
}

function fmt(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function formatDuration(start?: string | null, end?: string | null): string {
  if (!start || !end) return "—";
  const ms = new Date(end).getTime() - new Date(start).getTime();
  if (ms < 0 || isNaN(ms)) return "—";
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const sr = s % 60;
  if (m < 60) return `${m}m ${sr}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

export function ComparePage(): React.JSX.Element {
  const search = useSearch({ from: "/compare" }) as CompareSearch;
  const jobIds = useMemo(
    () =>
      (search.jobs ?? "")
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
    [search.jobs]
  );

  // Hooks must run in stable order — fetch up to 4 jobs.
  const j0 = useQuery<JobDetail>({
    queryKey: ["job", jobIds[0]],
    queryFn: () => getJob(jobIds[0]),
    enabled: !!jobIds[0],
  });
  const j1 = useQuery<JobDetail>({
    queryKey: ["job", jobIds[1]],
    queryFn: () => getJob(jobIds[1]),
    enabled: !!jobIds[1],
  });
  const j2 = useQuery<JobDetail>({
    queryKey: ["job", jobIds[2]],
    queryFn: () => getJob(jobIds[2]),
    enabled: !!jobIds[2],
  });
  const j3 = useQuery<JobDetail>({
    queryKey: ["job", jobIds[3]],
    queryFn: () => getJob(jobIds[3]),
    enabled: !!jobIds[3],
  });

  const jobs = [j0.data, j1.data, j2.data, j3.data]
    .filter((j): j is JobDetail => !!j)
    .slice(0, jobIds.length);
  const loading = [j0, j1, j2, j3].some((q) => q.isLoading);
  const error = [j0, j1, j2, j3].find((q) => q.error)?.error;

  // Build a sorted union of param keys across all jobs, then mark
  // rows where any value differs from the others.
  const paramRows = useMemo(() => {
    if (jobs.length === 0) return [] as { key: string; values: unknown[]; differs: boolean }[];
    const keys = new Set<string>();
    for (const j of jobs) {
      for (const k of Object.keys(j.params ?? {})) keys.add(k);
    }
    const rows = Array.from(keys)
      .sort()
      .map((k) => {
        const values = jobs.map((j) => (j.params ?? {})[k]);
        const first = JSON.stringify(values[0]);
        const differs = values.some((v) => JSON.stringify(v) !== first);
        return { key: k, values, differs };
      });
    return rows;
  }, [jobs]);

  if (jobIds.length < 2) {
    return (
      <div className="space-y-4">
        <h1 className="text-xl font-semibold">Compare Jobs</h1>
        <p className="text-sm text-zinc-400">
          Pass two or more job IDs as <code>?jobs=id1,id2</code>.
        </p>
        <Link to="/" className="text-sm text-blue-400 hover:underline">
          ← Dashboard
        </Link>
      </div>
    );
  }

  if (loading && jobs.length === 0) {
    return <p className="text-sm text-zinc-500">Loading jobs…</p>;
  }
  if (error) {
    return (
      <div className="space-y-4">
        <h1 className="text-xl font-semibold">Compare Jobs</h1>
        <p className="text-sm text-red-400">
          Failed to load: {(error as Error).message}
        </p>
        <Link to="/" className="text-sm text-blue-400 hover:underline">
          ← Dashboard
        </Link>
      </div>
    );
  }
  if (jobs.length < 2) {
    return (
      <div className="space-y-4">
        <h1 className="text-xl font-semibold">Compare Jobs</h1>
        <p className="text-sm text-zinc-400">
          Could not load enough jobs to compare ({jobs.length}/{jobIds.length}).
        </p>
        <Link to="/" className="text-sm text-blue-400 hover:underline">
          ← Dashboard
        </Link>
      </div>
    );
  }

  const diffCount = paramRows.filter((r) => r.differs).length;
  const sameCount = paramRows.length - diffCount;

  return (
    <CompareInner
      jobs={jobs}
      paramRows={paramRows}
      diffCount={diffCount}
      sameCount={sameCount}
    />
  );
}

/**
 * Pick the most representative single volume from a job's volume list.
 * Pipeline → mean.mrc; ComputeState/ComputeTrajectory → state000.mrc;
 * otherwise the first non-halfmap. Returns null if nothing usable.
 */
function pickRepresentativeVolume(volumes: VolumeEntry[] | undefined): VolumeEntry | null {
  if (!volumes || volumes.length === 0) return null;
  const usable = volumes.filter(
    (v) => !/half|unfil|locres|sampling/i.test(v.name)
  );
  const mean = usable.find((v) => /\bmean(_filt)?\.mrc$/i.test(v.name));
  if (mean) return mean;
  const state0 = usable.find((v) => /state0+\.mrc$/i.test(v.name));
  if (state0) return state0;
  return usable[0] ?? volumes[0] ?? null;
}

interface CompareInnerProps {
  jobs: JobDetail[];
  paramRows: { key: string; values: unknown[]; differs: boolean }[];
  diffCount: number;
  sameCount: number;
}

function CompareInner({ jobs, paramRows, diffCount, sameCount }: CompareInnerProps): React.JSX.Element {
  // Fetch volumes for up to 4 jobs (hooks must be in stable order).
  const v0 = useQuery<VolumeEntry[]>({
    queryKey: ["job-volumes", jobs[0]?.id],
    queryFn: () => getJobVolumes(jobs[0]!.id),
    enabled: !!jobs[0],
  });
  const v1 = useQuery<VolumeEntry[]>({
    queryKey: ["job-volumes", jobs[1]?.id],
    queryFn: () => getJobVolumes(jobs[1]!.id),
    enabled: !!jobs[1],
  });
  const v2 = useQuery<VolumeEntry[]>({
    queryKey: ["job-volumes", jobs[2]?.id],
    queryFn: () => getJobVolumes(jobs[2]!.id),
    enabled: !!jobs[2],
  });
  const v3 = useQuery<VolumeEntry[]>({
    queryKey: ["job-volumes", jobs[3]?.id],
    queryFn: () => getJobVolumes(jobs[3]!.id),
    enabled: !!jobs[3],
  });

  const repVolumes = useMemo(
    () => [v0.data, v1.data, v2.data, v3.data].slice(0, jobs.length).map(pickRepresentativeVolume),
    [v0.data, v1.data, v2.data, v3.data, jobs.length]
  );

  // FSC chart data for each job (returns null gracefully if unavailable).
  const fscQueries = useQueries({
    queries: jobs.map((j) => ({
      queryKey: ["chart-data", j.id, "fsc"] as const,
      queryFn: () => getChartData(j.id, "fsc"),
      retry: false,
      staleTime: Infinity,
    })),
  });
  const fscOverlay = useMemo(() => {
    const traces: Array<Record<string, unknown>> = [];
    fscQueries.forEach((q, i) => {
      const data = q.data as ChartData | null | undefined;
      if (!data || !data.traces || data.traces.length === 0) return;
      const color = JOB_COLORS[i % JOB_COLORS.length];
      data.traces.forEach((trace, ti) => {
        traces.push({
          ...trace,
          line: { color, width: 2, ...(trace.line as object | undefined) },
          marker: { color, ...(trace.marker as object | undefined) },
          name:
            data.traces.length > 1
              ? `${formatJobType(jobs[i].type)} #${i + 1} · ${trace.name ?? `curve ${ti}`}`
              : `${formatJobType(jobs[i].type)} #${i + 1}`,
        });
      });
    });
    return traces;
  }, [fscQueries, jobs]);
  const fscLoading = fscQueries.some((q) => q.isLoading);

  // Fallback: if structured FSC data is not available, show the existing
  // diagnostic PNG (mean_fsc.png) side-by-side per job.
  const plotQueries = useQueries({
    queries: jobs.map((j) => ({
      queryKey: ["job-plots", j.id] as const,
      queryFn: () => getJobPlots(j.id),
    })),
  });
  const fscPlotPerJob = useMemo(
    () =>
      plotQueries.map((q) => {
        const list = (q.data as PlotEntry[] | undefined) ?? [];
        return (
          list.find((p) => /mean_fsc|^fsc/i.test(p.name)) ?? null
        );
      }),
    [plotQueries]
  );
  const anyFscPlot = fscPlotPerJob.some((p) => !!p);
  const pinnedForOverlay = useMemo(
    () =>
      repVolumes
        .map((rep, i) =>
          rep
            ? {
                path: rep.path,
                name: `${formatJobType(jobs[i].type)} (${rep.name})`,
                threshold: 3.0,
                opacity: 0.6,
                visible: true,
                colorIndex: i % 4,
              }
            : null
        )
        .filter((p): p is NonNullable<typeof p> => !!p),
    [repVolumes, jobs]
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Link
          to="/jobs/$jobId"
          params={{ jobId: jobs[0].id }}
          className="flex items-center gap-1 text-sm text-zinc-400 hover:text-zinc-200"
        >
          <ArrowLeft className="h-4 w-4" /> Back
        </Link>
        <h1 className="flex items-center gap-2 text-xl font-semibold">
          <GitCompare className="h-5 w-5 text-blue-400" /> Compare Jobs
        </h1>
        <span className="text-xs text-zinc-500">
          {jobs.length} jobs · {diffCount} differing param{diffCount === 1 ? "" : "s"} · {sameCount} matching
        </span>
      </div>

      {/* Header row: job identifiers */}
      <div className="overflow-x-auto rounded-lg border border-zinc-800">
        <table className="w-full text-sm">
          <thead className="bg-zinc-900">
            <tr>
              <th className="w-44 px-3 py-2 text-left font-medium text-zinc-500">Job</th>
              {jobs.map((j) => (
                <th key={j.id} className="px-3 py-2 text-left">
                  <Link
                    to="/jobs/$jobId"
                    params={{ jobId: j.id }}
                    className="block truncate font-medium text-blue-300 hover:underline"
                    title={j.output_dir}
                  >
                    {formatJobType(j.type)}
                  </Link>
                  <p className="truncate text-xs text-zinc-500" title={j.output_dir}>
                    {j.output_dir.split("/").slice(-2).join("/")}
                  </p>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr className="border-t border-zinc-800">
              <td className="px-3 py-1.5 text-left text-zinc-500">Status</td>
              {jobs.map((j) => (
                <td key={j.id} className="px-3 py-1.5">{j.status}</td>
              ))}
            </tr>
            <tr className="border-t border-zinc-800">
              <td className="px-3 py-1.5 text-left text-zinc-500">Created</td>
              {jobs.map((j) => (
                <td key={j.id} className="px-3 py-1.5 font-mono text-xs">
                  {j.created ? new Date(j.created).toLocaleString() : "—"}
                </td>
              ))}
            </tr>
            <tr className="border-t border-zinc-800">
              <td className="px-3 py-1.5 text-left text-zinc-500">Duration</td>
              {jobs.map((j) => (
                <td key={j.id} className="px-3 py-1.5">{formatDuration(j.created, j.completed)}</td>
              ))}
            </tr>
            <tr className="border-t border-zinc-800">
              <td className="px-3 py-1.5 text-left text-zinc-500">SLURM ID</td>
              {jobs.map((j) => (
                <td key={j.id} className="px-3 py-1.5 font-mono text-xs">{j.slurm_id ?? "—"}</td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>

      {/* Volume overlay (isosurfaces) */}
      <div className="space-y-2">
        <h2 className="flex items-center gap-2 text-sm font-medium uppercase tracking-wider text-zinc-500">
          <Box className="h-4 w-4" /> Volume Overlay
          <span className="ml-auto text-xs normal-case tracking-normal text-zinc-600">
            Representative volume per job, overlaid as colored isosurfaces
          </span>
        </h2>
        <div className="rounded-lg border border-zinc-800 bg-black" style={{ height: 360 }}>
          {pinnedForOverlay.length === 0 ? (
            <div className="flex h-full items-center justify-center text-xs text-zinc-500">
              No comparable volumes found in these jobs.
            </div>
          ) : (
            <Suspense fallback={<Spinner label="Loading 3D viewer..." />}>
              <VtkViewer
                activeVolume={pinnedForOverlay[0].path}
                pinnedVolumes={pinnedForOverlay}
              />
            </Suspense>
          )}
        </div>
      </div>

      {/* FSC overlay */}
      <div className="space-y-2">
        <h2 className="flex items-center gap-2 text-sm font-medium uppercase tracking-wider text-zinc-500">
          <LineChart className="h-4 w-4" /> FSC
          <span className="ml-auto text-xs normal-case tracking-normal text-zinc-600">
            Half-map gold-standard FSC per job
          </span>
        </h2>
        <div className="rounded-lg border border-zinc-800 bg-zinc-950 p-2" style={{ minHeight: 280 }}>
          {fscLoading && fscOverlay.length === 0 && !anyFscPlot ? (
            <div className="flex h-64 items-center justify-center">
              <Spinner label="Loading FSC..." />
            </div>
          ) : fscOverlay.length === 0 && anyFscPlot ? (
            <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
              {fscPlotPerJob.map((p, i) => (
                <div key={jobs[i].id} className="space-y-1">
                  <div className="flex items-center gap-1.5 text-xs">
                    <span
                      className="inline-block h-2 w-2 rounded-full"
                      style={{ backgroundColor: JOB_COLORS[i % JOB_COLORS.length] }}
                    />
                    <span className="truncate text-zinc-400">
                      {formatJobType(jobs[i].type)} · {jobs[i].output_dir.split("/").slice(-1)[0]}
                    </span>
                  </div>
                  {p ? (
                    <img
                      src={`/api/files/serve?path=${encodeURIComponent(p.path)}`}
                      alt={`FSC for ${formatJobType(jobs[i].type)}`}
                      className="w-full rounded border border-zinc-800 bg-black"
                    />
                  ) : (
                    <div className="flex h-32 items-center justify-center rounded border border-zinc-800 text-xs text-zinc-600">
                      No FSC plot
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : fscOverlay.length === 0 ? (
            <div className="flex h-64 items-center justify-center text-xs text-zinc-500">
              No FSC data available for these jobs.
            </div>
          ) : (
            <Plot
              data={fscOverlay as Plotly.Data[]}
              layout={{
                autosize: true,
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(24,24,27,1)",
                font: { color: "#a1a1aa", size: 12 },
                xaxis: {
                  title: "Spatial frequency shell (index)",
                  gridcolor: "#3f3f46",
                  zerolinecolor: "#52525b",
                },
                yaxis: {
                  title: "FSC",
                  range: [0, 1],
                  gridcolor: "#3f3f46",
                  zerolinecolor: "#52525b",
                },
                margin: { t: 20, r: 20, b: 50, l: 60 },
                showlegend: true,
                legend: { bgcolor: "rgba(0,0,0,0)" },
              } as Partial<Plotly.Layout>}
              useResizeHandler
              style={{ width: "100%", height: 280 }}
              config={{ displaylogo: false, responsive: true }}
            />
          )}
        </div>
      </div>

      {/* Parameter diff */}
      <div className="space-y-2">
        <h2 className="flex items-center gap-2 text-sm font-medium uppercase tracking-wider text-zinc-500">
          <Diff className="h-4 w-4" /> Parameters
          <span className="ml-auto text-xs normal-case tracking-normal">
            <span className="text-amber-300">●</span> different &nbsp;
            <span className="text-zinc-600">●</span> identical
          </span>
        </h2>
        <div className="overflow-x-auto rounded-lg border border-zinc-800">
          <table className="w-full text-sm">
            <thead className="bg-zinc-900">
              <tr>
                <th className="w-44 px-3 py-2 text-left font-medium text-zinc-500">Key</th>
                {jobs.map((j) => (
                  <th key={j.id} className="px-3 py-2 text-left text-zinc-500">
                    {formatJobType(j.type)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {paramRows.length === 0 && (
                <tr>
                  <td colSpan={jobs.length + 1} className="px-3 py-4 text-center text-zinc-500">
                    No parameters recorded.
                  </td>
                </tr>
              )}
              {paramRows.map((row) => (
                <tr key={row.key} className={"border-t border-zinc-800 " + diffClass(row.differs)}>
                  <td className="px-3 py-1 font-mono text-xs">{row.key}</td>
                  {row.values.map((v, i) => (
                    <td key={i} className="px-3 py-1 font-mono text-xs">
                      {fmt(v)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
