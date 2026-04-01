import { useState, useEffect, useMemo, useCallback } from "react";
import { useParams, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Box, ChevronRight, ChevronDown, Eye, EyeOff } from "lucide-react";
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

// ---------------------------------------------------------------------------
// Volume categorization helpers (mirrors jobs/$jobId.tsx VolumesTab logic)
// ---------------------------------------------------------------------------

const HIDDEN_PATTERNS = [/_half[0-9]/, /_unfil/, /halfmap/, /unfiltered/i, /^sampling\.mrc$/i];

function isHiddenVolume(name: string): boolean {
  return HIDDEN_PATTERNS.some((pat) => pat.test(name.toLowerCase()));
}

function volumeDisplayName(v: VolumeEntry, needsDisambiguation: boolean): string {
  if (!needsDisambiguation) return v.name;
  const parts = v.path.replace(/\\/g, "/").split("/");
  if (parts.length >= 2) return `${parts[parts.length - 2]}/${v.name}`;
  return v.name;
}

const CATEGORY_LABELS: Record<string, string> = {
  mean: "Mean Reconstruction",
  eigen: "Eigenvolumes",
  variance: "Variance Map",
  halfmap: "Half-maps (raw)",
  mask: "Masks",
  kmeans_center: "K-means Centers",
  trajectory: "Trajectory Volumes",
  reconstruction: "Reconstructed States",
  density: "Density / Deconvolved",
  other: "Other",
};

const CATEGORY_ORDER: string[] = [
  "mean", "eigen", "variance", "kmeans_center", "trajectory",
  "reconstruction", "density", "mask", "other", "halfmap",
];

const COLLAPSED_BY_DEFAULT = new Set(["halfmap", "other"]);

function naturalCompare(a: string, b: string): number {
  const re = /(\d+)/g;
  const aParts = a.split(re);
  const bParts = b.split(re);
  const len = Math.min(aParts.length, bParts.length);
  for (let i = 0; i < len; i++) {
    const aNum = Number(aParts[i]);
    const bNum = Number(bParts[i]);
    if (!isNaN(aNum) && !isNaN(bNum)) {
      if (aNum !== bNum) return aNum - bNum;
    } else {
      const cmp = (aParts[i] ?? "").localeCompare(bParts[i] ?? "");
      if (cmp !== 0) return cmp;
    }
  }
  return aParts.length - bParts.length;
}

function VolumeCategoryGroup({
  cat,
  vols,
  selectedVolume,
  onSelect,
  ambiguousNames,
  defaultCollapsed,
}: {
  cat: string;
  vols: VolumeEntry[];
  selectedVolume: string | null;
  onSelect: (path: string) => void;
  ambiguousNames: Set<string>;
  defaultCollapsed: boolean;
}): React.JSX.Element {
  const [open, setOpen] = useState(!defaultCollapsed);

  return (
    <div>
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-1.5 py-1.5 text-left text-xs font-medium uppercase tracking-wider text-zinc-500 hover:text-zinc-300"
      >
        {open ? (
          <ChevronDown className="h-3.5 w-3.5 shrink-0" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 shrink-0" />
        )}
        {CATEGORY_LABELS[cat] ?? cat}
        <span className="font-normal normal-case tracking-normal text-zinc-600">
          ({vols.length})
        </span>
      </button>
      {open && (
        <div className="ml-5 space-y-px">
          {vols.map((v) => {
            const displayName = volumeDisplayName(v, ambiguousNames.has(v.name));
            const active = selectedVolume === v.path;
            return (
              <button
                key={v.path}
                onClick={() => onSelect(v.path)}
                className={clsx(
                  "flex w-full items-center gap-2 rounded px-2 py-1 text-left text-sm",
                  active
                    ? "bg-blue-500/15 text-blue-300"
                    : "text-zinc-300 hover:bg-zinc-800 hover:text-zinc-100"
                )}
                title={v.path}
              >
                <Box className="h-3.5 w-3.5 shrink-0 text-sky-400" />
                <span className="truncate">{displayName}</span>
                <span className="ml-auto shrink-0 text-xs text-zinc-600">
                  {(v.size_bytes / 1e6).toFixed(1)} MB
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

function CategorizedVolumesView({ volumes }: { volumes?: VolumeEntry[] }): React.JSX.Element {
  const [selectedVolume, setSelectedVolume] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);

  const { filteredVolumes, orderedGroups, hiddenCount, ambiguousNames } = useMemo(() => {
    if (!volumes) return { filteredVolumes: [], orderedGroups: [] as Array<[string, VolumeEntry[]]>, hiddenCount: 0, ambiguousNames: new Set<string>() };

    const hidden = volumes.filter((v) => isHiddenVolume(v.name));
    const filtered = showAll ? volumes : volumes.filter((v) => !isHiddenVolume(v.name));

    const nameCounts = new Map<string, number>();
    for (const v of filtered) {
      nameCounts.set(v.name, (nameCounts.get(v.name) ?? 0) + 1);
    }
    const ambiguous = new Set<string>();
    for (const [name, count] of nameCounts) {
      if (count > 1) ambiguous.add(name);
    }

    const grps: Record<string, VolumeEntry[]> = {};
    for (const v of filtered) {
      (grps[v.category] ??= []).push(v);
    }
    for (const vols of Object.values(grps)) {
      vols.sort((a, b) => naturalCompare(a.name, b.name));
    }

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

  if (!volumes || volumes.length === 0) {
    return <p className="text-sm text-zinc-500">No volumes in this job output.</p>;
  }

  return (
    <div className="flex flex-col" style={{ height: "calc(100vh - 200px)", minHeight: 600 }}>
      {/* Volume viewer */}
      <div className="shrink-0 rounded-lg border border-zinc-800 bg-zinc-950 p-4" style={selectedVolume ? { minHeight: 480 } : undefined}>
        {selectedVolume ? (
          <VolumeViewer volumes={filteredVolumes} initialVolumePath={selectedVolume} hideVolumeList />
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
          >
            {showAll ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
            {showAll ? "Hide" : "Show"} {hiddenCount} halfmap/unfiltered volume{hiddenCount !== 1 ? "s" : ""}
          </button>
        )}
      </div>

      {/* Categorized volume list */}
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
          />
        ))}
      </div>
    </div>
  );
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
          resultDir={((job.params as Record<string, unknown> | null)?.result_dir as string) ?? job.output_dir ?? ""}
          particlesStar={particlesStar}
          analyzeZdim={((job.params as Record<string, unknown> | null)?.zdim as number) ?? null}
        />
      ) : (
        <CategorizedVolumesView volumes={volumes} />
      )}
    </div>
  );
}
