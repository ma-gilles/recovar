import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import {
  Plus,
  ChevronDown,
  ChevronRight,
  Menu,
  CheckCircle,
  XCircle,
  Clock,
  Loader2,
  MinusCircle,
  HardDrive,
  Beaker,
  FolderOpen,
  FolderPlus,
} from "lucide-react";
import { clsx } from "clsx";
import { getProject, createProject, type ProjectDetail, type JobSummary } from "../../lib/api/client";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { FileBrowser } from "../file-browser/FileBrowser";

// Status icon mapping per DESIGN-SYSTEM.md
function StatusIcon({ status }: { status: string }): React.JSX.Element {
  switch (status) {
    case "running":
      return <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-500" />;
    case "completed":
      return <CheckCircle className="h-3.5 w-3.5 text-emerald-500" />;
    case "failed":
      return <XCircle className="h-3.5 w-3.5 text-red-500" />;
    case "queued":
      return <Clock className="h-3.5 w-3.5 text-amber-500" />;
    case "cancelled":
      return <MinusCircle className="h-3.5 w-3.5 text-zinc-500" />;
    default:
      return <MinusCircle className="h-3.5 w-3.5 text-zinc-500" />;
  }
}

function JobItem({ job }: { job: JobSummary }): React.JSX.Element {
  const dirName = job.output_dir.split("/").pop() ?? job.id.slice(0, 8);
  return (
    <Link
      to="/jobs/$jobId"
      params={{ jobId: job.id }}
      className={clsx(
        "flex items-center gap-2 rounded-md px-3 py-1.5 text-sm text-zinc-400",
        "hover:bg-zinc-800 hover:text-zinc-50",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
      )}
      activeProps={{ className: "bg-zinc-700 text-zinc-50" }}
    >
      <StatusIcon status={job.status} />
      <span className="truncate">{dirName}</span>
    </Link>
  );
}

function JobSection({
  title,
  jobs,
  defaultOpen = true,
}: {
  title: string;
  jobs: JobSummary[];
  defaultOpen?: boolean;
}): React.JSX.Element | null {
  const [open, setOpen] = useState(defaultOpen);
  if (jobs.length === 0) return null;

  return (
    <div>
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-1 px-3 py-1.5 text-xs font-medium uppercase tracking-wider text-zinc-500 hover:text-zinc-300"
      >
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        {title} ({jobs.length})
      </button>
      {open && (
        <div className="ml-2 space-y-0.5">
          {jobs.map((job) => (
            <JobItem key={job.id} job={job} />
          ))}
        </div>
      )}
    </div>
  );
}

function DiskUsage({ bytes, total }: { bytes: number; total: number }): React.JSX.Element {
  const usedGB = (bytes / 1e9).toFixed(1);
  const totalGB = (total / 1e9).toFixed(0);
  const pct = total > 0 ? (bytes / total) * 100 : 0;
  const color = pct > 95 ? "text-red-500" : pct > 80 ? "text-amber-500" : "text-emerald-500";

  return (
    <div className="px-3 py-2 text-xs text-zinc-500">
      <div className="flex items-center gap-1">
        <HardDrive className="h-3 w-3" />
        <span className={color}>
          {usedGB} GB / {totalGB} GB
        </span>
      </div>
      <div className="mt-1 h-1 w-full rounded-full bg-zinc-800">
        <div
          className={clsx("h-1 rounded-full", pct > 95 ? "bg-red-500" : pct > 80 ? "bg-amber-500" : "bg-emerald-500")}
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>
    </div>
  );
}

interface SidebarProps {
  projectId?: string;
  onProjectCreated?: (project: { id: string; path: string; name: string }) => void;
}

export function Sidebar({ projectId, onProjectCreated }: SidebarProps): React.JSX.Element {
  const [collapsed, setCollapsed] = useState(false);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [showOpenForm, setShowOpenForm] = useState(false);

  const { data: project } = useQuery<ProjectDetail>({
    queryKey: ["project", projectId],
    queryFn: () => getProject(projectId!),
    enabled: !!projectId,
    refetchInterval: 5000,
  });

  const pipelineJobs = project?.jobs.filter((j) => j.type.toLowerCase() === "pipeline") ?? [];
  const analyzeJobs = project?.jobs.filter((j) => j.type.toLowerCase() === "analyze") ?? [];
  const otherJobs =
    project?.jobs.filter((j) => j.type.toLowerCase() !== "pipeline" && j.type.toLowerCase() !== "analyze") ?? [];

  const handleProjectCreated = (p: { id: string; path: string; name: string }) => {
    setShowCreateForm(false);
    setShowOpenForm(false);
    onProjectCreated?.(p);
  };

  if (collapsed) {
    return (
      <aside className="flex w-12 flex-col items-center border-r border-zinc-800 bg-zinc-900 py-2">
        <button
          onClick={() => setCollapsed(false)}
          className="rounded p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-50"
          aria-label="Expand sidebar"
        >
          <Menu className="h-4 w-4" />
        </button>
        {projectId ? (
          <Link
            to="/jobs/new" search={{ type: undefined, result_dir: undefined }}
            className="mt-4 rounded p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-50"
            aria-label="New job"
          >
            <Plus className="h-4 w-4" />
          </Link>
        ) : (
          <button
            onClick={() => { setCollapsed(false); setShowCreateForm(true); }}
            className="mt-4 rounded p-1.5 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-50"
            aria-label="Create project"
          >
            <FolderPlus className="h-4 w-4" />
          </button>
        )}
      </aside>
    );
  }

  return (
    <aside className="flex w-60 flex-col border-r border-zinc-800 bg-zinc-900">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-zinc-800 px-3 py-3">
        <Link to="/" className="flex items-center gap-2 text-sm font-medium text-zinc-50">
          <Beaker className="h-4 w-4 text-blue-500" />
          recovar
        </Link>
        <button
          onClick={() => setCollapsed(true)}
          className="rounded p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-50"
          aria-label="Collapse sidebar"
        >
          <Menu className="h-4 w-4" />
        </button>
      </div>

      {projectId ? (
        <>
          {/* New Job button — only when project is open */}
          <div className="border-b border-zinc-800 p-2">
            <Link
              to="/jobs/new" search={{ type: undefined, result_dir: undefined }}
              className="flex w-full items-center justify-center gap-2 rounded-md bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-500"
            >
              <Plus className="h-4 w-4" />
              New Job
            </Link>
          </div>

          {/* Job tree */}
          <nav className="flex-1 overflow-y-auto py-2">
            {project ? (
              <div className="space-y-1">
                <div className="flex items-center gap-2 px-3 py-1 text-xs text-zinc-500">
                  <FolderOpen className="h-3 w-3" />
                  <span className="truncate">{project.name}</span>
                </div>
                <div className="border-t border-zinc-800 pt-1">
                  <JobSection title="Pipeline" jobs={pipelineJobs} />
                  <JobSection title="Analyze" jobs={analyzeJobs} />
                  <JobSection title="Other" jobs={otherJobs} defaultOpen={false} />
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-4 w-4 animate-spin text-zinc-500" />
              </div>
            )}
          </nav>

          {/* Disk usage */}
          {project && project.disk_usage_bytes > 0 && (
            <div className="border-t border-zinc-800">
              <DiskUsage bytes={project.disk_usage_bytes} total={project.disk_usage_total} />
            </div>
          )}
        </>
      ) : (
        <>
          {/* No project — show Create/Open buttons */}
          <div className="border-b border-zinc-800 p-2 space-y-1.5">
            <button
              onClick={() => { setShowCreateForm(true); setShowOpenForm(false); }}
              className="flex w-full items-center justify-center gap-2 rounded-md bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-500"
            >
              <FolderPlus className="h-4 w-4" />
              Create Project
            </button>
            <button
              onClick={() => { setShowOpenForm(true); setShowCreateForm(false); }}
              className="flex w-full items-center justify-center gap-2 rounded-md border border-zinc-700 px-3 py-2 text-sm font-medium text-zinc-300 hover:bg-zinc-800"
            >
              <FolderOpen className="h-4 w-4" />
              Open Project
            </button>
          </div>
          <nav className="flex-1 overflow-y-auto py-4">
            <div className="px-3 text-center text-xs text-zinc-500">
              No project open.
              <br />
              Create or open a project to start.
            </div>
          </nav>
        </>
      )}

      {/* Create/Open project modals rendered outside the aside */}
      {(showCreateForm || showOpenForm) && (
        <ProjectFormOverlay
          mode={showCreateForm ? "create" : "open"}
          onDone={handleProjectCreated}
          onCancel={() => { setShowCreateForm(false); setShowOpenForm(false); }}
        />
      )}
    </aside>
  );
}

// ---------------------------------------------------------------------------
// Overlay for Create / Open project
// ---------------------------------------------------------------------------

function ProjectFormOverlay({
  mode,
  onDone,
  onCancel,
}: {
  mode: "create" | "open";
  onDone: (project: { id: string; path: string; name: string }) => void;
  onCancel: () => void;
}): React.JSX.Element {
  const queryClient = useQueryClient();
  const [path, setPath] = useState("");
  const [name, setName] = useState("");
  const [showBrowser, setShowBrowser] = useState(true);

  const mutation = useMutation({
    mutationFn: () => {
      const projectName = mode === "create" ? name || path.split("/").pop() || "Project" : path.split("/").pop() || "Project";
      return createProject(path, projectName);
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["project"] });
      onDone({ id: data.id, path: data.path, name: data.name });
    },
  });

  const isCreate = mode === "create";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
      <div className="w-full max-w-lg rounded-lg border border-zinc-700 bg-zinc-900 p-6 shadow-xl">
        <h2 className="text-lg font-medium">
          {isCreate ? "Create Project" : "Open Project"}
        </h2>
        <p className="mt-1 text-sm text-zinc-400">
          {isCreate
            ? "Choose a directory for the new project. A recovar_project.db will be created there."
            : "Select a directory that contains an existing recovar project."}
        </p>

        <div className="mt-4 space-y-3">
          {/* Path */}
          <div className="space-y-1">
            <Label>Directory</Label>
            <div className="flex gap-2">
              <Input
                value={path}
                onChange={(e) => setPath(e.target.value)}
                placeholder="/scratch/gpfs/..."
                className="font-mono"
              />
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowBrowser(!showBrowser)}
              >
                {showBrowser ? "Hide" : "Browse"}
              </Button>
            </div>
          </div>

          {showBrowser && (
            <FileBrowser
              initialPath="/scratch/gpfs/GILLES/mg6942"
              selectDirectory
              onSelect={(selectedPath) => {
                setPath(selectedPath);
                // Auto-fill the name from directory
                if (!name) {
                  setName(selectedPath.split("/").pop() || "");
                }
              }}
            />
          )}

          {/* Name — only for Create */}
          {isCreate && (
            <div className="space-y-1">
              <Label>Project Name</Label>
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder={path.split("/").pop() || "My Project"}
              />
            </div>
          )}

          {mutation.isError && (
            <p className="text-sm text-red-400">{(mutation.error as Error).message}</p>
          )}

          <div className="flex justify-end gap-2 pt-2">
            <Button variant="outline" onClick={onCancel}>
              Cancel
            </Button>
            <Button
              onClick={() => mutation.mutate()}
              disabled={!path}
              loading={mutation.isPending}
            >
              {mutation.isPending
                ? isCreate ? "Creating..." : "Opening..."
                : isCreate ? "Create Project" : "Open Project"}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
