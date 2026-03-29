import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import {
  createProject,
  scanProject,
  getProject,
  getSystemInfo,
  ApiError,
  type SystemInfo,
  type ProjectDetail,
} from "../lib/api/client";
import { useProject } from "../lib/project-context";
import type { UseMutationResult } from "@tanstack/react-query";
import { Plus, Server, FolderPlus, FolderOpen, Search, Beaker, BarChart3, Play, ScanSearch, AlertTriangle } from "lucide-react";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import { FileBrowser } from "../components/file-browser/FileBrowser";
import { isEphemeralPath, EPHEMERAL_PATH_WARNING } from "../lib/constants";

export function DashboardPage(): React.JSX.Element {
  const { project, setProject } = useProject();
  const queryClient = useQueryClient();
  const { data: sysInfo } = useQuery<SystemInfo>({
    queryKey: ["system-info"],
    queryFn: getSystemInfo,
  });

  const [showCreate, setShowCreate] = useState(false);
  const [showOpen, setShowOpen] = useState(false);
  const [createPath, setCreatePath] = useState("");
  const [createName, setCreateName] = useState("");
  const [showCreateBrowser, setShowCreateBrowser] = useState(false);
  const [showOpenBrowser, setShowOpenBrowser] = useState(false);
  const [openPath, setOpenPath] = useState("");

  // Scan form state
  const [showScan, setShowScan] = useState(false);
  const [scanPath, setScanPath] = useState("");
  const [showScanBrowser, setShowScanBrowser] = useState(false);

  const createMutation = useMutation({
    mutationFn: () => createProject(createPath, createName || createPath.split("/").pop() || "Project"),
    onSuccess: (data) => {
      setProject({ id: data.id, path: data.path, name: data.name });
      queryClient.invalidateQueries({ queryKey: ["project"] });
      setShowCreate(false);
      setCreatePath("");
      setCreateName("");
    },
  });

  const openMutation = useMutation({
    mutationFn: () => createProject(openPath, openPath.split("/").pop() || "Project"),
    onSuccess: (data) => {
      setProject({ id: data.id, path: data.path, name: data.name });
      queryClient.invalidateQueries({ queryKey: ["project"] });
      setShowOpen(false);
      setOpenPath("");
    },
  });

  const scanMutation = useMutation({
    mutationFn: () => scanProject(project!.id, scanPath),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["project", project!.id] });
      setShowScan(false);
      setScanPath("");
    },
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">recovar GUI</h1>
        {project && (
          <Link
            to="/jobs/new"
            search={{ type: undefined, result_dir: undefined, density: undefined, input: undefined, particles: undefined, params: undefined }}
            className="inline-flex items-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500"
          >
            <Plus className="h-4 w-4" />
            New Job
          </Link>
        )}
      </div>

      {/* System info bar */}
      {sysInfo && (
        <div className="rounded-lg bg-zinc-900 p-4">
          <div className="flex items-center gap-2 text-sm text-zinc-400">
            <Server className="h-4 w-4" />
            <span>{sysInfo.hostname}</span>
            <span className="mx-1">|</span>
            <span>
              {sysInfo.executor_mode === "slurm" ? "Cluster mode" : "Local mode"}
            </span>
            <span className="mx-1">|</span>
            <span>recovar {sysInfo.recovar_version}</span>
            {sysInfo.gpu_count > 0 && (
              <>
                <span className="mx-1">|</span>
                <span>{sysInfo.gpu_count} GPU{sysInfo.gpu_count > 1 ? "s" : ""}</span>
              </>
            )}
          </div>
        </div>
      )}

      {/* Active project info or Create/Open prompts */}
      {project ? (
        <ProjectDashboard
          project={project}
          showScan={showScan}
          setShowScan={setShowScan}
          scanPath={scanPath}
          setScanPath={setScanPath}
          showScanBrowser={showScanBrowser}
          setShowScanBrowser={setShowScanBrowser}
          scanMutation={scanMutation}
        />
      ) : (
        <div className="space-y-4">
          {/* No project: empty state with Create/Open */}
          <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-8 text-center">
            <FolderOpen className="mx-auto h-10 w-10 text-zinc-600" />
            <p className="mt-3 text-zinc-400">
              Create a project or open an existing one to get started.
            </p>
            <div className="mt-4 flex items-center justify-center gap-3">
              <Button onClick={() => { setShowCreate(true); setShowOpen(false); }}>
                <FolderPlus className="h-4 w-4" />
                Create Project
              </Button>
              <Button variant="outline" onClick={() => { setShowOpen(true); setShowCreate(false); }}>
                <FolderOpen className="h-4 w-4" />
                Open Project
              </Button>
            </div>
          </div>

          {/* Create Project form */}
          {showCreate && (
            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-6 space-y-3">
              <h3 className="text-sm font-medium">Create New Project</h3>
              <p className="text-xs text-zinc-400">
                Choose a directory for your project. A recovar_project.db will be created there.
              </p>
              <div className="space-y-1">
                <Label>Directory</Label>
                <div className="flex gap-2">
                  <Input
                    value={createPath}
                    onChange={(e) => setCreatePath(e.target.value)}
                    placeholder="/scratch/gpfs/GILLES/mg6942/my_project"
                    className="font-mono"
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowCreateBrowser(!showCreateBrowser)}
                  >
                    Browse
                  </Button>
                </div>
              </div>
              {showCreateBrowser && (
                <FileBrowser
                  initialPath="/scratch/gpfs/GILLES/mg6942"
                  selectDirectory
                  onSelect={(p) => {
                    setCreatePath(p);
                    if (!createName) setCreateName(p.split("/").pop() || "");
                    setShowCreateBrowser(false);
                  }}
                />
              )}
              {createPath && isEphemeralPath(createPath) && (
                <div className="flex items-start gap-2 rounded-md border border-amber-600/50 bg-amber-950/50 px-3 py-2 text-xs text-amber-200">
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-500" />
                  <span>{EPHEMERAL_PATH_WARNING}</span>
                </div>
              )}
              <div className="space-y-1">
                <Label>Project Name</Label>
                <Input
                  value={createName}
                  onChange={(e) => setCreateName(e.target.value)}
                  placeholder={createPath.split("/").pop() || "My Project"}
                />
              </div>
              {createMutation.isError && (
                <p className="text-sm text-red-400">{(createMutation.error as Error).message}</p>
              )}
              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={() => setShowCreate(false)}>Cancel</Button>
                <Button
                  onClick={() => createMutation.mutate()}
                  disabled={!createPath}
                  loading={createMutation.isPending}
                >
                  {createMutation.isPending ? "Creating..." : "Create Project"}
                </Button>
              </div>
            </div>
          )}

          {/* Open Project form */}
          {showOpen && (
            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-6 space-y-3">
              <h3 className="text-sm font-medium">Open Existing Project</h3>
              <p className="text-xs text-zinc-400">
                Select a directory containing a recovar project (with project.json or recovar_project.db).
              </p>
              <div className="space-y-1">
                <Label>Directory</Label>
                <div className="flex gap-2">
                  <Input
                    value={openPath}
                    onChange={(e) => setOpenPath(e.target.value)}
                    placeholder="/scratch/gpfs/.../my_project"
                    className="font-mono"
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowOpenBrowser(!showOpenBrowser)}
                  >
                    Browse
                  </Button>
                </div>
              </div>
              {showOpenBrowser && (
                <FileBrowser
                  initialPath="/scratch/gpfs/GILLES/mg6942"
                  selectDirectory
                  onSelect={(p) => { setOpenPath(p); setShowOpenBrowser(false); }}
                />
              )}
              {openPath && isEphemeralPath(openPath) && (
                <div className="flex items-start gap-2 rounded-md border border-amber-600/50 bg-amber-950/50 px-3 py-2 text-xs text-amber-200">
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-500" />
                  <span>{EPHEMERAL_PATH_WARNING}</span>
                </div>
              )}
              {openMutation.isError && (
                <p className="text-sm text-red-400">{(openMutation.error as Error).message}</p>
              )}
              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={() => setShowOpen(false)}>Cancel</Button>
                <Button
                  onClick={() => openMutation.mutate()}
                  disabled={!openPath}
                  loading={openMutation.isPending}
                >
                  {openMutation.isPending ? "Opening..." : "Open Project"}
                </Button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Project Dashboard — shown when a project is open
// ---------------------------------------------------------------------------

interface ProjectDashboardProps {
  project: { id: string; path: string; name: string };
  showScan: boolean;
  setShowScan: (v: boolean) => void;
  scanPath: string;
  setScanPath: (v: string) => void;
  showScanBrowser: boolean;
  setShowScanBrowser: (v: boolean) => void;
  scanMutation: UseMutationResult<{ imported: { id: string; type: string; status: string; output_dir: string; legacy: boolean }[] }, Error, void, unknown>;
}

function ProjectDashboard({
  project,
  showScan,
  setShowScan,
  scanPath,
  setScanPath,
  showScanBrowser,
  setShowScanBrowser,
  scanMutation,
}: ProjectDashboardProps): React.JSX.Element {
  const { data: projectDetail } = useQuery<ProjectDetail>({
    queryKey: ["project", project.id],
    queryFn: () => getProject(project.id),
    // Stop polling if the project was deleted (404).
    refetchInterval: (query) => {
      if (query.state.error instanceof ApiError && query.state.error.status === 404) {
        return false;
      }
      return 10000;
    },
  });

  const jobs = projectDetail?.jobs ?? [];
  const pipelineCount = jobs.filter((j) => j.type.toLowerCase() === "pipeline").length;
  const analyzeCount = jobs.filter((j) => j.type.toLowerCase() === "analyze").length;
  const otherCount = jobs.length - pipelineCount - analyzeCount;
  const runningCount = jobs.filter((j) => j.status === "running" || j.status === "queued").length;
  const lastModified = jobs.length > 0
    ? new Date(Math.max(...jobs.map((j) => new Date(j.completed ?? j.created).getTime()))).toLocaleString()
    : null;

  return (
    <div className="space-y-4">
      {/* Project header card */}
      <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2">
              <FolderOpen className="h-4 w-4 text-blue-400" />
              <span className="font-medium">{project.name}</span>
            </div>
            <p className="mt-1 font-mono text-xs text-zinc-500">{project.path}</p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowScan(!showScan)}
          >
            <Search className="h-3.5 w-3.5" />
            Scan for Existing Jobs
          </Button>
        </div>
      </div>

      {/* Job summary stats */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <div className="flex items-center gap-2 text-zinc-500">
            <Beaker className="h-4 w-4" />
            <span className="text-xs font-medium uppercase tracking-wider">Total Jobs</span>
          </div>
          <p className="mt-2 text-2xl font-semibold">{jobs.length}</p>
        </div>
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <div className="flex items-center gap-2 text-zinc-500">
            <Play className="h-4 w-4" />
            <span className="text-xs font-medium uppercase tracking-wider">Pipeline</span>
          </div>
          <p className="mt-2 text-2xl font-semibold">{pipelineCount}</p>
        </div>
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <div className="flex items-center gap-2 text-zinc-500">
            <BarChart3 className="h-4 w-4" />
            <span className="text-xs font-medium uppercase tracking-wider">Analyze</span>
          </div>
          <p className="mt-2 text-2xl font-semibold">{analyzeCount}</p>
        </div>
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <div className="flex items-center gap-2 text-zinc-500">
            <ScanSearch className="h-4 w-4" />
            <span className="text-xs font-medium uppercase tracking-wider">Other</span>
          </div>
          <p className="mt-2 text-2xl font-semibold">{otherCount}</p>
        </div>
      </div>

      {/* Running jobs indicator + last modified */}
      <div className="flex items-center gap-4 text-sm text-zinc-400">
        {runningCount > 0 && (
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 animate-pulse rounded-full bg-blue-500" />
            {runningCount} job{runningCount > 1 ? "s" : ""} running
          </span>
        )}
        {lastModified && (
          <span>Last activity: {lastModified}</span>
        )}
      </div>

      {/* Quick-launch cards */}
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <Link
          to="/jobs/new"
          search={{ type: "pipeline", result_dir: undefined, density: undefined, input: undefined, particles: undefined, params: undefined }}
          className="flex items-center gap-3 rounded-lg border border-zinc-800 bg-zinc-900 p-4 hover:border-blue-500/50 hover:bg-zinc-800"
        >
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-600/20">
            <Play className="h-5 w-5 text-blue-400" />
          </div>
          <div>
            <p className="text-sm font-medium">Run Pipeline</p>
            <p className="text-xs text-zinc-500">Process particles from .star file</p>
          </div>
        </Link>
        <button
          onClick={() => setShowScan(true)}
          className="flex items-center gap-3 rounded-lg border border-zinc-800 bg-zinc-900 p-4 text-left hover:border-blue-500/50 hover:bg-zinc-800"
        >
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-emerald-600/20">
            <ScanSearch className="h-5 w-5 text-emerald-400" />
          </div>
          <div>
            <p className="text-sm font-medium">Scan for Jobs</p>
            <p className="text-xs text-zinc-500">Import existing pipeline outputs</p>
          </div>
        </button>
      </div>

      {/* Scan form */}
      {showScan && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4 space-y-3">
          <h3 className="text-sm font-medium">Scan Directory for Pipeline Outputs</h3>
          <p className="text-xs text-zinc-400">
            Point to a directory containing existing recovar pipeline outputs. They will be imported into this project.
          </p>
          <div className="space-y-1">
            <Label>Scan Path</Label>
            <div className="flex gap-2">
              <Input
                value={scanPath}
                onChange={(e) => setScanPath(e.target.value)}
                placeholder="/scratch/gpfs/.../pipeline_output/"
                className="font-mono"
              />
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowScanBrowser(!showScanBrowser)}
              >
                Browse
              </Button>
            </div>
          </div>
          {showScanBrowser && (
            <FileBrowser
              initialPath={project.path}
              selectDirectory
              onSelect={(p) => { setScanPath(p); setShowScanBrowser(false); }}
            />
          )}
          {scanMutation.isError && (
            <p className="text-sm text-red-400">{(scanMutation.error as Error).message}</p>
          )}
          {scanMutation.isSuccess && (
            <p className="text-sm text-emerald-400">
              Imported {(scanMutation.data as { imported: unknown[] }).imported.length} job(s).
            </p>
          )}
          <div className="flex justify-end gap-2">
            <Button variant="outline" size="sm" onClick={() => setShowScan(false)}>
              Cancel
            </Button>
            <Button
              size="sm"
              onClick={() => scanMutation.mutate()}
              disabled={!scanPath}
              loading={scanMutation.isPending}
            >
              {scanMutation.isPending ? "Scanning..." : "Scan"}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
