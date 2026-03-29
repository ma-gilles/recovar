import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import {
  createProject,
  scanProject,
  getSystemInfo,
  type SystemInfo,
} from "../lib/api/client";
import { useProject } from "../lib/project-context";
import { Plus, Server, FolderPlus, FolderOpen, Search } from "lucide-react";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import { FileBrowser } from "../components/file-browser/FileBrowser";

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
        <div className="space-y-4">
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
