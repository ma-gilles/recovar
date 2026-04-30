import { useState, useEffect, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronDown, ChevronRight, Save, Code2, Eye } from "lucide-react";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { TooltipIcon } from "../ui/tooltip-icon";
import { tooltips } from "../../lib/tooltips";
import {
  getSlurmDefaults as getServerDefaults,
  getSystemInfo,
  previewSbatchScript,
  type SlurmDefaults,
  type SbatchPreview,
} from "../../lib/api/client";
import { useProject } from "../../lib/project-context";
import {
  getSlurmDefaults as getLocalDefaults,
  saveSlurmDefaults as saveLocalDefaults,
} from "../../lib/user-prefs";

export interface SlurmOpts {
  partition: string;
  account: string;
  gpus: number;
  cpus: number;
  memory: string;
  time: string;
  raw_directives?: string;
}

interface SlurmSettingsProps {
  value: SlurmOpts | null;
  onChange: (opts: SlurmOpts | null) => void;
}

/**
 * Collapsible SLURM settings section for job submission forms.
 *
 * Priority for initial values:
 *   1. Per-project saved defaults (localStorage)
 *   2. Server-reported defaults
 *   3. Hard-coded fallbacks
 *
 * Includes an optional raw directives editor for advanced users.
 */
export function SlurmSettings({ value, onChange }: SlurmSettingsProps): React.JSX.Element | null {
  const [expanded, setExpanded] = useState(false);
  const [showRawEditor, setShowRawEditor] = useState(false);
  const [savedToast, setSavedToast] = useState(false);
  const [showPreview, setShowPreview] = useState(false);
  const [preview, setPreview] = useState<SbatchPreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  const { project } = useProject();
  const projectPath = project?.path ?? "";

  const { data: sysInfo } = useQuery({
    queryKey: ["system-info"],
    queryFn: getSystemInfo,
    staleTime: 60_000,
  });

  // Pass projectPath so the server merges per-project recovar.toml on top
  // of its built-in defaults.
  const { data: serverDefaults } = useQuery<SlurmDefaults>({
    queryKey: ["slurm-defaults", projectPath],
    queryFn: () => getServerDefaults(projectPath || undefined),
    staleTime: 60_000,
    enabled: sysInfo?.slurm_available === true,
  });

  // Initialize from per-project localStorage, then server defaults
  useEffect(() => {
    if (value !== null) return;

    // Try localStorage first
    if (projectPath) {
      const local = getLocalDefaults(projectPath);
      if (local) {
        onChange(local);
        if (local.raw_directives) setShowRawEditor(true);
        return;
      }
    }

    // Fall back to server defaults
    if (serverDefaults) {
      onChange({
        partition: serverDefaults.partition,
        account: serverDefaults.account,
        gpus: serverDefaults.gpus,
        cpus: serverDefaults.cpus,
        memory: serverDefaults.memory,
        time: serverDefaults.time,
      });
    }
  }, [serverDefaults, value, onChange, projectPath]);

  const handleSaveDefaults = useCallback(() => {
    if (!projectPath || !value) return;
    saveLocalDefaults(projectPath, value);
    setSavedToast(true);
    setTimeout(() => setSavedToast(false), 2000);
  }, [projectPath, value]);

  // Render the would-be submit.sh on demand. We don't know the user's exact
  // command yet (that's built by the parent form when they pick a job type),
  // so we show a placeholder argv that demonstrates the wrapper structure.
  const handlePreview = useCallback(async () => {
    setShowPreview(true);
    setPreviewLoading(true);
    setPreviewError(null);
    try {
      const result = await previewSbatchScript({
        command: ["recovar", "<job-type>", "<args...>"],
        env_vars: {},
        output_path: "<job_dir>/slurm-%j.out",
        job_name: "recovar-preview",
        slurm_opts: value ? { ...value } : {},
      });
      setPreview(result);
    } catch (e) {
      setPreviewError(e instanceof Error ? e.message : String(e));
      setPreview(null);
    } finally {
      setPreviewLoading(false);
    }
  }, [value]);

  // Don't render if SLURM is not available
  if (!sysInfo?.slurm_available) {
    return null;
  }

  // Empty partition/account → backend renderer omits the directive entirely,
  // letting the cluster's default apply. Do NOT bake site-specific defaults
  // here; doing so leaks them into every user's form regardless of cluster.
  const current = value ?? {
    partition: serverDefaults?.partition ?? "",
    account: serverDefaults?.account ?? "",
    gpus: serverDefaults?.gpus ?? 1,
    cpus: serverDefaults?.cpus ?? 4,
    memory: serverDefaults?.memory ?? "300G",
    time: serverDefaults?.time ?? "12:00:00",
  };

  function update(field: keyof SlurmOpts, val: string | number): void {
    onChange({ ...current, [field]: val });
  }

  return (
    <div>
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1 text-sm text-zinc-500 hover:text-zinc-300"
      >
        {expanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        SLURM Settings
      </button>

      {expanded && (
        <div className="ml-4 mt-2 space-y-3 border-l border-zinc-800 pl-4">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <div className="flex items-center gap-1">
                <Label>Partition</Label>
                <TooltipIcon text={tooltips["slurm.partition"]} />
              </div>
              <Input
                value={current.partition}
                placeholder="leave blank to use cluster default"
                onChange={(e) => update("partition", e.target.value)}
              />
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-1">
                <Label>Account</Label>
                <TooltipIcon text={tooltips["slurm.account"]} />
              </div>
              <Input
                value={current.account}
                placeholder="leave blank to use cluster default"
                onChange={(e) => update("account", e.target.value)}
              />
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div className="space-y-1">
              <div className="flex items-center gap-1">
                <Label>GPUs</Label>
                <TooltipIcon text={tooltips["slurm.gpus"]} />
              </div>
              <Input
                type="number"
                min={0}
                value={current.gpus}
                onChange={(e) => update("gpus", parseInt(e.target.value) || 0)}
              />
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-1">
                <Label>CPUs</Label>
                <TooltipIcon text={tooltips["slurm.cpus"]} />
              </div>
              <Input
                type="number"
                min={1}
                value={current.cpus}
                onChange={(e) => update("cpus", parseInt(e.target.value) || 1)}
              />
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-1">
                <Label>Memory</Label>
                <TooltipIcon text={tooltips["slurm.memory"]} />
              </div>
              <Input
                value={current.memory}
                onChange={(e) => update("memory", e.target.value)}
                placeholder="300G"
              />
            </div>
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Label>Time Limit</Label>
              <TooltipIcon text={tooltips["slurm.time"]} />
            </div>
            <Input
              value={current.time}
              onChange={(e) => update("time", e.target.value)}
              placeholder="12:00:00"
            />
          </div>

          {/* Raw directives editor */}
          <div>
            <button
              onClick={() => setShowRawEditor(!showRawEditor)}
              className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300"
            >
              <Code2 className="h-3 w-3" />
              {showRawEditor ? "Hide" : "Show"} extra SBATCH directives
            </button>
            {showRawEditor && (
              <div className="mt-2 space-y-1">
                <Label className="text-xs text-zinc-500">
                  Additional #SBATCH lines (one per line, without the #SBATCH prefix)
                </Label>
                <textarea
                  value={current.raw_directives ?? ""}
                  onChange={(e) => update("raw_directives", e.target.value)}
                  placeholder={"--constraint=gpu80\n--mail-type=END\n--mail-user=you@example.com"}
                  rows={3}
                  className="w-full rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 font-mono text-xs text-zinc-200 placeholder:text-zinc-600 focus:border-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-500"
                />
              </div>
            )}
          </div>

          {/* Save as default for this project */}
          {projectPath && (
            <div className="flex items-center gap-2">
              <button
                onClick={handleSaveDefaults}
                className="flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
              >
                <Save className="h-3 w-3" />
                Save as default for this project
              </button>
              {savedToast && (
                <span className="text-xs text-emerald-400">Saved</span>
              )}
            </div>
          )}

          {/* Preview generated submit.sh */}
          <div>
            <button
              onClick={() => (showPreview ? setShowPreview(false) : handlePreview())}
              className="flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
            >
              <Eye className="h-3 w-3" />
              {showPreview ? "Hide" : "Preview"} generated submit.sh
            </button>
            {showPreview && (
              <div className="mt-2 space-y-2">
                {previewLoading && (
                  <div className="text-xs text-zinc-500">Rendering…</div>
                )}
                {previewError && (
                  <div className="rounded border border-red-800 bg-red-950/40 px-2 py-1 text-xs text-red-300">
                    Failed to render preview: {previewError}
                  </div>
                )}
                {preview && preview.warnings.length > 0 && (
                  <ul className="space-y-1 text-xs text-amber-400">
                    {preview.warnings.map((w, i) => (
                      <li key={i}>• {w}</li>
                    ))}
                  </ul>
                )}
                {preview && (
                  <pre className="max-h-80 overflow-auto rounded border border-zinc-800 bg-zinc-950 px-3 py-2 font-mono text-xs text-zinc-300">
                    {preview.script}
                  </pre>
                )}
                <p className="text-xs text-zinc-600">
                  Note: the actual command line is filled in at submit time.
                  This preview uses a placeholder so you can verify the
                  surrounding wrapper (directives, modules, paths).
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
