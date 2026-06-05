import { useState, useEffect, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Save, CheckCircle, Info, Plus, Trash2 } from "lucide-react";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import { useProject } from "../lib/project-context";
import {
  getSlurmDefaultsLayered,
  updateUserSlurmDefaults,
  updateProjectSlurmDefaults,
  getLocalDefaultsLayered,
  updateUserLocalDefaults,
  updateProjectLocalDefaults,
  getSystemInfo,
  type SlurmDefaultsUpdate,
  type LocalDefaultsUpdate,
  type SystemInfo,
} from "../lib/api/client";

// ── SLURM defaults form ──

const SLURM_FIELDS: { key: keyof SlurmDefaultsUpdate; label: string; type: string; placeholder: string }[] = [
  { key: "partition", label: "Partition", type: "text", placeholder: "leave blank for cluster default" },
  { key: "account", label: "Account", type: "text", placeholder: "leave blank for cluster default" },
  { key: "gpus", label: "GPUs", type: "number", placeholder: "1" },
  { key: "cpus", label: "CPUs", type: "number", placeholder: "4" },
  { key: "memory", label: "Memory", type: "text", placeholder: "400G" },
  { key: "time", label: "Time Limit", type: "text", placeholder: "12:00:00" },
];

function provenance(
  key: string,
  data: { user: Record<string, unknown>; project: Record<string, unknown> },
): "built-in" | "user" | "project" {
  if (key in data.project) return "project";
  if (key in data.user) return "user";
  return "built-in";
}

const BADGE_STYLES = {
  "built-in": "bg-zinc-800 text-zinc-400",
  user: "bg-blue-900/50 text-blue-300",
  project: "bg-emerald-900/50 text-emerald-300",
} as const;

// ── Shared save button ──

function SaveButton({ onClick, saving, saved, label }: {
  onClick: () => void;
  saving: boolean;
  saved: boolean;
  label: string;
}): React.JSX.Element {
  return (
    <div className="flex items-center gap-3">
      <button
        onClick={onClick}
        disabled={saving}
        className="flex items-center gap-1.5 rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50"
      >
        <Save className="h-3.5 w-3.5" />
        {saving ? "Saving..." : label}
      </button>
      {saved && (
        <span className="flex items-center gap-1 text-xs text-emerald-400">
          <CheckCircle className="h-3 w-3" />
          Saved
        </span>
      )}
    </div>
  );
}

// ── Local defaults form ──

function LocalDefaultsForm({ title, description, filePath, values, onChange, onSave, saving, saved }: {
  title: string;
  description: string;
  filePath: string | null;
  values: LocalDefaultsUpdate;
  onChange: (v: LocalDefaultsUpdate) => void;
  onSave: () => void;
  saving: boolean;
  saved: boolean;
}): React.JSX.Element {
  const { data: sysInfo } = useQuery<SystemInfo>({
    queryKey: ["system-info"],
    queryFn: getSystemInfo,
    staleTime: 60_000,
  });
  const gpuList = sysInfo?.gpu_list;
  const envEntries = Object.entries(values.env_vars ?? {});

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-5">
      <div className="mb-4">
        <h3 className="text-sm font-medium text-zinc-200">{title}</h3>
        <p className="mt-1 text-xs text-zinc-500">{description}</p>
        {filePath && <p className="mt-1 font-mono text-xs text-zinc-600">{filePath}</p>}
      </div>

      {/* GPU selection */}
      <div className="space-y-1">
        <Label className="text-xs">GPUs to use</Label>
        <div className="flex flex-wrap gap-1.5">
          <button
            type="button"
            onClick={() => onChange({ ...values, gpus: "all" })}
            className={`rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${
              (values.gpus ?? "all") === "all"
                ? "border-emerald-500 bg-emerald-600/20 text-emerald-300"
                : "border-zinc-700 text-zinc-400 hover:border-zinc-600"
            }`}
          >
            All ({sysInfo?.gpu_count ?? "?"} GPUs)
          </button>
          {gpuList?.map((gpu) => {
            const id = String(gpu.index);
            const gpus = values.gpus ?? "all";
            const selected = gpus !== "all" && gpus.split(",").includes(id);
            return (
              <button
                key={gpu.index}
                type="button"
                onClick={() => {
                  if (gpus === "all") {
                    onChange({ ...values, gpus: id });
                  } else {
                    const ids = gpus.split(",").filter(Boolean);
                    if (selected) {
                      const remaining = ids.filter((x) => x !== id);
                      onChange({ ...values, gpus: remaining.length > 0 ? remaining.join(",") : "all" });
                    } else {
                      onChange({ ...values, gpus: [...ids, id].sort().join(",") });
                    }
                  }
                }}
                className={`rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${
                  selected
                    ? "border-blue-500 bg-blue-600/20 text-blue-300"
                    : "border-zinc-700 text-zinc-400 hover:border-zinc-600"
                }`}
              >
                GPU {gpu.index} <span className="ml-1 text-[10px] text-zinc-500">{gpu.name.replace(/NVIDIA /i, "")}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Setup command */}
      <div className="mt-3 space-y-1">
        <Label className="text-xs">Setup command</Label>
        <Input
          value={values.setup_command ?? ""}
          onChange={(e) => onChange({ ...values, setup_command: e.target.value })}
          placeholder="e.g. module load cudatoolkit/12.8"
        />
        <p className="text-[10px] text-zinc-600">Runs before the pipeline. Use for loading modules or environment setup.</p>
      </div>

      {/* Environment variables */}
      <div className="mt-3 space-y-1">
        <Label className="text-xs">Environment variables</Label>
        <div className="space-y-1.5">
          {envEntries.map(([key, val], i) => (
            <div key={i} className="flex items-center gap-1.5">
              <Input
                value={key}
                onChange={(e) => {
                  const newVars = { ...(values.env_vars ?? {}) };
                  delete newVars[key];
                  newVars[e.target.value] = val;
                  onChange({ ...values, env_vars: newVars });
                }}
                placeholder="KEY"
                className="w-40 font-mono text-xs"
              />
              <span className="text-zinc-600">=</span>
              <Input
                value={val}
                onChange={(e) => onChange({ ...values, env_vars: { ...(values.env_vars ?? {}), [key]: e.target.value } })}
                placeholder="value"
                className="flex-1 font-mono text-xs"
              />
              <button
                type="button"
                onClick={() => {
                  const newVars = { ...(values.env_vars ?? {}) };
                  delete newVars[key];
                  onChange({ ...values, env_vars: newVars });
                }}
                className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-red-400"
              >
                <Trash2 className="h-3 w-3" />
              </button>
            </div>
          ))}
          <button
            type="button"
            onClick={() => onChange({ ...values, env_vars: { ...(values.env_vars ?? {}), "": "" } })}
            className="flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
          >
            <Plus className="h-3 w-3" />
            Add variable
          </button>
        </div>
      </div>

      <div className="mt-4">
        <SaveButton onClick={onSave} saving={saving} saved={saved} label={`Save ${title.includes("User") ? "User" : "Project"} Defaults`} />
      </div>
    </div>
  );
}

// ── Main settings page ──

export function SettingsPage(): React.JSX.Element {
  const { project } = useProject();
  const queryClient = useQueryClient();
  const [tab, setTab] = useState<"slurm" | "local">("slurm");

  // Track save-confirmation timers so they can be cleared on unmount.
  const savedTimers = useRef<ReturnType<typeof setTimeout>[]>([]);
  useEffect(() => () => savedTimers.current.forEach(clearTimeout), []);
  const flashSaved = (setSaved: (v: boolean) => void) => {
    setSaved(true);
    savedTimers.current.push(setTimeout(() => setSaved(false), 3000));
  };

  // SLURM data
  const { data: slurmData, isLoading: slurmLoading } = useQuery({
    queryKey: ["slurm-defaults-layered", project?.path],
    queryFn: () => getSlurmDefaultsLayered(project?.path),
  });

  // Local data
  const { data: localData, isLoading: localLoading } = useQuery({
    queryKey: ["local-defaults-layered", project?.path],
    queryFn: () => getLocalDefaultsLayered(project?.path),
  });

  // SLURM form state
  const [slurmUserValues, setSlurmUserValues] = useState<SlurmDefaultsUpdate | null>(null);
  const [slurmUserSaved, setSlurmUserSaved] = useState(false);
  const [slurmProjectValues, setSlurmProjectValues] = useState<SlurmDefaultsUpdate | null>(null);
  const [slurmProjectSaved, setSlurmProjectSaved] = useState(false);

  // Local form state
  const [localUserValues, setLocalUserValues] = useState<LocalDefaultsUpdate | null>(null);
  const [localUserSaved, setLocalUserSaved] = useState(false);
  const [localProjectValues, setLocalProjectValues] = useState<LocalDefaultsUpdate | null>(null);
  const [localProjectSaved, setLocalProjectSaved] = useState(false);

  // Initialize SLURM form values
  if (slurmData && slurmUserValues === null) {
    setSlurmUserValues({
      partition: String(slurmData.user.partition ?? ""),
      account: String(slurmData.user.account ?? ""),
      gpus: slurmData.user.gpus != null ? Number(slurmData.user.gpus) : undefined,
      cpus: slurmData.user.cpus != null ? Number(slurmData.user.cpus) : undefined,
      memory: String(slurmData.user.memory ?? ""),
      time: String(slurmData.user.time ?? ""),
    });
  }
  if (slurmData && project && slurmProjectValues === null) {
    setSlurmProjectValues({
      partition: String(slurmData.project.partition ?? ""),
      account: String(slurmData.project.account ?? ""),
      gpus: slurmData.project.gpus != null ? Number(slurmData.project.gpus) : undefined,
      cpus: slurmData.project.cpus != null ? Number(slurmData.project.cpus) : undefined,
      memory: String(slurmData.project.memory ?? ""),
      time: String(slurmData.project.time ?? ""),
    });
  }

  // Initialize Local form values
  if (localData && localUserValues === null) {
    setLocalUserValues({
      gpus: String(localData.user.gpus ?? ""),
      setup_command: String(localData.user.setup_command ?? ""),
      env_vars: (localData.user.env_vars as Record<string, string>) ?? {},
    });
  }
  if (localData && project && localProjectValues === null) {
    setLocalProjectValues({
      gpus: String(localData.project.gpus ?? ""),
      setup_command: String(localData.project.setup_command ?? ""),
      env_vars: (localData.project.env_vars as Record<string, string>) ?? {},
    });
  }

  // Mutations
  const slurmUserMutation = useMutation({
    mutationFn: (vals: SlurmDefaultsUpdate) => updateUserSlurmDefaults(vals, project?.path),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["slurm-defaults"] });
      queryClient.invalidateQueries({ queryKey: ["slurm-defaults-layered"] });
      flashSaved(setSlurmUserSaved);
    },
  });
  const slurmProjectMutation = useMutation({
    mutationFn: (vals: SlurmDefaultsUpdate) => updateProjectSlurmDefaults(project!.path, vals),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["slurm-defaults"] });
      queryClient.invalidateQueries({ queryKey: ["slurm-defaults-layered"] });
      flashSaved(setSlurmProjectSaved);
    },
  });
  const localUserMutation = useMutation({
    mutationFn: (vals: LocalDefaultsUpdate) => updateUserLocalDefaults(vals, project?.path),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["local-defaults-layered"] });
      flashSaved(setLocalUserSaved);
    },
  });
  const localProjectMutation = useMutation({
    mutationFn: (vals: LocalDefaultsUpdate) => updateProjectLocalDefaults(project!.path, vals),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["local-defaults-layered"] });
      flashSaved(setLocalProjectSaved);
    },
  });

  if ((slurmLoading && localLoading) || (!slurmData && !localData)) {
    return (
      <div className="space-y-4">
        <h1 className="text-xl font-semibold">Settings</h1>
        <p className="text-sm text-zinc-500">Loading...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Settings</h1>
        <p className="mt-1 text-sm text-zinc-500">
          Configure default settings for job submissions. Set your cluster account details and local GPU preferences here so you don't have to fill them in every time.
        </p>
      </div>

      {/* Tab selector */}
      <div className="flex gap-1 rounded-lg bg-zinc-900 p-1">
        <button
          onClick={() => setTab("slurm")}
          className={`flex-1 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
            tab === "slurm"
              ? "bg-zinc-700 text-zinc-100"
              : "text-zinc-400 hover:text-zinc-200"
          }`}
        >
          SLURM Cluster
        </button>
        <button
          onClick={() => setTab("local")}
          className={`flex-1 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
            tab === "local"
              ? "bg-zinc-700 text-zinc-100"
              : "text-zinc-400 hover:text-zinc-200"
          }`}
        >
          Local GPU
        </button>
      </div>

      {/* ── SLURM tab ── */}
      {tab === "slurm" && slurmData && (
        <div className="space-y-4">
          {/* Effective summary */}
          <div className="rounded-lg border border-zinc-700/50 bg-zinc-900/30 p-4">
            <div className="mb-2 flex items-center gap-2">
              <Info className="h-4 w-4 text-zinc-500" />
              <h3 className="text-sm font-medium text-zinc-300">Effective SLURM Defaults</h3>
            </div>
            <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-zinc-400">
              {SLURM_FIELDS.map(({ key, label }) => {
                const val = slurmData.effective[key];
                const src = provenance(key, slurmData);
                return (
                  <span key={key}>
                    <span className="text-zinc-500">{label}:</span>{" "}
                    <span className="text-zinc-200">{val === "" ? "(cluster default)" : String(val)}</span>
                    {src !== "built-in" && (
                      <span className={`ml-1 rounded px-1 py-0.5 text-[9px] ${BADGE_STYLES[src]}`}>{src}</span>
                    )}
                  </span>
                );
              })}
            </div>
          </div>

          {/* User-global SLURM */}
          {slurmUserValues && (
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-5">
              <div className="mb-4">
                <h3 className="text-sm font-medium text-zinc-200">User-Global Defaults</h3>
                <p className="mt-1 text-xs text-zinc-500">Apply to all projects. Stored in your home directory.</p>
                <p className="mt-1 font-mono text-xs text-zinc-600">{slurmData.user_config_path}</p>
              </div>
              <div className="grid grid-cols-2 gap-3">
                {SLURM_FIELDS.slice(0, 2).map(({ key, label, placeholder }) => (
                  <div key={key} className="space-y-1">
                    <Label className="text-xs">{label}</Label>
                    <Input value={String(slurmUserValues[key] ?? "")} placeholder={placeholder} onChange={(e) => setSlurmUserValues({ ...slurmUserValues, [key]: e.target.value })} />
                  </div>
                ))}
              </div>
              <div className="mt-3 grid grid-cols-4 gap-3">
                {SLURM_FIELDS.slice(2).map(({ key, label, type, placeholder }) => (
                  <div key={key} className="space-y-1">
                    <Label className="text-xs">{label}</Label>
                    <Input type={type} value={String(slurmUserValues[key] ?? "")} placeholder={placeholder} onChange={(e) => { const n = parseInt(e.target.value, 10); setSlurmUserValues({ ...slurmUserValues, [key]: type === "number" ? (e.target.value === "" ? undefined : (Number.isNaN(n) ? undefined : n)) : e.target.value }); }} />
                  </div>
                ))}
              </div>
              <div className="mt-4">
                <SaveButton onClick={() => slurmUserMutation.mutate(slurmUserValues)} saving={slurmUserMutation.isPending} saved={slurmUserSaved} label="Save User Defaults" />
              </div>
            </div>
          )}

          {/* Project SLURM */}
          {project && slurmProjectValues ? (
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-5">
              <div className="mb-4">
                <h3 className="text-sm font-medium text-zinc-200">Project Defaults: {project.name}</h3>
                <p className="mt-1 text-xs text-zinc-500">Override user defaults for this project only.</p>
              </div>
              <div className="grid grid-cols-2 gap-3">
                {SLURM_FIELDS.slice(0, 2).map(({ key, label, placeholder }) => (
                  <div key={key} className="space-y-1">
                    <Label className="text-xs">{label}</Label>
                    <Input value={String(slurmProjectValues[key] ?? "")} placeholder={placeholder} onChange={(e) => setSlurmProjectValues({ ...slurmProjectValues, [key]: e.target.value })} />
                  </div>
                ))}
              </div>
              <div className="mt-3 grid grid-cols-4 gap-3">
                {SLURM_FIELDS.slice(2).map(({ key, label, type, placeholder }) => (
                  <div key={key} className="space-y-1">
                    <Label className="text-xs">{label}</Label>
                    <Input type={type} value={String(slurmProjectValues[key] ?? "")} placeholder={placeholder} onChange={(e) => { const n = parseInt(e.target.value, 10); setSlurmProjectValues({ ...slurmProjectValues, [key]: type === "number" ? (e.target.value === "" ? undefined : (Number.isNaN(n) ? undefined : n)) : e.target.value }); }} />
                  </div>
                ))}
              </div>
              <div className="mt-4">
                <SaveButton onClick={() => slurmProjectMutation.mutate(slurmProjectValues)} saving={slurmProjectMutation.isPending} saved={slurmProjectSaved} label="Save Project Defaults" />
              </div>
            </div>
          ) : (
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/30 p-5">
              <h3 className="text-sm font-medium text-zinc-400">Project Defaults</h3>
              <p className="mt-1 text-xs text-zinc-600">Open a project to configure project-specific defaults.</p>
            </div>
          )}
        </div>
      )}

      {/* ── Local GPU tab ── */}
      {tab === "local" && (
        <div className="space-y-4">
          {/* Effective summary */}
          {localData && (
            <div className="rounded-lg border border-zinc-700/50 bg-zinc-900/30 p-4">
              <div className="mb-2 flex items-center gap-2">
                <Info className="h-4 w-4 text-zinc-500" />
                <h3 className="text-sm font-medium text-zinc-300">Effective Local GPU Defaults</h3>
              </div>
              <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-zinc-400">
                <span>
                  <span className="text-zinc-500">GPUs:</span>{" "}
                  <span className="text-zinc-200">{String(localData.effective.gpus ?? "all")}</span>
                </span>
                <span>
                  <span className="text-zinc-500">Setup:</span>{" "}
                  <span className="text-zinc-200">{localData.effective.setup_command ? String(localData.effective.setup_command) : "(none)"}</span>
                </span>
                {Object.keys(localData.effective.env_vars as Record<string, string> ?? {}).length > 0 && (
                  <span>
                    <span className="text-zinc-500">Env vars:</span>{" "}
                    <span className="text-zinc-200">{Object.keys(localData.effective.env_vars as Record<string, string>).join(", ")}</span>
                  </span>
                )}
              </div>
            </div>
          )}

          {/* User-global Local */}
          {localUserValues && (
            <LocalDefaultsForm
              title="User-Global Defaults"
              description="Apply to all local GPU jobs. Stored in your home directory."
              filePath={localData?.user_config_path ?? null}
              values={localUserValues}
              onChange={setLocalUserValues}
              onSave={() => localUserMutation.mutate(localUserValues)}
              saving={localUserMutation.isPending}
              saved={localUserSaved}
            />
          )}

          {/* Project Local */}
          {project && localProjectValues ? (
            <LocalDefaultsForm
              title={`Project Defaults: ${project.name}`}
              description="Override user defaults for local GPU jobs in this project."
              filePath={localData?.project_config_path ?? null}
              values={localProjectValues}
              onChange={setLocalProjectValues}
              onSave={() => localProjectMutation.mutate(localProjectValues)}
              saving={localProjectMutation.isPending}
              saved={localProjectSaved}
            />
          ) : (
            <div className="rounded-lg border border-zinc-800 bg-zinc-900/30 p-5">
              <h3 className="text-sm font-medium text-zinc-400">Project Defaults</h3>
              <p className="mt-1 text-xs text-zinc-600">Open a project to configure project-specific local GPU defaults.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
