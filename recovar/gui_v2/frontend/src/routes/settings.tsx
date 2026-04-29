import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Save, CheckCircle, Info } from "lucide-react";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import { useProject } from "../lib/project-context";
import {
  getSlurmDefaultsLayered,
  updateUserSlurmDefaults,
  updateProjectSlurmDefaults,
  type SlurmDefaultsLayered,
  type SlurmDefaultsUpdate,
} from "../lib/api/client";

const SLURM_FIELDS: { key: keyof SlurmDefaultsUpdate; label: string; type: string; placeholder: string }[] = [
  { key: "partition", label: "Partition", type: "text", placeholder: "leave blank for cluster default" },
  { key: "account", label: "Account", type: "text", placeholder: "leave blank for cluster default" },
  { key: "gpus", label: "GPUs", type: "number", placeholder: "1" },
  { key: "cpus", label: "CPUs", type: "number", placeholder: "4" },
  { key: "memory", label: "Memory", type: "text", placeholder: "300G" },
  { key: "time", label: "Time Limit", type: "text", placeholder: "12:00:00" },
];

function provenance(
  key: string,
  data: SlurmDefaultsLayered,
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

interface DefaultsFormProps {
  title: string;
  description: string;
  filePath: string | null;
  values: SlurmDefaultsUpdate;
  onChange: (values: SlurmDefaultsUpdate) => void;
  onSave: () => void;
  saving: boolean;
  saved: boolean;
  data: SlurmDefaultsLayered;
  layer: "user" | "project";
}

function DefaultsForm({
  title,
  description,
  filePath,
  values,
  onChange,
  onSave,
  saving,
  saved,
  data,
  layer,
}: DefaultsFormProps): React.JSX.Element {
  function update(key: keyof SlurmDefaultsUpdate, val: string): void {
    const parsed = key === "gpus" || key === "cpus" ? parseInt(val) || 0 : val;
    onChange({ ...values, [key]: parsed });
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-5">
      <div className="mb-4">
        <h3 className="text-sm font-medium text-zinc-200">{title}</h3>
        <p className="mt-1 text-xs text-zinc-500">{description}</p>
        {filePath && (
          <p className="mt-1 font-mono text-xs text-zinc-600">{filePath}</p>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3">
        {SLURM_FIELDS.slice(0, 2).map(({ key, label, placeholder }) => (
          <div key={key} className="space-y-1">
            <div className="flex items-center gap-2">
              <Label className="text-xs">{label}</Label>
              <span className={`rounded px-1.5 py-0.5 text-[10px] ${BADGE_STYLES[provenance(key, data)]}`}>
                {provenance(key, data)}
              </span>
            </div>
            <Input
              value={String(values[key] ?? "")}
              placeholder={placeholder}
              onChange={(e) => update(key, e.target.value)}
            />
          </div>
        ))}
      </div>

      <div className="mt-3 grid grid-cols-4 gap-3">
        {SLURM_FIELDS.slice(2).map(({ key, label, type, placeholder }) => (
          <div key={key} className="space-y-1">
            <div className="flex items-center gap-2">
              <Label className="text-xs">{label}</Label>
              <span className={`rounded px-1.5 py-0.5 text-[10px] ${BADGE_STYLES[provenance(key, data)]}`}>
                {provenance(key, data)}
              </span>
            </div>
            <Input
              type={type}
              value={String(values[key] ?? "")}
              placeholder={placeholder}
              onChange={(e) => update(key, e.target.value)}
              min={type === "number" ? 0 : undefined}
            />
          </div>
        ))}
      </div>

      <div className="mt-4 flex items-center gap-3">
        <button
          onClick={onSave}
          disabled={saving}
          className="flex items-center gap-1.5 rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50"
        >
          <Save className="h-3.5 w-3.5" />
          {saving ? "Saving..." : `Save ${layer === "user" ? "User" : "Project"} Defaults`}
        </button>
        {saved && (
          <span className="flex items-center gap-1 text-xs text-emerald-400">
            <CheckCircle className="h-3 w-3" />
            Saved
          </span>
        )}
      </div>
    </div>
  );
}

export function SettingsPage(): React.JSX.Element {
  const { project } = useProject();
  const queryClient = useQueryClient();

  const { data, isLoading } = useQuery({
    queryKey: ["slurm-defaults-layered", project?.path],
    queryFn: () => getSlurmDefaultsLayered(project?.path),
  });

  // User defaults form state
  const [userValues, setUserValues] = useState<SlurmDefaultsUpdate | null>(null);
  const [userSaved, setUserSaved] = useState(false);

  // Project defaults form state
  const [projectValues, setProjectValues] = useState<SlurmDefaultsUpdate | null>(null);
  const [projectSaved, setProjectSaved] = useState(false);

  // Initialize form values from server data
  if (data && userValues === null) {
    setUserValues({
      partition: String(data.user.partition ?? ""),
      account: String(data.user.account ?? ""),
      gpus: Number(data.user.gpus ?? ""),
      cpus: Number(data.user.cpus ?? ""),
      memory: String(data.user.memory ?? ""),
      time: String(data.user.time ?? ""),
    });
  }
  if (data && project && projectValues === null) {
    setProjectValues({
      partition: String(data.project.partition ?? ""),
      account: String(data.project.account ?? ""),
      gpus: Number(data.project.gpus ?? ""),
      cpus: Number(data.project.cpus ?? ""),
      memory: String(data.project.memory ?? ""),
      time: String(data.project.time ?? ""),
    });
  }

  const userMutation = useMutation({
    mutationFn: (vals: SlurmDefaultsUpdate) =>
      updateUserSlurmDefaults(vals, project?.path),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["slurm-defaults-layered"] });
      queryClient.invalidateQueries({ queryKey: ["slurm-defaults"] });
      setUserSaved(true);
      setTimeout(() => setUserSaved(false), 3000);
    },
  });

  const projectMutation = useMutation({
    mutationFn: (vals: SlurmDefaultsUpdate) =>
      updateProjectSlurmDefaults(project!.path, vals),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["slurm-defaults-layered"] });
      queryClient.invalidateQueries({ queryKey: ["slurm-defaults"] });
      setProjectSaved(true);
      setTimeout(() => setProjectSaved(false), 3000);
    },
  });

  if (isLoading || !data) {
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
          Configure default SLURM settings for job submissions. Values are layered: user-global defaults apply everywhere, project defaults override for that project, and per-job form overrides both.
        </p>
      </div>

      {/* Effective values summary */}
      <div className="rounded-lg border border-zinc-700/50 bg-zinc-900/30 p-4">
        <div className="mb-2 flex items-center gap-2">
          <Info className="h-4 w-4 text-zinc-500" />
          <h3 className="text-sm font-medium text-zinc-300">Effective Defaults</h3>
          <span className="text-xs text-zinc-500">(what new job forms will show)</span>
        </div>
        <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-zinc-400">
          {SLURM_FIELDS.map(({ key, label }) => {
            const val = data.effective[key];
            const src = provenance(key, data);
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

      {/* User-global defaults */}
      {userValues && (
        <DefaultsForm
          title="User-Global Defaults"
          description="Apply to all projects for your user account. Stored in your home directory."
          filePath={data.user_config_path}
          values={userValues}
          onChange={setUserValues}
          onSave={() => userMutation.mutate(userValues)}
          saving={userMutation.isPending}
          saved={userSaved}
          data={data}
          layer="user"
        />
      )}

      {/* Per-project defaults */}
      {project && projectValues ? (
        <DefaultsForm
          title={`Project Defaults: ${project.name}`}
          description="Override user-global defaults for this project only. Stored in the project directory."
          filePath={data.project_config_path}
          values={projectValues}
          onChange={setProjectValues}
          onSave={() => projectMutation.mutate(projectValues)}
          saving={projectMutation.isPending}
          saved={projectSaved}
          data={data}
          layer="project"
        />
      ) : (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/30 p-5">
          <h3 className="text-sm font-medium text-zinc-400">Project Defaults</h3>
          <p className="mt-1 text-xs text-zinc-600">
            Open a project to configure project-specific SLURM defaults.
          </p>
        </div>
      )}
    </div>
  );
}
