import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronDown, ChevronRight, Plus, Trash2 } from "lucide-react";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { TooltipIcon } from "../ui/tooltip-icon";
import { getSystemInfo, type SystemInfo } from "../../lib/api/client";

export interface LocalOpts {
  gpus: string;
  setup_command: string;
  env_vars: Record<string, string>;
  preallocate_gpu: boolean;
}

interface LocalSettingsProps {
  value: LocalOpts | null;
  onChange: (opts: LocalOpts) => void;
}

/**
 * Settings panel for local GPU execution. Shows GPU picker,
 * setup command, and environment variable editor.
 */
export function LocalSettings({ value, onChange }: LocalSettingsProps): React.JSX.Element {
  const [expanded, setExpanded] = useState(false);

  const { data: sysInfo } = useQuery<SystemInfo>({
    queryKey: ["system-info"],
    queryFn: getSystemInfo,
    staleTime: 60_000,
  });

  const gpuList = (sysInfo as any)?.gpu_list as { index: number; name: string }[] | undefined;

  const current: LocalOpts = value ?? {
    gpus: "all",
    setup_command: "",
    env_vars: {},
    preallocate_gpu: true,
  };

  // Initialize value on first render
  useEffect(() => {
    if (!value) {
      onChange(current);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  function update(field: keyof LocalOpts, val: unknown): void {
    onChange({ ...current, [field]: val });
  }

  const envEntries = Object.entries(current.env_vars);

  return (
    <div>
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1 text-sm text-zinc-500 hover:text-zinc-300"
      >
        {expanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        Local GPU Settings
      </button>

      {expanded && (
        <div className="ml-4 mt-2 space-y-3 border-l border-zinc-800 pl-4">
          {/* GPU Selection */}
          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Label>GPUs to use</Label>
              <TooltipIcon text="Select which GPUs to use for this job. 'All' uses every available GPU. Pick specific GPUs to share the node with others." />
            </div>
            <div className="flex flex-wrap gap-1.5">
              <button
                type="button"
                onClick={() => update("gpus", "all")}
                className={`rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${
                  current.gpus === "all"
                    ? "border-emerald-500 bg-emerald-600/20 text-emerald-300"
                    : "border-zinc-700 text-zinc-400 hover:border-zinc-600 hover:text-zinc-300"
                }`}
              >
                All ({sysInfo?.gpu_count ?? "?"} GPUs)
              </button>
              {gpuList?.map((gpu) => {
                const id = String(gpu.index);
                const selected = current.gpus !== "all" && current.gpus.split(",").includes(id);
                return (
                  <button
                    key={gpu.index}
                    type="button"
                    onClick={() => {
                      if (current.gpus === "all") {
                        update("gpus", id);
                      } else {
                        const ids = current.gpus.split(",").filter(Boolean);
                        if (selected) {
                          const remaining = ids.filter((x) => x !== id);
                          update("gpus", remaining.length > 0 ? remaining.join(",") : "all");
                        } else {
                          update("gpus", [...ids, id].sort().join(","));
                        }
                      }
                    }}
                    className={`rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${
                      selected
                        ? "border-blue-500 bg-blue-600/20 text-blue-300"
                        : "border-zinc-700 text-zinc-400 hover:border-zinc-600 hover:text-zinc-300"
                    }`}
                  >
                    GPU {gpu.index}
                    <span className="ml-1 text-[10px] text-zinc-500">{gpu.name.replace(/NVIDIA /i, "")}</span>
                  </button>
                );
              })}
            </div>
            {!gpuList?.length && current.gpus !== "all" && (
              <Input
                value={current.gpus}
                onChange={(e) => update("gpus", e.target.value)}
                placeholder="0,1"
                className="mt-1"
              />
            )}
          </div>

          {/* Preallocate GPU memory */}
          <label className="flex items-center gap-2 text-sm text-zinc-400">
            <input
              type="checkbox"
              checked={current.preallocate_gpu !== false}
              onChange={(e) => update("preallocate_gpu", e.target.checked)}
              className="rounded border-zinc-600 bg-zinc-800"
            />
            Preallocate GPU memory
            <TooltipIcon text="Reserve one contiguous block of GPU memory up front (XLA_PYTHON_CLIENT_PREALLOCATE). Recommended ON for a dedicated GPU — it avoids fragmentation out-of-memory errors on large jobs (e.g. the box-256 PCA basis needs ~27 GB contiguous). Turn OFF only to share the GPU with other processes." />
          </label>

          {/* Setup Command */}
          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Label>Setup command</Label>
              <TooltipIcon text="Shell command to run before the pipeline starts. Use this to load modules or set up the environment. Example: module load cudatoolkit/12.8" />
            </div>
            <Input
              value={current.setup_command}
              onChange={(e) => update("setup_command", e.target.value)}
              placeholder="e.g. module load cudatoolkit/12.8"
            />
          </div>

          {/* Environment Variables */}
          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Label>Environment variables</Label>
              <TooltipIcon text="Extra environment variables passed to the pipeline process. These are set in addition to any defaults." />
            </div>
            <div className="space-y-1.5">
              {envEntries.map(([key, val], i) => (
                <div key={i} className="flex items-center gap-1.5">
                  <Input
                    value={key}
                    onChange={(e) => {
                      const newVars = { ...current.env_vars };
                      delete newVars[key];
                      newVars[e.target.value] = val;
                      update("env_vars", newVars);
                    }}
                    placeholder="KEY"
                    className="w-40 font-mono text-xs"
                  />
                  <span className="text-zinc-600">=</span>
                  <Input
                    value={val}
                    onChange={(e) => {
                      update("env_vars", { ...current.env_vars, [key]: e.target.value });
                    }}
                    placeholder="value"
                    className="flex-1 font-mono text-xs"
                  />
                  <button
                    type="button"
                    onClick={() => {
                      const newVars = { ...current.env_vars };
                      delete newVars[key];
                      update("env_vars", newVars);
                    }}
                    className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-red-400"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </div>
              ))}
              <button
                type="button"
                onClick={() => update("env_vars", { ...current.env_vars, "": "" })}
                className="flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
              >
                <Plus className="h-3 w-3" />
                Add variable
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
