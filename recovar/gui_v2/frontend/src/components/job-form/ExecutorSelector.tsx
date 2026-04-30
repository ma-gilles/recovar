import { useQuery } from "@tanstack/react-query";
import { getSystemInfo } from "../../lib/api/client";
import { Label } from "../ui/label";
import { TooltipIcon } from "../ui/tooltip-icon";

interface ExecutorSelectorProps {
  value: string | null;
  onChange: (value: string | null) => void;
}

/**
 * Lets the user choose where to run a job: SLURM cluster or local GPU.
 * Only shown when both options are available (sbatch on PATH + GPUs present).
 * When only one executor is available, this component returns null and
 * the executor is chosen automatically.
 */
export function ExecutorSelector({ value, onChange }: ExecutorSelectorProps): React.JSX.Element | null {
  const { data: sysInfo } = useQuery({
    queryKey: ["system-info"],
    queryFn: getSystemInfo,
    staleTime: 60_000,
  });

  // Only show selector when both options are available
  if (!sysInfo || sysInfo.executor_mode !== "both") {
    return null;
  }

  return (
    <div className="space-y-1">
      <div className="flex items-center gap-1">
        <Label>Run on</Label>
        <TooltipIcon text="Choose where to run this job. SLURM submits to the cluster queue. Local runs directly on this node's GPUs." />
      </div>
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => onChange("slurm")}
          className={`flex-1 rounded-md border px-3 py-2 text-sm font-medium transition-colors ${
            (value ?? "slurm") === "slurm"
              ? "border-blue-500 bg-blue-600/20 text-blue-300"
              : "border-zinc-700 bg-zinc-900 text-zinc-400 hover:border-zinc-600 hover:text-zinc-300"
          }`}
        >
          SLURM Cluster
        </button>
        <button
          type="button"
          onClick={() => onChange("local")}
          className={`flex-1 rounded-md border px-3 py-2 text-sm font-medium transition-colors ${
            value === "local"
              ? "border-emerald-500 bg-emerald-600/20 text-emerald-300"
              : "border-zinc-700 bg-zinc-900 text-zinc-400 hover:border-zinc-600 hover:text-zinc-300"
          }`}
        >
          Local GPU
        </button>
      </div>
    </div>
  );
}
