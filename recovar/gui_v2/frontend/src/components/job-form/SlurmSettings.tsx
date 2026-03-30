import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { TooltipIcon } from "../ui/tooltip-icon";
import { tooltips } from "../../lib/tooltips";
import { getSlurmDefaults, getSystemInfo, type SlurmDefaults } from "../../lib/api/client";

export interface SlurmOpts {
  partition: string;
  account: string;
  gpus: number;
  cpus: number;
  memory: string;
  time: string;
}

interface SlurmSettingsProps {
  value: SlurmOpts | null;
  onChange: (opts: SlurmOpts | null) => void;
}

/**
 * Collapsible SLURM settings section for job submission forms.
 *
 * Fetches defaults from the server and pre-fills editable fields.
 * Only shown when the backend reports SLURM is available.
 */
export function SlurmSettings({ value, onChange }: SlurmSettingsProps): React.JSX.Element | null {
  const [expanded, setExpanded] = useState(false);

  const { data: sysInfo } = useQuery({
    queryKey: ["system-info"],
    queryFn: getSystemInfo,
    staleTime: 60_000,
  });

  const { data: defaults } = useQuery<SlurmDefaults>({
    queryKey: ["slurm-defaults"],
    queryFn: getSlurmDefaults,
    staleTime: 60_000,
    enabled: sysInfo?.slurm_available === true,
  });

  // Initialize value from defaults once loaded
  useEffect(() => {
    if (defaults && value === null) {
      onChange({
        partition: defaults.partition,
        account: defaults.account,
        gpus: defaults.gpus,
        cpus: defaults.cpus,
        memory: defaults.memory,
        time: defaults.time,
      });
    }
  }, [defaults, value, onChange]);

  // Don't render if SLURM is not available
  if (!sysInfo?.slurm_available) {
    return null;
  }

  const current = value ?? {
    partition: defaults?.partition ?? "cryoem",
    account: defaults?.account ?? "amits",
    gpus: defaults?.gpus ?? 1,
    cpus: defaults?.cpus ?? 4,
    memory: defaults?.memory ?? "300G",
    time: defaults?.time ?? "12:00:00",
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
        </div>
      )}
    </div>
  );
}
