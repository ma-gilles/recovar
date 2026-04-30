import { useState, useCallback } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { PathInput } from "../ui/PathInput";
import { Label } from "../ui/label";
import { TooltipIcon } from "../ui/tooltip-icon";
import { FileBrowser } from "../file-browser/FileBrowser";
import { SlurmSettings, type SlurmOpts } from "./SlurmSettings";
import { ExecutorSelector } from "./ExecutorSelector";
import { LocalSettings, type LocalOpts } from "./LocalSettings";
import { tooltips } from "../../lib/tooltips";
import { submitJob } from "../../lib/api/client";

interface StableStatesFormProps {
  projectId: string;
  prefilledDensity?: string;
  onSubmitted?: (jobId: string) => void;
}

export function StableStatesForm({
  projectId,
  prefilledDensity,
  onSubmitted,
}: StableStatesFormProps): React.JSX.Element {
  const queryClient = useQueryClient();
  const [density, setDensity] = useState(prefilledDensity ?? "");
  const [showDensityBrowser, setShowDensityBrowser] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Advanced fields
  const [percentTop, setPercentTop] = useState("1");
  const [nLocalMaxs, setNLocalMaxs] = useState("3");
  const [slurmOpts, setSlurmOpts] = useState<SlurmOpts | null>(null);
  const [executorMode, setExecutorMode] = useState<string | null>(null);
  const [localOpts, setLocalOpts] = useState<LocalOpts | null>(null);
  const handleSlurmChange = useCallback((opts: SlurmOpts | null) => setSlurmOpts(opts), []);

  const mutation = useMutation({
    mutationFn: () => {
      const params: Record<string, unknown> = {
        density,
      };
      if (percentTop) params.percent_top = parseFloat(percentTop);
      if (nLocalMaxs) params.n_local_maxs = parseInt(nLocalMaxs);
      if (slurmOpts) params.slurm_opts = slurmOpts;
      if (localOpts && executorMode === "local") params.local_opts = localOpts;
      return submitJob(projectId, "stable_states", params, executorMode);
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["project", projectId] });
      onSubmitted?.(data.id);
    },
  });

  return (
    <div className="space-y-4">
      <div className="space-y-1">
        <div className="flex items-center gap-1">
          <Label>Density File</Label>
          <TooltipIcon text={tooltips["stable_states.density"]} />
        </div>
        <div className="flex gap-2">
          <PathInput
            value={density}
            onChange={setDensity}
            accept={[".pkl"]}
            placeholder="/path/to/deconv_density_knee.pkl"
            className="font-mono"
          />
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowDensityBrowser(!showDensityBrowser)}
          >
            Browse
          </Button>
        </div>
        {showDensityBrowser && (
          <FileBrowser
            initialPath={density ? density.split("/").slice(0, -1).join("/") || "/" : "/scratch/gpfs"}
            accept={[".pkl"]}
            onSelect={(path) => { setDensity(path); setShowDensityBrowser(false); }}
          />
        )}
      </div>

      {/* Advanced Section */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="flex items-center gap-1 text-sm text-zinc-500 hover:text-zinc-300"
      >
        {showAdvanced ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        Advanced
      </button>

      {showAdvanced && (
        <div className="ml-4 space-y-3 border-l border-zinc-800 pl-4">
          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Label>Percent Top</Label>
              <TooltipIcon text={tooltips["stable_states.percent_top"]} />
            </div>
            <Input
              type="number"
              value={percentTop}
              onChange={(e) => setPercentTop(e.target.value)}
              placeholder="1"
            />
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Label>Number of Local Maxima</Label>
              <TooltipIcon text={tooltips["stable_states.n_local_maxs"]} />
            </div>
            <Input
              type="number"
              value={nLocalMaxs}
              onChange={(e) => setNLocalMaxs(e.target.value)}
              placeholder="3"
            />
          </div>
        </div>
      )}

      {/* SLURM Settings */}
      <ExecutorSelector value={executorMode} onChange={setExecutorMode} />
      {executorMode === "local" ? (
        <LocalSettings value={localOpts} onChange={setLocalOpts} />
      ) : (
        <SlurmSettings value={slurmOpts} onChange={handleSlurmChange} />
      )}

      {/* Submit */}
      <div className="flex items-center justify-between pt-2">
        {mutation.isError && (
          <span className="text-sm text-red-400">{(mutation.error as Error).message}</span>
        )}
        <div className="ml-auto">
          <Button
            onClick={() => mutation.mutate()}
            disabled={!density}
            loading={mutation.isPending}
          >
            {mutation.isPending ? "Submitting..." : "Find Stable States"}
          </Button>
        </div>
      </div>
    </div>
  );
}
