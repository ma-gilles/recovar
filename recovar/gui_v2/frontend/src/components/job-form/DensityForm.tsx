import { useState, useCallback } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { TooltipIcon } from "../ui/tooltip-icon";
import { PipelineOutputPicker } from "./PipelineOutputPicker";
import { SlurmSettings, type SlurmOpts } from "./SlurmSettings";
import { ExecutorSelector } from "./ExecutorSelector";
import { LocalSettings, type LocalOpts } from "./LocalSettings";
import { tooltips } from "../../lib/tooltips";
import { submitJob } from "../../lib/api/client";

interface DensityFormProps {
  projectId: string;
  prefilledResultDir?: string;
  onSubmitted?: (jobId: string) => void;
}

export function DensityForm({
  projectId,
  prefilledResultDir,
  onSubmitted,
}: DensityFormProps): React.JSX.Element {
  const queryClient = useQueryClient();
  const [resultDir, setResultDir] = useState(prefilledResultDir ?? "");
  const [pcaDim, setPcaDim] = useState("4");
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Advanced fields
  const [zDimUsed, setZDimUsed] = useState("");
  const [percentileReject, setPercentileReject] = useState("10");
  const [numDiscPoints, setNumDiscPoints] = useState("");
  const [percentileBound, setPercentileBound] = useState("1");
  const [slurmOpts, setSlurmOpts] = useState<SlurmOpts | null>(null);
  const [executorMode, setExecutorMode] = useState<string | null>(null);
  const [localOpts, setLocalOpts] = useState<LocalOpts | null>(null);
  const handleSlurmChange = useCallback((opts: SlurmOpts | null) => setSlurmOpts(opts), []);

  const mutation = useMutation({
    mutationFn: () => {
      const params: Record<string, unknown> = {
        result_dir: resultDir,
      };
      if (pcaDim) params.pca_dim = parseInt(pcaDim);
      if (zDimUsed) params.z_dim_used = parseInt(zDimUsed);
      if (percentileReject) params.percentile_reject = parseInt(percentileReject);
      if (numDiscPoints) params.num_disc_points = parseInt(numDiscPoints);
      if (percentileBound) params.percentile_bound = parseInt(percentileBound);
      if (slurmOpts) params.slurm_opts = slurmOpts;
      if (localOpts && executorMode === "local") params.local_opts = localOpts;
      return submitJob(projectId, "density", params, executorMode);
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["project", projectId] });
      onSubmitted?.(data.id);
    },
  });

  const missingFields = [!resultDir && "Result Directory"].filter(Boolean) as string[];

  return (
    <div className="space-y-4">
      <PipelineOutputPicker value={resultDir} onChange={setResultDir} tooltip={tooltips["density.result_dir"]} />

      <div className="space-y-1">
        <div className="flex items-center gap-1">
          <Label>PCA Dimension</Label>
          <TooltipIcon text={tooltips["density.pca_dim"]} />
        </div>
        <Input
          type="number"
          value={pcaDim}
          onChange={(e) => setPcaDim(e.target.value)}
          placeholder="4"
        />
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
              <Label>Z Dimension Used</Label>
              <TooltipIcon text={tooltips["density.z_dim_used"]} />
            </div>
            <Input
              type="number"
              value={zDimUsed}
              onChange={(e) => setZDimUsed(e.target.value)}
              placeholder="Auto (smallest >= pca_dim)"
            />
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Label>Percentile Reject</Label>
              <TooltipIcon text={tooltips["density.percentile_reject"]} />
            </div>
            <Input
              type="number"
              value={percentileReject}
              onChange={(e) => setPercentileReject(e.target.value)}
              placeholder="10"
            />
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Label>Grid Points</Label>
              <TooltipIcon text={tooltips["density.num_disc_points"]} />
            </div>
            <Input
              type="number"
              value={numDiscPoints}
              onChange={(e) => setNumDiscPoints(e.target.value)}
              placeholder="Auto"
            />
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Label>Percentile Bound</Label>
              <TooltipIcon text={tooltips["density.percentile_bound"]} />
            </div>
            <Input
              type="number"
              value={percentileBound}
              onChange={(e) => setPercentileBound(e.target.value)}
              placeholder="1"
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
      {missingFields.length > 0 && (
        <p className="text-xs text-amber-400">Required to submit: {missingFields.join(", ")}</p>
      )}
      <div className="flex items-center justify-between pt-2">
        {mutation.isError && (
          <span className="text-sm text-red-400">{(mutation.error as Error).message}</span>
        )}
        <div className="ml-auto">
          <Button
            onClick={() => mutation.mutate()}
            disabled={!resultDir}
            loading={mutation.isPending}
          >
            {mutation.isPending ? "Submitting..." : "Estimate Density"}
          </Button>
        </div>
      </div>
    </div>
  );
}
