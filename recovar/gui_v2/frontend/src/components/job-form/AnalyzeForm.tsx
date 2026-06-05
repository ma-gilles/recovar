import { useState, useCallback } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Select } from "../ui/select";
import { TooltipIcon } from "../ui/tooltip-icon";
import { PipelineOutputPicker } from "./PipelineOutputPicker";
import { SlurmSettings, type SlurmOpts } from "./SlurmSettings";
import { ExecutorSelector } from "./ExecutorSelector";
import { LocalSettings, type LocalOpts } from "./LocalSettings";
import { tooltips } from "../../lib/tooltips";
import { submitJob, validateJob, type ValidationResult } from "../../lib/api/client";

interface AnalyzeFormProps {
  projectId: string;
  prefilledResultDir?: string;
  availableZdims?: number[];
  onSubmitted?: (jobId: string) => void;
}

export function AnalyzeForm({
  projectId,
  prefilledResultDir,
  availableZdims,
  onSubmitted,
}: AnalyzeFormProps): React.JSX.Element {
  const queryClient = useQueryClient();
  const [resultDir, setResultDir] = useState(prefilledResultDir ?? "");
  const [zdim, setZdim] = useState(availableZdims?.[0]?.toString() ?? "4");
  const [noZReg, setNoZReg] = useState(false);
  const [nClusters, setNClusters] = useState("20");
  const [nTrajectories, setNTrajectories] = useState("0");
  const [outputName, setOutputName] = useState("");
  const [slurmOpts, setSlurmOpts] = useState<SlurmOpts | null>(null);
  const [executorMode, setExecutorMode] = useState<string | null>(null);
  const [localOpts, setLocalOpts] = useState<LocalOpts | null>(null);
  const handleSlurmChange = useCallback((opts: SlurmOpts | null) => setSlurmOpts(opts), []);
  const [validating, setValidating] = useState(false);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [validationWarnings, setValidationWarnings] = useState<string[]>([]);

  const buildParams = useCallback((): Record<string, unknown> => {
    const params: Record<string, unknown> = {
      result_dir: resultDir,
      zdim: parseInt(zdim),
    };
    if (nClusters) params.n_clusters = parseInt(nClusters);
    if (nTrajectories) params.n_trajectories = parseInt(nTrajectories);
    if (noZReg) params.no_z_regularization = true;
    if (outputName) params.output_name = outputName;
    if (slurmOpts) params.slurm_opts = slurmOpts;
      if (localOpts && executorMode === "local") params.local_opts = localOpts;
    return params;
  }, [resultDir, zdim, nClusters, nTrajectories, noZReg, outputName, slurmOpts, localOpts, executorMode]);

  const mutation = useMutation({
    mutationFn: () => submitJob(projectId, "analyze", buildParams(), executorMode),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["project", projectId] });
      onSubmitted?.(data.id);
    },
  });

  const handleSubmit = useCallback(async () => {
    setValidationErrors([]);
    setValidationWarnings([]);
    setValidating(true);
    try {
      const result: ValidationResult = await validateJob(projectId, "analyze", buildParams());
      if (!result.valid) {
        setValidationErrors(result.errors);
        setValidationWarnings(result.warnings);
        return;
      }
      setValidationWarnings(result.warnings);
      mutation.mutate();
    } catch {
      mutation.mutate();
    } finally {
      setValidating(false);
    }
  }, [projectId, buildParams, mutation]);

  return (
    <div className="space-y-4">
      <PipelineOutputPicker value={resultDir} onChange={setResultDir} tooltip={tooltips["analyze.result_dir"]} />

      <div className="space-y-1">
        <div className="flex items-center gap-1">
          <Label>zdim</Label>
          <TooltipIcon text={tooltips["analyze.zdim"]} />
        </div>
        {availableZdims && availableZdims.length > 0 ? (
          <Select value={zdim} onChange={(e) => setZdim(e.target.value)}>
            {availableZdims.map((z) => (
              <option key={z} value={z}>
                {z}
              </option>
            ))}
          </Select>
        ) : (
          <Input
            type="number"
            value={zdim}
            onChange={(e) => setZdim(e.target.value)}
            placeholder="4"
          />
        )}
      </div>

      <div className="space-y-1">
        <div className="flex items-center gap-1">
          <Label>K-means Clusters</Label>
          <TooltipIcon text={tooltips["analyze.n_clusters"]} />
        </div>
        <Input
          type="number"
          value={nClusters}
          onChange={(e) => setNClusters(e.target.value)}
          placeholder="20"
        />
      </div>

      <div className="space-y-1">
        <div className="flex items-center gap-1">
          <Label>Trajectories</Label>
          <TooltipIcon text={tooltips["analyze.n_trajectories"]} />
        </div>
        <Input
          type="number"
          value={nTrajectories}
          onChange={(e) => setNTrajectories(e.target.value)}
          placeholder="0"
        />
      </div>

      <div className="space-y-1">
        <Label>Output Name</Label>
        <Input value={outputName} onChange={(e) => setOutputName(e.target.value)} placeholder="Auto-generated" />
      </div>

      <label className="flex items-center gap-2 text-sm text-zinc-400">
        <input
          type="checkbox"
          checked={noZReg}
          onChange={(e) => setNoZReg(e.target.checked)}
          className="rounded border-zinc-600 bg-zinc-800"
        />
        No z-regularization
        <TooltipIcon text={tooltips["analyze.no_z_regularization"]} />
      </label>

      {/* SLURM Settings */}
      <ExecutorSelector value={executorMode} onChange={setExecutorMode} />
      {executorMode === "local" ? (
        <LocalSettings value={localOpts} onChange={setLocalOpts} />
      ) : (
        <SlurmSettings value={slurmOpts} onChange={handleSlurmChange} />
      )}

      {/* Validation feedback */}
      {validationErrors.length > 0 && (
        <div className="space-y-1 rounded border border-red-800 bg-red-950/30 p-3">
          {validationErrors.map((err, i) => (
            <div key={i} className="text-sm text-red-400">{err}</div>
          ))}
        </div>
      )}
      {validationWarnings.length > 0 && (
        <div className="space-y-1 rounded border border-amber-800 bg-amber-950/30 p-3">
          {validationWarnings.map((warn, i) => (
            <div key={i} className="text-sm text-amber-400">{warn}</div>
          ))}
        </div>
      )}

      <div className="flex items-center justify-between pt-2">
        {mutation.isError && (
          <span className="text-sm text-red-400">{(mutation.error as Error).message}</span>
        )}
        <div className="ml-auto">
          <Button
            onClick={handleSubmit}
            disabled={!resultDir || !zdim}
            loading={validating || mutation.isPending}
          >
            {validating ? "Validating inputs..." : mutation.isPending ? "Submitting..." : "Submit Analyze Job"}
          </Button>
        </div>
      </div>
    </div>
  );
}
