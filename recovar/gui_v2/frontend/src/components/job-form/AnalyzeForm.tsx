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
import { Zap, ChevronDown, ChevronRight } from "lucide-react";

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
  const [nBins, setNBins] = useState("");
  const [maskradFraction, setMaskradFraction] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [slurmOpts, setSlurmOpts] = useState<SlurmOpts | null>(null);
  const [executorMode, setExecutorMode] = useState<string | null>(null);
  const [localOpts, setLocalOpts] = useState<LocalOpts | null>(null);
  const handleSlurmChange = useCallback((opts: SlurmOpts | null) => setSlurmOpts(opts), []);
  const [validating, setValidating] = useState(false);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [validationWarnings, setValidationWarnings] = useState<string[]>([]);

  const buildParams = useCallback((overrides?: Record<string, unknown>): Record<string, unknown> => {
    const params: Record<string, unknown> = {
      result_dir: resultDir,
      zdim: parseInt(zdim),
    };
    if (nClusters) params.n_clusters = parseInt(nClusters);
    if (nTrajectories) params.n_trajectories = parseInt(nTrajectories);
    if (nBins) params.n_bins = parseInt(nBins);
    if (maskradFraction) params.maskrad_fraction = parseFloat(maskradFraction);
    if (noZReg) params.no_z_regularization = true;
    if (outputName) params.output_name = outputName;
    if (slurmOpts && executorMode !== "local") params.slurm_opts = slurmOpts;
    if (localOpts && executorMode === "local") params.local_opts = localOpts;
    return overrides ? { ...params, ...overrides } : params;
  }, [resultDir, zdim, nClusters, nTrajectories, nBins, maskradFraction, noZReg, outputName, slurmOpts, localOpts, executorMode]);

  const mutation = useMutation({
    mutationFn: (params: Record<string, unknown>) => submitJob(projectId, "analyze", params, executorMode),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["project", projectId] });
      onSubmitted?.(data.id);
    },
  });

  const handleSubmit = useCallback(async (overrides?: Record<string, unknown>) => {
    setValidationErrors([]);
    setValidationWarnings([]);
    setValidating(true);
    try {
      const params = buildParams(overrides);
      const result: ValidationResult = await validateJob(projectId, "analyze", params);
      if (!result.valid) {
        setValidationErrors(result.errors);
        setValidationWarnings(result.warnings);
        return;
      }
      setValidationWarnings(result.warnings);
      mutation.mutate(params);
    } catch {
      mutation.mutate(buildParams(overrides));
    } finally {
      setValidating(false);
    }
  }, [projectId, buildParams, mutation]);

  // "Quick Analyze": issue #14 preset — fast, lower-resolution center volumes
  // (n-bins 50->10 ~5x, maskrad-fraction 20->10 ~8x; ~40x overall). UMAP and
  // k-means are unchanged. Reflects the values in the form, then submits.
  const handleQuickSubmit = useCallback(() => {
    setNBins("10");
    setMaskradFraction("10");
    setShowAdvanced(true);
    void handleSubmit({ n_bins: 10, maskrad_fraction: 10 });
  }, [handleSubmit]);

  const missingFields = [
    !resultDir && "Result Directory",
    !zdim && "zdim",
  ].filter(Boolean) as string[];

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

      {/* Advanced: kernel-regression speed/quality knobs (issue #14) */}
      <div>
        <button
          type="button"
          onClick={() => setShowAdvanced((s) => !s)}
          className="flex items-center gap-1 text-sm text-zinc-500 hover:text-zinc-300"
        >
          {showAdvanced ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          Advanced
        </button>
        {showAdvanced && (
          <div className="ml-4 mt-2 grid grid-cols-2 gap-3 border-l border-zinc-800 pl-4">
            <div className="space-y-1">
              <div className="flex items-center gap-1">
                <Label>n-bins</Label>
                <TooltipIcon text={tooltips["analyze.n_bins"]} />
              </div>
              <Input
                type="number"
                value={nBins}
                onChange={(e) => setNBins(e.target.value)}
                placeholder="50"
              />
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-1">
                <Label>maskrad-fraction</Label>
                <TooltipIcon text={tooltips["analyze.maskrad_fraction"]} />
              </div>
              <Input
                type="number"
                value={maskradFraction}
                onChange={(e) => setMaskradFraction(e.target.value)}
                placeholder="20"
              />
            </div>
          </div>
        )}
      </div>

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

      {missingFields.length > 0 && (
        <p className="text-xs text-amber-400">Required to submit: {missingFields.join(", ")}</p>
      )}
      <div className="flex items-center justify-between pt-2">
        {mutation.isError && (
          <span className="text-sm text-red-400">{(mutation.error as Error).message}</span>
        )}
        <div className="ml-auto flex items-center gap-2">
          <Button
            variant="outline"
            onClick={handleQuickSubmit}
            disabled={!resultDir || !zdim || validating || mutation.isPending}
            title="Fast, lower-resolution center volumes: n-bins=10, maskrad-fraction=10 (~40x faster). UMAP and k-means unchanged."
          >
            <Zap className="h-4 w-4" />
            Quick Analyze
          </Button>
          <Button
            onClick={() => handleSubmit()}
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
