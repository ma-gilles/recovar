import { useState, useCallback } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { PathInput } from "../ui/PathInput";
import { Label } from "../ui/label";
import { Select } from "../ui/select";
import { TooltipIcon } from "../ui/tooltip-icon";
import { FileBrowser } from "../file-browser/FileBrowser";
import { SlurmSettings, type SlurmOpts } from "./SlurmSettings";
import { ExecutorSelector } from "./ExecutorSelector";
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
  const [showResultDirBrowser, setShowResultDirBrowser] = useState(false);
  const [zdim, setZdim] = useState(availableZdims?.[0]?.toString() ?? "4");
  const [nClusters, setNClusters] = useState("40");
  const [nTrajectories, setNTrajectories] = useState("0");
  const [outputName, setOutputName] = useState("");
  const [slurmOpts, setSlurmOpts] = useState<SlurmOpts | null>(null);
  const [executorMode, setExecutorMode] = useState<string | null>(null);
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
    if (outputName) params.output_name = outputName;
    if (slurmOpts) params.slurm_opts = slurmOpts;
    return params;
  }, [resultDir, zdim, nClusters, nTrajectories, outputName, slurmOpts]);

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
      <div className="space-y-1">
        <div className="flex items-center gap-1">
          <Label>Result Directory</Label>
          <TooltipIcon text={tooltips["analyze.result_dir"]} />
        </div>
        <div className="flex gap-2">
          <PathInput
            value={resultDir}
            onChange={setResultDir}
            directoryOnly
            placeholder="/path/to/pipeline/output"
            className="font-mono"
          />
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowResultDirBrowser(!showResultDirBrowser)}
          >
            Browse
          </Button>
        </div>
        {showResultDirBrowser && (
          <FileBrowser
            initialPath={resultDir || "/scratch/gpfs"}
            selectDirectory
            onSelect={(path) => { setResultDir(path); setShowResultDirBrowser(false); }}
          />
        )}
      </div>

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
          placeholder="40"
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

      {/* SLURM Settings */}
      <ExecutorSelector value={executorMode} onChange={setExecutorMode} />
      {executorMode !== "local" && (
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
