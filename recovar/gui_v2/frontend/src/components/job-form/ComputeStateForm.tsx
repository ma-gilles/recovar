import { useState, useCallback } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { Crosshair } from "lucide-react";
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

interface ComputeStateFormProps {
  projectId: string;
  prefilledResultDir?: string;
  prefilledZdim?: number;
  prefilledCoords?: number[];
  /** Job ID for linking to the Explore view to pick coordinates interactively */
  exploreJobId?: string;
  onSubmitted?: (jobId: string) => void;
}

export function ComputeStateForm({
  projectId,
  prefilledResultDir,
  prefilledZdim,
  prefilledCoords,
  exploreJobId,
  onSubmitted,
}: ComputeStateFormProps): React.JSX.Element {
  const queryClient = useQueryClient();
  const [resultDir, setResultDir] = useState(prefilledResultDir ?? "");
  const [zdim, setZdim] = useState(prefilledZdim?.toString() ?? "");
  const [coords, setCoords] = useState(prefilledCoords?.join(", ") ?? "");
  const [slurmOpts, setSlurmOpts] = useState<SlurmOpts | null>(null);
  const [executorMode, setExecutorMode] = useState<string | null>(null);
  const [localOpts, setLocalOpts] = useState<LocalOpts | null>(null);
  const handleSlurmChange = useCallback((opts: SlurmOpts | null) => setSlurmOpts(opts), []);

  const mutation = useMutation({
    mutationFn: () => {
      const latentPoints = coords.split(",").map((s) => parseFloat(s.trim()));
      const params: Record<string, unknown> = {
        result_dir: resultDir,
        zdim: parseInt(zdim),
        latent_points: latentPoints,
      };
      if (slurmOpts) params.slurm_opts = slurmOpts;
      if (localOpts && executorMode === "local") params.local_opts = localOpts;
      return submitJob(projectId, "compute_state", params, executorMode);
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["project", projectId] });
      onSubmitted?.(data.id);
    },
  });

  const coordsValid = coords.length > 0 && coords.split(",").every((s) => !isNaN(parseFloat(s.trim())));

  const zdimNum = parseInt(zdim);
  // Validate the coordinate count for a single point (one comma-separated vector).
  let coordCountError: { expected: number; got: number } | null = null;
  if (coordsValid && !isNaN(zdimNum) && zdimNum > 0) {
    const count = coords.split(",").filter((v) => v.trim().length > 0).length;
    if (count !== zdimNum) {
      coordCountError = { expected: zdimNum, got: count };
    }
  }
  const hasCoordMismatch = coordCountError !== null;

  const missingFields = [
    !resultDir && "Result Directory",
    !zdim && "zdim",
    !coordsValid && "Latent Coordinates",
  ].filter(Boolean) as string[];

  return (
    <div className="space-y-4">
      <PipelineOutputPicker value={resultDir} onChange={setResultDir} tooltip={tooltips["compute_state.result_dir"]} />

      <div className="space-y-1">
        <div className="flex items-center gap-1">
          <Label>zdim (expected coordinate count)</Label>
          <TooltipIcon text={tooltips["compute_state.zdim"]} />
        </div>
        <Input
          type="number"
          value={zdim}
          onChange={(e) => setZdim(e.target.value)}
          placeholder="4"
        />
      </div>

      <div className="space-y-1">
        <div className="flex items-center gap-1">
          <Label>Latent Coordinates</Label>
          <TooltipIcon text={tooltips["compute_state.latent_coords"]} />
        </div>
        <Input
          value={coords}
          onChange={(e) => setCoords(e.target.value)}
          placeholder="0.1, -0.3, 0.5, 0.2"
          className="font-mono"
        />
        {coords.length > 0 && !coordsValid && (
          <p className="text-xs text-red-400">Enter comma-separated numbers</p>
        )}
        {coordCountError && (
          <p className="text-xs text-red-400">
            Expected {coordCountError.expected} coordinates, got {coordCountError.got}
          </p>
        )}
        {exploreJobId && (
          <Link
            to="/explore/$jobId"
            params={{ jobId: exploreJobId }}
            className="inline-flex items-center gap-1.5 text-xs text-blue-400 hover:text-blue-300"
          >
            <Crosshair className="h-3.5 w-3.5" />
            Select point in Explore view
          </Link>
        )}
      </div>

      {/* SLURM Settings */}
      <ExecutorSelector value={executorMode} onChange={setExecutorMode} />
      {executorMode === "local" ? (
        <LocalSettings value={localOpts} onChange={setLocalOpts} />
      ) : (
        <SlurmSettings value={slurmOpts} onChange={handleSlurmChange} />
      )}

      <div className="space-y-2 pt-2">
        {missingFields.length > 0 && (
          <p className="text-xs text-amber-400">Required to submit: {missingFields.join(", ")}</p>
        )}
        <div className="flex justify-end">
          <Button
            onClick={() => mutation.mutate()}
            disabled={!resultDir || !zdim || !coordsValid || hasCoordMismatch}
            loading={mutation.isPending}
          >
            {mutation.isPending ? "Submitting..." : "Compute State"}
          </Button>
        </div>
      </div>
      {mutation.isError && (
        <p className="text-sm text-red-400">{(mutation.error as Error).message}</p>
      )}
    </div>
  );
}
