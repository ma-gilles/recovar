import { useState, useCallback } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { Crosshair } from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { PathInput } from "../ui/PathInput";
import { Label } from "../ui/label";
import { TooltipIcon } from "../ui/tooltip-icon";
import { SlurmSettings, type SlurmOpts } from "./SlurmSettings";
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
      return submitJob(projectId, "compute_state", params);
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["project", projectId] });
      onSubmitted?.(data.id);
    },
  });

  const coordsValid = coords.length > 0 && coords.split(",").every((s) => !isNaN(parseFloat(s.trim())));

  const zdimNum = parseInt(zdim);
  // Validate coordinate count per line: single-line comma-separated, or multi-line with one point per line
  const coordLines = coords.trim().split(/\n/).filter((line) => line.trim().length > 0);
  const coordCountErrors: Array<{ line: number; expected: number; got: number }> = [];
  if (coordsValid && !isNaN(zdimNum) && zdimNum > 0) {
    if (coordLines.length > 1) {
      // Multi-line: validate each line independently
      coordLines.forEach((line, idx) => {
        const count = line.split(",").filter((v) => v.trim().length > 0).length;
        if (count !== zdimNum) {
          coordCountErrors.push({ line: idx + 1, expected: zdimNum, got: count });
        }
      });
    } else {
      // Single line: validate total count
      const count = coords.split(",").filter((v) => v.trim().length > 0).length;
      if (count !== zdimNum) {
        coordCountErrors.push({ line: 1, expected: zdimNum, got: count });
      }
    }
  }
  const hasCoordMismatch = coordCountErrors.length > 0;

  return (
    <div className="space-y-4">
      <div className="space-y-1">
        <div className="flex items-center gap-1">
          <Label>Result Directory</Label>
          <TooltipIcon text={tooltips["compute_state.result_dir"]} />
        </div>
        <PathInput
          value={resultDir}
          onChange={setResultDir}
          directoryOnly
          placeholder="/path/to/pipeline/output"
          className="font-mono"
        />
      </div>

      <div className="space-y-1">
        <div className="flex items-center gap-1">
          <Label>zdim</Label>
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
        {hasCoordMismatch && coordCountErrors.map((err) => (
          <p key={err.line} className="text-xs text-red-400">
            {coordLines.length > 1
              ? `Line ${err.line}: expected ${err.expected} coordinates, got ${err.got}`
              : `Expected ${err.expected} coordinates, got ${err.got}`}
          </p>
        ))}
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
      <SlurmSettings value={slurmOpts} onChange={handleSlurmChange} />

      <div className="flex justify-end pt-2">
        <Button
          onClick={() => mutation.mutate()}
          disabled={!resultDir || !zdim || !coordsValid || hasCoordMismatch}
          loading={mutation.isPending}
        >
          {mutation.isPending ? "Submitting..." : "Compute State"}
        </Button>
      </div>
      {mutation.isError && (
        <p className="text-sm text-red-400">{(mutation.error as Error).message}</p>
      )}
    </div>
  );
}
