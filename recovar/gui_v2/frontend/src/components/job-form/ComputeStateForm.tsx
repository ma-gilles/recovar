import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { TooltipIcon } from "../ui/tooltip-icon";
import { tooltips } from "../../lib/tooltips";
import { submitJob } from "../../lib/api/client";

interface ComputeStateFormProps {
  projectId: string;
  prefilledResultDir?: string;
  prefilledZdim?: number;
  prefilledCoords?: number[];
  onSubmitted?: (jobId: string) => void;
}

export function ComputeStateForm({
  projectId,
  prefilledResultDir,
  prefilledZdim,
  prefilledCoords,
  onSubmitted,
}: ComputeStateFormProps): React.JSX.Element {
  const queryClient = useQueryClient();
  const [resultDir, setResultDir] = useState(prefilledResultDir ?? "");
  const [zdim, setZdim] = useState(prefilledZdim?.toString() ?? "");
  const [coords, setCoords] = useState(prefilledCoords?.join(", ") ?? "");

  const mutation = useMutation({
    mutationFn: () => {
      const latentPoints = coords.split(",").map((s) => parseFloat(s.trim()));
      return submitJob(projectId, "compute_state", {
        result_dir: resultDir,
        zdim: parseInt(zdim),
        latent_points: latentPoints,
      });
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["project", projectId] });
      onSubmitted?.(data.id);
    },
  });

  const coordsValid = coords.length > 0 && coords.split(",").every((s) => !isNaN(parseFloat(s.trim())));

  return (
    <div className="space-y-4">
      <div className="space-y-1">
        <div className="flex items-center gap-1">
          <Label>Result Directory</Label>
          <TooltipIcon text={tooltips["compute_state.result_dir"]} />
        </div>
        <Input
          value={resultDir}
          onChange={(e) => setResultDir(e.target.value)}
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
      </div>

      <div className="flex justify-end pt-2">
        <Button
          onClick={() => mutation.mutate()}
          disabled={!resultDir || !zdim || !coordsValid}
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
