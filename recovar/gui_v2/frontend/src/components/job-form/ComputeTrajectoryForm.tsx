import { useState, useCallback } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { PathInput } from "../ui/PathInput";
import { Label } from "../ui/label";
import { TooltipIcon } from "../ui/tooltip-icon";
import { SlurmSettings, type SlurmOpts } from "./SlurmSettings";
import { tooltips } from "../../lib/tooltips";
import { submitJob } from "../../lib/api/client";

interface ComputeTrajectoryFormProps {
  projectId: string;
  prefilledResultDir?: string;
  prefilledZdim?: number;
  prefilledStart?: number[];
  prefilledEnd?: number[];
  onSubmitted?: (jobId: string) => void;
}

export function ComputeTrajectoryForm({
  projectId,
  prefilledResultDir,
  prefilledZdim,
  prefilledStart,
  prefilledEnd,
  onSubmitted,
}: ComputeTrajectoryFormProps): React.JSX.Element {
  const queryClient = useQueryClient();
  const [resultDir, setResultDir] = useState(prefilledResultDir ?? "");
  const [zdim, setZdim] = useState(prefilledZdim?.toString() ?? "");
  const [zStart, setZStart] = useState(prefilledStart?.join(", ") ?? "");
  const [zEnd, setZEnd] = useState(prefilledEnd?.join(", ") ?? "");
  const [nVols, setNVols] = useState("6");
  const [slurmOpts, setSlurmOpts] = useState<SlurmOpts | null>(null);
  const handleSlurmChange = useCallback((opts: SlurmOpts | null) => setSlurmOpts(opts), []);

  const parseCoords = (s: string) => s.split(",").map((v) => parseFloat(v.trim()));
  const isValidCoords = (s: string) =>
    s.length > 0 && s.split(",").every((v) => !isNaN(parseFloat(v.trim())));

  const countCoords = (s: string): number =>
    s.split(",").filter((v) => v.trim().length > 0).length;
  const zdimNum = parseInt(zdim);
  const zStartCount = zStart.length > 0 ? countCoords(zStart) : 0;
  const zEndCount = zEnd.length > 0 ? countCoords(zEnd) : 0;
  const zStartMismatch = isValidCoords(zStart) && !isNaN(zdimNum) && zdimNum > 0 && zStartCount !== zdimNum;
  const zEndMismatch = isValidCoords(zEnd) && !isNaN(zdimNum) && zdimNum > 0 && zEndCount !== zdimNum;

  const mutation = useMutation({
    mutationFn: () => {
      const params: Record<string, unknown> = {
        result_dir: resultDir,
        zdim: parseInt(zdim),
        z_start: parseCoords(zStart),
        z_end: parseCoords(zEnd),
        n_vols_along_path: parseInt(nVols),
      };
      if (slurmOpts) params.slurm_opts = slurmOpts;
      return submitJob(projectId, "compute_trajectory", params);
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["project", projectId] });
      onSubmitted?.(data.id);
    },
  });

  const canSubmit =
    resultDir.length > 0 &&
    zdim.length > 0 &&
    isValidCoords(zStart) &&
    isValidCoords(zEnd) &&
    !zStartMismatch &&
    !zEndMismatch;

  return (
    <div className="space-y-4">
      <div className="space-y-1">
        <div className="flex items-center gap-1">
          <Label>Result Directory</Label>
          <TooltipIcon text={tooltips["compute_trajectory.result_dir"]} />
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
          <TooltipIcon text={tooltips["compute_trajectory.zdim"]} />
        </div>
        <Input type="number" value={zdim} onChange={(e) => setZdim(e.target.value)} placeholder="4" />
      </div>

      <div className="space-y-1">
        <div className="flex items-center gap-1">
          <Label>Start Coordinates</Label>
          <TooltipIcon text={tooltips["compute_trajectory.z_start"]} />
        </div>
        <Input
          value={zStart}
          onChange={(e) => setZStart(e.target.value)}
          placeholder="0.1, -0.3, 0.5, 0.2"
          className="font-mono"
        />
        {zStart.length > 0 && !isValidCoords(zStart) && (
          <p className="text-xs text-red-400">Enter comma-separated numbers</p>
        )}
        {zStartMismatch && (
          <p className="text-xs text-red-400">Expected {zdimNum} coordinates, got {zStartCount}</p>
        )}
      </div>

      <div className="space-y-1">
        <div className="flex items-center gap-1">
          <Label>End Coordinates</Label>
          <TooltipIcon text={tooltips["compute_trajectory.z_end"]} />
        </div>
        <Input
          value={zEnd}
          onChange={(e) => setZEnd(e.target.value)}
          placeholder="-0.2, 0.4, -0.1, 0.3"
          className="font-mono"
        />
        {zEnd.length > 0 && !isValidCoords(zEnd) && (
          <p className="text-xs text-red-400">Enter comma-separated numbers</p>
        )}
        {zEndMismatch && (
          <p className="text-xs text-red-400">Expected {zdimNum} coordinates, got {zEndCount}</p>
        )}
      </div>

      <div className="space-y-1">
        <div className="flex items-center gap-1">
          <Label>Volumes Along Path</Label>
          <TooltipIcon text={tooltips["compute_trajectory.n_vols"]} />
        </div>
        <Input type="number" value={nVols} onChange={(e) => setNVols(e.target.value)} placeholder="6" />
      </div>

      {/* SLURM Settings */}
      <SlurmSettings value={slurmOpts} onChange={handleSlurmChange} />

      <div className="flex justify-end pt-2">
        <Button onClick={() => mutation.mutate()} disabled={!canSubmit} loading={mutation.isPending}>
          {mutation.isPending ? "Submitting..." : "Compute Trajectory"}
        </Button>
      </div>
      {mutation.isError && (
        <p className="text-sm text-red-400">{(mutation.error as Error).message}</p>
      )}
    </div>
  );
}
