import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { TooltipIcon } from "../ui/tooltip-icon";
import { tooltips } from "../../lib/tooltips";
import { submitJob } from "../../lib/api/client";

interface PostprocessFormProps {
  projectId: string;
  prefilledInput?: string;
  onSubmitted?: (jobId: string) => void;
}

export function PostprocessForm({
  projectId,
  prefilledInput,
  onSubmitted,
}: PostprocessFormProps): React.JSX.Element {
  const queryClient = useQueryClient();
  const [input, setInput] = useState(prefilledInput ?? "");
  const [bFactor, setBFactor] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Advanced fields
  const [halfmap2, setHalfmap2] = useState("");
  const [voxelSize, setVoxelSize] = useState("");
  const [maskRadius, setMaskRadius] = useState("");
  const [fscMask, setFscMask] = useState("");
  const [applyMask, setApplyMask] = useState("");
  const [batch, setBatch] = useState(false);
  const [estimateBFactor, setEstimateBFactor] = useState(false);
  const [local, setLocal] = useState(false);

  const mutation = useMutation({
    mutationFn: () => {
      const params: Record<string, unknown> = {
        input,
      };
      if (bFactor) params.B_factor = parseFloat(bFactor);
      if (halfmap2) params.halfmap2 = halfmap2;
      if (voxelSize) params.voxel_size = parseFloat(voxelSize);
      if (maskRadius) params.mask_radius = parseFloat(maskRadius);
      if (fscMask) params.fsc_mask = fscMask;
      if (applyMask) params.apply_mask = applyMask;
      if (batch) params.batch = true;
      if (estimateBFactor) params.estimate_B_factor = true;
      if (local) params.local = true;
      return submitJob(projectId, "postprocess", params);
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
          <Label>Input Volume / Halfmap</Label>
          <TooltipIcon text={tooltips["postprocess.input"]} />
        </div>
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="/path/to/halfmap1.mrc or volume directory"
          className="font-mono"
        />
      </div>

      <div className="space-y-1">
        <div className="flex items-center gap-1">
          <Label>B-factor</Label>
          <TooltipIcon text={tooltips["postprocess.B_factor"]} />
        </div>
        <Input
          type="number"
          value={bFactor}
          onChange={(e) => setBFactor(e.target.value)}
          placeholder="None (no sharpening)"
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
              <Label>Second Halfmap</Label>
              <TooltipIcon text={tooltips["postprocess.halfmap2"]} />
            </div>
            <Input
              value={halfmap2}
              onChange={(e) => setHalfmap2(e.target.value)}
              placeholder="Auto-detected"
              className="font-mono"
            />
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Label>Voxel Size</Label>
              <TooltipIcon text={tooltips["postprocess.voxel_size"]} />
            </div>
            <Input
              type="number"
              value={voxelSize}
              onChange={(e) => setVoxelSize(e.target.value)}
              placeholder="From MRC header"
            />
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Label>Mask Radius</Label>
              <TooltipIcon text={tooltips["postprocess.mask_radius"]} />
            </div>
            <Input
              type="number"
              value={maskRadius}
              onChange={(e) => setMaskRadius(e.target.value)}
              placeholder="None"
            />
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Label>FSC Mask</Label>
              <TooltipIcon text={tooltips["postprocess.fsc_mask"]} />
            </div>
            <Input
              value={fscMask}
              onChange={(e) => setFscMask(e.target.value)}
              placeholder="Optional .mrc path"
              className="font-mono"
            />
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Label>Apply Mask</Label>
              <TooltipIcon text={tooltips["postprocess.apply_mask"]} />
            </div>
            <Input
              value={applyMask}
              onChange={(e) => setApplyMask(e.target.value)}
              placeholder="Optional .mrc path"
              className="font-mono"
            />
          </div>

          <div className="flex gap-6">
            <label className="flex items-center gap-2 text-sm text-zinc-400">
              <input
                type="checkbox"
                checked={batch}
                onChange={(e) => setBatch(e.target.checked)}
                className="rounded border-zinc-600 bg-zinc-800"
              />
              Batch processing
              <TooltipIcon text={tooltips["postprocess.batch"]} />
            </label>
            <label className="flex items-center gap-2 text-sm text-zinc-400">
              <input
                type="checkbox"
                checked={estimateBFactor}
                onChange={(e) => setEstimateBFactor(e.target.checked)}
                className="rounded border-zinc-600 bg-zinc-800"
              />
              Estimate B-factor
              <TooltipIcon text={tooltips["postprocess.estimate_B_factor"]} />
            </label>
            <label className="flex items-center gap-2 text-sm text-zinc-400">
              <input
                type="checkbox"
                checked={local}
                onChange={(e) => setLocal(e.target.checked)}
                className="rounded border-zinc-600 bg-zinc-800"
              />
              Local filtering
              <TooltipIcon text={tooltips["postprocess.local"]} />
            </label>
          </div>
        </div>
      )}

      {/* Submit */}
      <div className="flex items-center justify-between pt-2">
        {mutation.isError && (
          <span className="text-sm text-red-400">{(mutation.error as Error).message}</span>
        )}
        <div className="ml-auto">
          <Button
            onClick={() => mutation.mutate()}
            disabled={!input}
            loading={mutation.isPending}
          >
            {mutation.isPending ? "Submitting..." : "Run Postprocess"}
          </Button>
        </div>
      </div>
    </div>
  );
}
