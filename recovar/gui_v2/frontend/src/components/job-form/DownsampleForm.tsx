import { useState, useCallback } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { TooltipIcon } from "../ui/tooltip-icon";
import { FileBrowser } from "../file-browser/FileBrowser";
import { SlurmSettings, type SlurmOpts } from "./SlurmSettings";
import { tooltips } from "../../lib/tooltips";
import { submitJob } from "../../lib/api/client";

interface DownsampleFormProps {
  projectId: string;
  prefilledParticles?: string;
  onSubmitted?: (jobId: string) => void;
}

export function DownsampleForm({
  projectId,
  prefilledParticles,
  onSubmitted,
}: DownsampleFormProps): React.JSX.Element {
  const queryClient = useQueryClient();
  const [particles, setParticles] = useState(prefilledParticles ?? "");
  const [showParticlesBrowser, setShowParticlesBrowser] = useState(false);
  const [targetD, setTargetD] = useState("128");
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Advanced fields
  const [datadir, setDatadir] = useState("");
  const [stripPrefix, setStripPrefix] = useState("");
  const [batchSize, setBatchSize] = useState("1000");
  const [slurmOpts, setSlurmOpts] = useState<SlurmOpts | null>(null);
  const handleSlurmChange = useCallback((opts: SlurmOpts | null) => setSlurmOpts(opts), []);

  const targetDValid = targetD.length > 0 && parseInt(targetD) > 0 && parseInt(targetD) % 2 === 0;

  const mutation = useMutation({
    mutationFn: () => {
      const params: Record<string, unknown> = {
        particles,
        target_D: parseInt(targetD),
      };
      if (datadir) params.datadir = datadir;
      if (stripPrefix) params.strip_prefix = stripPrefix;
      if (batchSize) params.batch_size = parseInt(batchSize);
      if (slurmOpts) params.slurm_opts = slurmOpts;
      return submitJob(projectId, "downsample", params);
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["project", projectId] });
      onSubmitted?.(data.id);
    },
  });

  const canSubmit = particles.length > 0 && targetDValid;

  return (
    <div className="space-y-4">
      <div className="space-y-1">
        <div className="flex items-center gap-1">
          <Label>Particles</Label>
          <TooltipIcon text={tooltips["downsample.particles"]} />
        </div>
        <div className="flex gap-2">
          <Input
            value={particles}
            onChange={(e) => setParticles(e.target.value)}
            placeholder="/path/to/particles.star"
            className="font-mono"
          />
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowParticlesBrowser(!showParticlesBrowser)}
          >
            Browse
          </Button>
        </div>
        {showParticlesBrowser && (
          <FileBrowser
            initialPath={particles ? particles.split("/").slice(0, -1).join("/") || "/" : "/scratch/gpfs"}
            accept={[".star", ".cs", ".mrcs", ".txt"]}
            onSelect={(path) => { setParticles(path); setShowParticlesBrowser(false); }}
          />
        )}
      </div>

      <div className="space-y-1">
        <div className="flex items-center gap-1">
          <Label>Target Box Size</Label>
          <TooltipIcon text={tooltips["downsample.target_D"]} />
        </div>
        <Input
          type="number"
          value={targetD}
          onChange={(e) => setTargetD(e.target.value)}
          placeholder="128"
        />
        {targetD.length > 0 && !targetDValid && (
          <p className="text-xs text-red-400">Must be a positive even number</p>
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
              <Label>Data Directory</Label>
              <TooltipIcon text={tooltips["downsample.datadir"]} />
            </div>
            <Input
              value={datadir}
              onChange={(e) => setDatadir(e.target.value)}
              placeholder="Override data dir"
              className="font-mono"
            />
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Label>Strip Prefix</Label>
              <TooltipIcon text={tooltips["downsample.strip_prefix"]} />
            </div>
            <Input
              value={stripPrefix}
              onChange={(e) => setStripPrefix(e.target.value)}
              placeholder="Prefix to strip"
            />
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Label>Batch Size</Label>
              <TooltipIcon text={tooltips["downsample.batch_size"]} />
            </div>
            <Input
              type="number"
              value={batchSize}
              onChange={(e) => setBatchSize(e.target.value)}
              placeholder="1000"
            />
          </div>
        </div>
      )}

      {/* SLURM Settings */}
      <SlurmSettings value={slurmOpts} onChange={handleSlurmChange} />

      {/* Submit */}
      <div className="flex items-center justify-between pt-2">
        {mutation.isError && (
          <span className="text-sm text-red-400">{(mutation.error as Error).message}</span>
        )}
        <div className="ml-auto">
          <Button
            onClick={() => mutation.mutate()}
            disabled={!canSubmit}
            loading={mutation.isPending}
          >
            {mutation.isPending ? "Submitting..." : "Downsample"}
          </Button>
        </div>
      </div>
    </div>
  );
}
