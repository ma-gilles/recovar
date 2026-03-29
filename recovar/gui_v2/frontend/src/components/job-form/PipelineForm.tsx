import { useState, useCallback } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { PathInput } from "../ui/PathInput";
import { Label } from "../ui/label";
import { Select } from "../ui/select";
import { TooltipIcon } from "../ui/tooltip-icon";
import { FileBrowser } from "../file-browser/FileBrowser";
import { SlurmSettings, type SlurmOpts } from "./SlurmSettings";
import { tooltips } from "../../lib/tooltips";
import { submitJob } from "../../lib/api/client";

interface PipelineFormProps {
  projectId: string;
  projectPath: string;
  onSubmitted?: (jobId: string) => void;
}

export function PipelineForm({ projectId, projectPath, onSubmitted }: PipelineFormProps): React.JSX.Element {
  const queryClient = useQueryClient();
  const [particles, setParticles] = useState("");
  const [mask, setMask] = useState("from_halfmaps");
  const [maskPath, setMaskPath] = useState("");
  const [showParticleBrowser, setShowParticleBrowser] = useState(false);
  const [showMaskBrowser, setShowMaskBrowser] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [validationInfo, setValidationInfo] = useState<{
    n_particles?: number;
    box_size?: number;
    error?: string;
  } | null>(null);

  // Advanced fields
  const [zdim, setZdim] = useState("1,2,4,10,20");
  const [downsample, setDownsample] = useState("256");
  const [lazy, setLazy] = useState(false);
  const [correctContrast, setCorrectContrast] = useState(false);
  const [focusMask, setFocusMask] = useState("");
  const [datadir, setDatadir] = useState("");
  const [nImages, setNImages] = useState("");
  const [halfsets, setHalfsets] = useState("");
  const [poses, setPoses] = useState("");
  const [ctf, setCtf] = useState("");
  const [tiltSeries, setTiltSeries] = useState(false);
  const [stripPrefix, setStripPrefix] = useState("");
  const [outputName, setOutputName] = useState("");
  const [slurmOpts, setSlurmOpts] = useState<SlurmOpts | null>(null);
  const handleSlurmChange = useCallback((opts: SlurmOpts | null) => setSlurmOpts(opts), []);

  const mutation = useMutation({
    mutationFn: () => {
      const params: Record<string, unknown> = {
        particles,
        mask: mask === "file" ? maskPath : mask,
      };
      if (zdim) params.zdim = zdim;
      if (downsample) params.downsample = parseInt(downsample);
      if (lazy) params.lazy = true;
      if (correctContrast) params.correct_contrast = true;
      if (focusMask) params.focus_mask = focusMask;
      if (datadir) params.datadir = datadir;
      if (nImages) params.n_images = parseInt(nImages);
      if (halfsets) params.halfsets = halfsets;
      if (poses) params.poses = poses;
      if (ctf) params.ctf = ctf;
      if (tiltSeries) params.tilt_series = true;
      if (stripPrefix) params.strip_prefix = stripPrefix;
      if (outputName) params.output_name = outputName;
      if (slurmOpts) params.slurm_opts = slurmOpts;
      return submitJob(projectId, "pipeline", params);
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["project", projectId] });
      onSubmitted?.(data.id);
    },
  });

  const canSubmit = particles.length > 0 && (mask !== "file" || maskPath.length > 0);

  return (
    <div className="space-y-4">
      {/* Particles */}
      <div className="space-y-1">
        <div className="flex items-center gap-1">
          <Label>Particles</Label>
          <TooltipIcon text={tooltips["pipeline.particles"]} />
        </div>
        <div className="flex gap-2">
          <PathInput
            value={particles}
            onChange={setParticles}
            accept={[".star", ".cs", ".mrcs", ".txt"]}
            placeholder="/path/to/particles.star"
            className="font-mono"
          />
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowParticleBrowser(!showParticleBrowser)}
          >
            Browse
          </Button>
        </div>
        {showParticleBrowser && (
          <FileBrowser
            initialPath={projectPath}
            accept={[".star", ".cs", ".mrcs", ".txt"]}
            onSelect={(path) => {
              setParticles(path);
              setShowParticleBrowser(false);
            }}
            onValidation={(result) => setValidationInfo(result)}
          />
        )}
        {validationInfo?.n_particles && (
          <div className="text-xs text-zinc-400">
            {validationInfo.n_particles.toLocaleString()} particles, box size {validationInfo.box_size}
          </div>
        )}
        {validationInfo?.error && (
          <div className="text-xs text-red-400">{validationInfo.error}</div>
        )}
        {validationInfo?.n_particles && validationInfo.n_particles < 10000 && (
          <div className="text-xs text-amber-400">
            Only {validationInfo.n_particles.toLocaleString()} particles. Results may be unreliable below ~10K.
          </div>
        )}
      </div>

      {/* Mask */}
      <div className="space-y-1">
        <div className="flex items-center gap-1">
          <Label>Mask</Label>
          <TooltipIcon text={tooltips["pipeline.mask"]} />
        </div>
        <Select value={mask} onChange={(e) => setMask(e.target.value)}>
          <option value="from_halfmaps">Auto (from halfmaps)</option>
          <option value="sphere">Sphere</option>
          <option value="none">None</option>
          <option value="file">Custom .mrc file</option>
        </Select>
        {mask === "file" && (
          <div className="mt-1 space-y-1">
            <div className="flex gap-2">
              <PathInput
                value={maskPath}
                onChange={setMaskPath}
                accept={[".mrc"]}
                placeholder="/path/to/mask.mrc"
                className="font-mono"
              />
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowMaskBrowser(!showMaskBrowser)}
              >
                Browse
              </Button>
            </div>
            {showMaskBrowser && (
              <FileBrowser
                initialPath={projectPath}
                accept={[".mrc"]}
                onSelect={(path) => {
                  setMaskPath(path);
                  setShowMaskBrowser(false);
                }}
              />
            )}
          </div>
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
              <Label>Output Name</Label>
              <TooltipIcon text={tooltips["pipeline.output_name"]} />
            </div>
            <Input value={outputName} onChange={(e) => setOutputName(e.target.value)} placeholder="Auto-generated" />
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Label>zdim</Label>
              <TooltipIcon text={tooltips["pipeline.zdim"]} />
            </div>
            <Input value={zdim} onChange={(e) => setZdim(e.target.value)} placeholder="1,2,4,10,20" />
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Label>Downsample</Label>
              <TooltipIcon text={tooltips["pipeline.downsample"]} />
            </div>
            <Input
              type="number"
              value={downsample}
              onChange={(e) => setDownsample(e.target.value)}
              placeholder="256"
            />
          </div>

          <div className="flex gap-6">
            <label className="flex items-center gap-2 text-sm text-zinc-400">
              <input
                type="checkbox"
                checked={lazy}
                onChange={(e) => setLazy(e.target.checked)}
                className="rounded border-zinc-600 bg-zinc-800"
              />
              Lazy loading
              <TooltipIcon text={tooltips["pipeline.lazy"]} />
            </label>
            <label className="flex items-center gap-2 text-sm text-zinc-400">
              <input
                type="checkbox"
                checked={correctContrast}
                onChange={(e) => setCorrectContrast(e.target.checked)}
                className="rounded border-zinc-600 bg-zinc-800"
              />
              Correct contrast
              <TooltipIcon text={tooltips["pipeline.correct_contrast"]} />
            </label>
            <label className="flex items-center gap-2 text-sm text-zinc-400">
              <input
                type="checkbox"
                checked={tiltSeries}
                onChange={(e) => setTiltSeries(e.target.checked)}
                className="rounded border-zinc-600 bg-zinc-800"
              />
              Tilt series
              <TooltipIcon text={tooltips["pipeline.tilt_series"]} />
            </label>
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Label>Focus Mask</Label>
              <TooltipIcon text={tooltips["pipeline.focus_mask"]} />
            </div>
            <PathInput value={focusMask} onChange={setFocusMask} accept={[".mrc"]} placeholder="Optional .mrc path" className="font-mono" />
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Label>Data Directory</Label>
              <TooltipIcon text={tooltips["pipeline.datadir"]} />
            </div>
            <PathInput value={datadir} onChange={setDatadir} directoryOnly placeholder="Override data dir" className="font-mono" />
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Label>N Images</Label>
              <TooltipIcon text={tooltips["pipeline.n_images"]} />
            </div>
            <Input type="number" value={nImages} onChange={(e) => setNImages(e.target.value)} placeholder="All" />
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div className="space-y-1">
              <div className="flex items-center gap-1">
                <Label>Halfsets</Label>
                <TooltipIcon text={tooltips["pipeline.halfsets"]} />
              </div>
              <Input value={halfsets} onChange={(e) => setHalfsets(e.target.value)} placeholder="Column name" />
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-1">
                <Label>Poses</Label>
                <TooltipIcon text={tooltips["pipeline.poses"]} />
              </div>
              <Input value={poses} onChange={(e) => setPoses(e.target.value)} placeholder="Column prefix" />
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-1">
                <Label>CTF</Label>
                <TooltipIcon text={tooltips["pipeline.ctf"]} />
              </div>
              <Input value={ctf} onChange={(e) => setCtf(e.target.value)} placeholder="Column prefix" />
            </div>
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <Label>Strip Prefix</Label>
              <TooltipIcon text={tooltips["pipeline.strip_prefix"]} />
            </div>
            <Input value={stripPrefix} onChange={(e) => setStripPrefix(e.target.value)} placeholder="Prefix to strip" />
          </div>
        </div>
      )}

      {/* SLURM Settings */}
      <SlurmSettings value={slurmOpts} onChange={handleSlurmChange} />

      {/* Submit */}
      <div className="flex items-center justify-between pt-2">
        <div>
          {mutation.isError && (
            <span className="text-sm text-red-400">
              {(mutation.error as Error).message}
            </span>
          )}
        </div>
        <Button
          onClick={() => mutation.mutate()}
          disabled={!canSubmit}
          loading={mutation.isPending}
        >
          {mutation.isPending ? "Submitting..." : "Submit Pipeline Job"}
        </Button>
      </div>
    </div>
  );
}
