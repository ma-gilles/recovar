import { useCallback, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { PathInput } from "../ui/PathInput";
import { Label } from "../ui/label";
import { Select } from "../ui/select";
import { TooltipIcon } from "../ui/tooltip-icon";
import { FileBrowser } from "../file-browser/FileBrowser";
import { ExecutorSelector } from "./ExecutorSelector";
import { LocalSettings, type LocalOpts } from "./LocalSettings";
import { SlurmSettings, type SlurmOpts } from "./SlurmSettings";
import { tooltips } from "../../lib/tooltips";
import {
  getInitialModelDefaults,
  submitJob,
  validateJob,
  type InitialModelDefaults,
  type ValidationResult,
} from "../../lib/api/client";

interface InitialModelFormProps {
  projectId: string;
  projectPath: string;
  prefilledInput?: string;
  prefilledParams?: Record<string, unknown>;
  onSubmitted?: (jobId: string) => void;
}

interface FieldProps {
  label: string;
  tooltip: string;
  value: string;
  onChange: (value: string) => void;
  step?: string;
}

function NumberField({ label, tooltip, value, onChange, step }: FieldProps): React.JSX.Element {
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-1">
        <Label>{label}</Label>
        <TooltipIcon text={tooltips[tooltip]} />
      </div>
      <Input type="number" step={step} value={value} onChange={(event) => onChange(event.target.value)} />
    </div>
  );
}

function CheckField({
  label,
  tooltip,
  checked,
  onChange,
}: {
  label: string;
  tooltip: string;
  checked: boolean;
  onChange: (value: boolean) => void;
}): React.JSX.Element {
  return (
    <label className="flex items-center gap-2 text-sm text-zinc-400">
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="rounded border-zinc-600 bg-zinc-800"
      />
      {label}
      <TooltipIcon text={tooltips[tooltip]} />
    </label>
  );
}

export function InitialModelForm(props: InitialModelFormProps): React.JSX.Element {
  const defaults = useQuery({
    queryKey: ["initial-model-defaults"],
    queryFn: getInitialModelDefaults,
    staleTime: Infinity,
  });

  if (defaults.isPending) return <p className="text-sm text-zinc-400">Loading native InitialModel defaults...</p>;
  if (defaults.isError) {
    return <p className="text-sm text-red-400">Could not load InitialModel defaults: {defaults.error.message}</p>;
  }
  return <InitialModelFormLoaded {...props} defaults={defaults.data} />;
}

function InitialModelFormLoaded({
  projectId,
  projectPath,
  prefilledInput,
  prefilledParams,
  onSubmitted,
  defaults,
}: InitialModelFormProps & { defaults: InitialModelDefaults }): React.JSX.Element {
  const queryClient = useQueryClient();
  const p = prefilledParams ?? {};
  const pick = <T,>(name: string, fallback: T): T => (p[name] === undefined ? fallback : p[name] as T);
  const [inputStar, setInputStar] = useState(prefilledInput ?? String(p.input_star ?? ""));
  const [showBrowser, setShowBrowser] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [nrIter, setNrIter] = useState(String(pick("nr_iter", defaults.nr_iter)));
  const [gradWriteIter, setGradWriteIter] = useState(String(pick("grad_write_iter", defaults.grad_write_iter)));
  const [nrClasses, setNrClasses] = useState(String(pick("nr_classes", defaults.nr_classes)));
  const [tau2Fudge, setTau2Fudge] = useState(String(pick("tau2_fudge", defaults.tau2_fudge)));
  const [gradIniFrac, setGradIniFrac] = useState(String(pick("grad_ini_frac", defaults.grad_ini_frac)));
  const [gradFinFrac, setGradFinFrac] = useState(String(pick("grad_fin_frac", defaults.grad_fin_frac)));
  const [gradEmIters, setGradEmIters] = useState(String(pick("grad_em_iters", defaults.grad_em_iters)));
  const [stepsize, setStepsize] = useState(String(pick("stepsize", defaults.stepsize)));
  const [mu, setMu] = useState(String(pick("mu", defaults.mu)));
  const [symName, setSymName] = useState(String(pick("sym_name", defaults.sym_name)));
  const [particleDiameter, setParticleDiameter] = useState(String(pick("particle_diameter", defaults.particle_diameter)));
  const [runInC1, setRunInC1] = useState(Boolean(pick("do_run_C1", defaults.do_run_C1)));
  const [doSolvent, setDoSolvent] = useState(Boolean(pick("do_solvent", defaults.do_solvent)));
  const [doZeroMask, setDoZeroMask] = useState(Boolean(pick("do_zero_mask", defaults.do_zero_mask)));
  const [doCtf, setDoCtf] = useState(Boolean(pick("do_ctf_correction", defaults.do_ctf_correction)));
  const [randomSeed, setRandomSeed] = useState(String(pick("random_seed", defaults.random_seed)));
  const [healpixOrder, setHealpixOrder] = useState(String(pick("healpix_order", defaults.healpix_order)));
  const [oversampling, setOversampling] = useState(String(pick("oversampling", defaults.oversampling)));
  const [offsetRange, setOffsetRange] = useState(String(pick("offset_range_px", defaults.offset_range_px)));
  const [offsetStep, setOffsetStep] = useState(String(pick("offset_step_px", defaults.offset_step_px)));
  const [perturbationFactor, setPerturbationFactor] = useState(String(pick("perturbation_factor", defaults.perturbation_factor)));
  const [randomPerturbation, setRandomPerturbation] = useState(
    pick("random_perturbation", defaults.random_perturbation) === null ? "" : String(pick("random_perturbation", defaults.random_perturbation)),
  );
  const [imageBatchSize, setImageBatchSize] = useState(String(pick("image_batch_size", defaults.image_batch_size)));
  const [rotationBlockSize, setRotationBlockSize] = useState(String(pick("rotation_block_size", defaults.rotation_block_size)));
  const [pass2Engine, setPass2Engine] = useState(String(pick("pass2_engine", defaults.pass2_engine)));
  const [bootstrapMin, setBootstrapMin] = useState(String(pick("bootstrap_min_particles", defaults.bootstrap_min_particles)));
  const [sigma2Min, setSigma2Min] = useState(String(pick("sigma2_min_particles", defaults.sigma2_min_particles)));
  const [translationSigma, setTranslationSigma] = useState(
    pick("translation_sigma_angstrom", defaults.translation_sigma_angstrom) === null ? "" : String(pick("translation_sigma_angstrom", defaults.translation_sigma_angstrom)),
  );
  const [paddingFactor, setPaddingFactor] = useState(String(pick("padding_factor", defaults.padding_factor)));
  const [imageBackend, setImageBackend] = useState(String(pick("image_fourier_backend", defaults.image_fourier_backend)));
  const [gpuIds, setGpuIds] = useState(String(pick("gpu_ids", defaults.gpu_ids)));
  const [lazy, setLazy] = useState(Boolean(pick("lazy", defaults.lazy)));
  const [writeArtifacts, setWriteArtifacts] = useState(Boolean(pick("write_iter_artifacts", defaults.write_iter_artifacts)));
  const [requireCuda, setRequireCuda] = useState(Boolean(pick("require_custom_cuda", defaults.require_custom_cuda)));
  const [deterministicCuda, setDeterministicCuda] = useState(Boolean(pick("deterministic_cuda", defaults.deterministic_cuda)));
  const [datadir, setDatadir] = useState(String(p.datadir ?? ""));
  const [stripPrefix, setStripPrefix] = useState(String(p.strip_prefix ?? ""));
  const [executorMode, setExecutorMode] = useState<string | null>(null);
  const [slurmOpts, setSlurmOpts] = useState<SlurmOpts | null>(null);
  const [localOpts, setLocalOpts] = useState<LocalOpts | null>(null);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [validationWarnings, setValidationWarnings] = useState<string[]>([]);
  const [validating, setValidating] = useState(false);

  const buildParams = useCallback((): Record<string, unknown> => {
    const params: Record<string, unknown> = {
      input_star: inputStar,
      nr_iter: parseInt(nrIter),
      grad_write_iter: parseInt(gradWriteIter),
      nr_classes: parseInt(nrClasses),
      tau2_fudge: parseFloat(tau2Fudge),
      grad_ini_frac: parseFloat(gradIniFrac),
      grad_fin_frac: parseFloat(gradFinFrac),
      grad_em_iters: parseInt(gradEmIters),
      stepsize: parseFloat(stepsize),
      mu: parseFloat(mu),
      sym_name: symName,
      particle_diameter: parseFloat(particleDiameter),
      do_run_C1: runInC1,
      do_solvent: doSolvent,
      do_zero_mask: doZeroMask,
      do_ctf_correction: doCtf,
      random_seed: parseInt(randomSeed),
      healpix_order: parseInt(healpixOrder),
      oversampling: parseInt(oversampling),
      offset_range_px: parseFloat(offsetRange),
      offset_step_px: parseFloat(offsetStep),
      perturbation_factor: parseFloat(perturbationFactor),
      image_batch_size: parseInt(imageBatchSize),
      rotation_block_size: parseInt(rotationBlockSize),
      pass2_engine: pass2Engine,
      bootstrap_min_particles: parseInt(bootstrapMin),
      sigma2_min_particles: parseInt(sigma2Min),
      padding_factor: parseInt(paddingFactor),
      image_fourier_backend: imageBackend,
      gpu_ids: gpuIds,
      lazy,
      write_iter_artifacts: writeArtifacts,
      require_custom_cuda: requireCuda,
      deterministic_cuda: deterministicCuda,
    };
    if (randomPerturbation) params.random_perturbation = parseFloat(randomPerturbation);
    if (translationSigma) params.translation_sigma_angstrom = parseFloat(translationSigma);
    if (datadir) params.datadir = datadir;
    if (stripPrefix) params.strip_prefix = stripPrefix;
    if (slurmOpts && executorMode !== "local") params.slurm_opts = slurmOpts;
    if (localOpts && executorMode === "local") params.local_opts = localOpts;
    return params;
  }, [
    inputStar, nrIter, gradWriteIter, nrClasses, tau2Fudge, gradIniFrac, gradFinFrac,
    gradEmIters, stepsize, mu, symName, particleDiameter,
    runInC1, doSolvent, doZeroMask, doCtf, randomSeed, healpixOrder, oversampling,
    offsetRange, offsetStep, perturbationFactor, imageBatchSize, rotationBlockSize, pass2Engine,
    bootstrapMin, sigma2Min, paddingFactor, imageBackend, gpuIds, lazy, writeArtifacts,
    requireCuda, deterministicCuda, randomPerturbation, translationSigma, datadir,
    stripPrefix, slurmOpts, localOpts, executorMode,
  ]);

  const mutation = useMutation({
    mutationFn: () => submitJob(projectId, "initial_model", buildParams(), executorMode),
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
      const result: ValidationResult = await validateJob(projectId, "initial_model", buildParams());
      setValidationErrors(result.errors);
      setValidationWarnings(result.warnings);
      if (result.valid) mutation.mutate();
    } catch {
      mutation.mutate();
    } finally {
      setValidating(false);
    }
  }, [projectId, buildParams, mutation]);

  const positive = [nrIter, gradWriteIter, nrClasses, particleDiameter, imageBatchSize, rotationBlockSize,
    bootstrapMin, sigma2Min, paddingFactor].every((value) => Number(value) > 0);
  const canSubmit = inputStar.length > 0 && symName.length > 0 && positive;

  return (
    <div className="space-y-4">
      <div className="space-y-1">
        <div className="flex items-center gap-1">
          <Label>Input STAR</Label>
          <TooltipIcon text={tooltips["initial_model.input_star"]} />
        </div>
        <div className="flex gap-2">
          <PathInput value={inputStar} onChange={setInputStar} accept={[".star"]} placeholder="/path/to/particles.star" className="font-mono" />
          <Button variant="outline" size="sm" onClick={() => setShowBrowser(!showBrowser)}>Browse</Button>
        </div>
        {showBrowser && (
          <FileBrowser
            initialPath={projectPath}
            accept={[".star"]}
            onSelect={(path) => { setInputStar(path); setShowBrowser(false); }}
          />
        )}
      </div>

      <div className="grid grid-cols-2 gap-3">
        <NumberField label="Iterations" tooltip="initial_model.nr_iter" value={nrIter} onChange={setNrIter} />
        <NumberField label="Classes (K)" tooltip="initial_model.nr_classes" value={nrClasses} onChange={setNrClasses} />
        <NumberField label="Tau2 Fudge" tooltip="initial_model.tau2_fudge" value={tau2Fudge} onChange={setTau2Fudge} step="0.1" />
        <NumberField label="Particle Diameter (Å)" tooltip="initial_model.particle_diameter" value={particleDiameter} onChange={setParticleDiameter} step="0.1" />
        <div className="space-y-1">
          <div className="flex items-center gap-1"><Label>Symmetry</Label><TooltipIcon text={tooltips["initial_model.sym_name"]} /></div>
          <Input value={symName} onChange={(event) => setSymName(event.target.value)} />
        </div>
      </div>
      <CheckField label="Run refinement in C1" tooltip="initial_model.run_in_c1" checked={runInC1} onChange={setRunInC1} />

      <button onClick={() => setShowAdvanced(!showAdvanced)} className="flex items-center gap-1 text-sm text-zinc-500 hover:text-zinc-300">
        {showAdvanced ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        Advanced InitialModel controls
      </button>

      {showAdvanced && (
        <div className="ml-4 space-y-4 border-l border-zinc-800 pl-4">
          <div className="grid grid-cols-2 gap-3">
            <NumberField label="Write Every N Iterations" tooltip="initial_model.grad_write_iter" value={gradWriteIter} onChange={setGradWriteIter} />
            <NumberField label="Random Seed" tooltip="initial_model.random_seed" value={randomSeed} onChange={setRandomSeed} />
            <NumberField label="Healpix Order" tooltip="initial_model.healpix_order" value={healpixOrder} onChange={setHealpixOrder} />
            <NumberField label="Oversampling" tooltip="initial_model.oversampling" value={oversampling} onChange={setOversampling} />
            <NumberField label="Offset Range (px)" tooltip="initial_model.offset_range" value={offsetRange} onChange={setOffsetRange} step="0.1" />
            <NumberField label="Offset Step (px)" tooltip="initial_model.offset_step" value={offsetStep} onChange={setOffsetStep} step="0.1" />
            <NumberField label="Perturbation Factor" tooltip="initial_model.perturbation_factor" value={perturbationFactor} onChange={setPerturbationFactor} step="0.01" />
            <NumberField label="Fixed Perturbation" tooltip="initial_model.random_perturbation" value={randomPerturbation} onChange={setRandomPerturbation} step="0.01" />
            <NumberField label="Image Batch Size" tooltip="initial_model.image_batch_size" value={imageBatchSize} onChange={setImageBatchSize} />
            <NumberField label="Rotation Block Size" tooltip="initial_model.rotation_block_size" value={rotationBlockSize} onChange={setRotationBlockSize} />
            <NumberField label="Bootstrap Minimum" tooltip="initial_model.bootstrap_min" value={bootstrapMin} onChange={setBootstrapMin} />
            <NumberField label="Sigma2 Minimum" tooltip="initial_model.sigma2_min" value={sigma2Min} onChange={setSigma2Min} />
            <NumberField label="Translation Sigma (Å)" tooltip="initial_model.translation_sigma" value={translationSigma} onChange={setTranslationSigma} step="0.1" />
            <NumberField label="Padding Factor" tooltip="initial_model.padding_factor" value={paddingFactor} onChange={setPaddingFactor} />
            <NumberField label="Initial Phase Fraction" tooltip="initial_model.grad_ini_frac" value={gradIniFrac} onChange={setGradIniFrac} step="0.01" />
            <NumberField label="Final Phase Fraction" tooltip="initial_model.grad_fin_frac" value={gradFinFrac} onChange={setGradFinFrac} step="0.01" />
            <NumberField label="Terminal EM Iterations" tooltip="initial_model.grad_em_iters" value={gradEmIters} onChange={setGradEmIters} />
            <NumberField label="VDAM Step Size" tooltip="initial_model.stepsize" value={stepsize} onChange={setStepsize} step="0.01" />
            <NumberField label="VDAM Momentum (μ)" tooltip="initial_model.mu" value={mu} onChange={setMu} step="0.01" />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1">
              <div className="flex items-center gap-1"><Label>Pass-2 Engine</Label><TooltipIcon text={tooltips["initial_model.pass2_engine"]} /></div>
              <Select value={pass2Engine} onChange={(event) => setPass2Engine(event.target.value)}>
                <option value="auto">Auto</option><option value="local">Local</option><option value="compact">Compact K-class</option>
              </Select>
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-1"><Label>Fourier Backend</Label><TooltipIcon text={tooltips["initial_model.image_backend"]} /></div>
              <Select value={imageBackend} onChange={(event) => setImageBackend(event.target.value)}>
                <option value="auto">Auto</option><option value="relion_cuda">RELION CUDA</option>
                <option value="jax_gpu">JAX GPU</option><option value="host_numpy">Host NumPy</option>
              </Select>
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-1"><Label>GPU IDs</Label><TooltipIcon text={tooltips["initial_model.gpu_ids"]} /></div>
              <Input value={gpuIds} onChange={(event) => setGpuIds(event.target.value)} />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <CheckField label="Solvent flattening" tooltip="initial_model.solvent" checked={doSolvent} onChange={setDoSolvent} />
            <CheckField label="Zero outside mask" tooltip="initial_model.zero_mask" checked={doZeroMask} onChange={setDoZeroMask} />
            <CheckField label="CTF correction" tooltip="initial_model.ctf" checked={doCtf} onChange={setDoCtf} />
            <CheckField label="Lazy image loading" tooltip="initial_model.lazy" checked={lazy} onChange={setLazy} />
            <CheckField label="Write iteration artifacts" tooltip="initial_model.write_artifacts" checked={writeArtifacts} onChange={setWriteArtifacts} />
            <CheckField label="Require custom CUDA" tooltip="initial_model.require_cuda" checked={requireCuda} onChange={setRequireCuda} />
            <CheckField label="Deterministic CUDA diagnostics" tooltip="initial_model.deterministic_cuda" checked={deterministicCuda} onChange={setDeterministicCuda} />
          </div>
          <div className="space-y-1"><Label>Data Directory</Label><PathInput value={datadir} onChange={setDatadir} directoryOnly placeholder="Optional image path override" className="font-mono" /></div>
          <div className="space-y-1"><Label>Strip Prefix</Label><Input value={stripPrefix} onChange={(event) => setStripPrefix(event.target.value)} placeholder="Optional STAR path prefix" /></div>
        </div>
      )}

      <ExecutorSelector value={executorMode} onChange={setExecutorMode} />
      {executorMode === "local" ? <LocalSettings value={localOpts} onChange={setLocalOpts} /> : <SlurmSettings value={slurmOpts} onChange={setSlurmOpts} />}

      {validationErrors.map((error) => <p key={error} className="text-sm text-red-400">{error}</p>)}
      {validationWarnings.map((warning) => <p key={warning} className="text-sm text-amber-400">{warning}</p>)}
      {mutation.isError && <p className="text-sm text-red-400">{(mutation.error as Error).message}</p>}
      <div className="flex justify-end pt-2">
        <Button onClick={handleSubmit} disabled={!canSubmit} loading={validating || mutation.isPending}>
          {validating ? "Validating inputs..." : mutation.isPending ? "Submitting..." : "Submit InitialModel Job"}
        </Button>
      </div>
    </div>
  );
}
