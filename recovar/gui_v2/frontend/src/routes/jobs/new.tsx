import { useState } from "react";
import { useNavigate, useSearch, Link } from "@tanstack/react-router";
import { FolderOpen } from "lucide-react";
import { useProject } from "../../lib/project-context";
import { Select } from "../../components/ui/select";
import { Label } from "../../components/ui/label";
import { PipelineForm } from "../../components/job-form/PipelineForm";
import { AnalyzeForm } from "../../components/job-form/AnalyzeForm";
import { ComputeStateForm } from "../../components/job-form/ComputeStateForm";
import { ComputeTrajectoryForm } from "../../components/job-form/ComputeTrajectoryForm";
import { DensityForm } from "../../components/job-form/DensityForm";
import { StableStatesForm } from "../../components/job-form/StableStatesForm";
import { PostprocessForm } from "../../components/job-form/PostprocessForm";
import { DownsampleForm } from "../../components/job-form/DownsampleForm";

const JOB_TYPES = [
  { value: "pipeline", label: "Pipeline" },
  { value: "analyze", label: "Analyze" },
  { value: "compute_state", label: "Compute State" },
  { value: "compute_trajectory", label: "Compute Trajectory" },
  { value: "density", label: "Density Estimation" },
  { value: "stable_states", label: "Stable States" },
  { value: "postprocess", label: "Postprocess" },
  { value: "downsample", label: "Downsample" },
] as const;

export function NewJobPage(): React.JSX.Element {
  const navigate = useNavigate();
  const { project } = useProject();
  const searchParams = useSearch({ from: "/jobs/new" }) as {
    type?: string;
    result_dir?: string;
    density?: string;
    input?: string;
    particles?: string;
    params?: string;
  };
  const [jobType, setJobType] = useState<string>(() => {
    const raw = searchParams.type ?? "pipeline";
    // Normalize: URL may use "Pipeline" but option values are lowercase
    const found = JOB_TYPES.find((t) => t.value === raw.toLowerCase() || t.label === raw);
    return found ? found.value : raw.toLowerCase();
  });

  if (!project) {
    return (
      <div className="space-y-6">
        <h1 className="text-xl font-semibold">New Job</h1>
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-8 text-center">
          <FolderOpen className="mx-auto h-10 w-10 text-zinc-600" />
          <p className="mt-3 text-zinc-400">
            You need to create or open a project before submitting jobs.
          </p>
          <Link
            to="/"
            className="mt-4 inline-flex items-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500"
          >
            Go to Dashboard
          </Link>
        </div>
      </div>
    );
  }

  const handleSubmitted = (jobId: string) => {
    navigate({ to: "/jobs/$jobId", params: { jobId } });
  };

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-semibold">New Job</h1>

      <div className="max-w-2xl space-y-6">
        <div className="space-y-1">
          <Label>Job Type</Label>
          <Select value={jobType} onChange={(e) => setJobType(e.target.value)}>
            {JOB_TYPES.map((t) => (
              <option key={t.value} value={t.value}>
                {t.label}
              </option>
            ))}
          </Select>
        </div>

        <div key={jobType} className="rounded-lg border border-zinc-800 bg-zinc-900 p-6">
          {jobType === "pipeline" && (
            <PipelineForm
              projectId={project.id}
              projectPath={project.path}
              onSubmitted={handleSubmitted}
            />
          )}
          {jobType === "analyze" && (
            <AnalyzeForm
              projectId={project.id}
              prefilledResultDir={searchParams.result_dir}
              onSubmitted={handleSubmitted}
            />
          )}
          {jobType === "compute_state" && (
            <ComputeStateForm
              projectId={project.id}
              prefilledResultDir={searchParams.result_dir}
              onSubmitted={handleSubmitted}
            />
          )}
          {jobType === "compute_trajectory" && (
            <ComputeTrajectoryForm
              projectId={project.id}
              prefilledResultDir={searchParams.result_dir}
              onSubmitted={handleSubmitted}
            />
          )}
          {jobType === "density" && (
            <DensityForm
              projectId={project.id}
              prefilledResultDir={searchParams.result_dir}
              onSubmitted={handleSubmitted}
            />
          )}
          {jobType === "stable_states" && (
            <StableStatesForm
              projectId={project.id}
              prefilledDensity={searchParams.density}
              onSubmitted={handleSubmitted}
            />
          )}
          {jobType === "postprocess" && (
            <PostprocessForm
              projectId={project.id}
              prefilledInput={searchParams.input}
              onSubmitted={handleSubmitted}
            />
          )}
          {jobType === "downsample" && (
            <DownsampleForm
              projectId={project.id}
              prefilledParticles={searchParams.particles}
              onSubmitted={handleSubmitted}
            />
          )}
        </div>
      </div>
    </div>
  );
}
