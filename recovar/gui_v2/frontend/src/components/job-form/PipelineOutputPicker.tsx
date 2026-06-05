import { useState } from "react";
import { Button } from "../ui/button";
import { PathInput } from "../ui/PathInput";
import { Label } from "../ui/label";
import { TooltipIcon } from "../ui/tooltip-icon";
import { FileBrowser } from "../file-browser/FileBrowser";
import { useProject } from "../../lib/project-context";

interface PipelineOutputPickerProps {
  value: string;
  onChange: (path: string) => void;
  label?: string;
  tooltip?: string;
}

/**
 * Standard "pick a pipeline output directory" control: a path input + Browse
 * button that opens a directory FileBrowser. Shared by every downstream-job
 * form (Analyze, Density, ComputeState, ComputeTrajectory) so the way you
 * point a job at a pipeline output is identical everywhere. The browser
 * defaults to the current project directory rather than the filesystem root.
 */
export function PipelineOutputPicker({
  value,
  onChange,
  label = "Result Directory",
  tooltip,
}: PipelineOutputPickerProps): React.JSX.Element {
  const { project } = useProject();
  const [showBrowser, setShowBrowser] = useState(false);

  return (
    <div className="space-y-1">
      <div className="flex items-center gap-1">
        <Label>{label}</Label>
        {tooltip && <TooltipIcon text={tooltip} />}
      </div>
      <div className="flex gap-2">
        <PathInput
          value={value}
          onChange={onChange}
          directoryOnly
          placeholder="/path/to/pipeline/output"
          className="font-mono"
        />
        <Button
          variant="outline"
          size="sm"
          onClick={() => setShowBrowser(!showBrowser)}
        >
          Browse
        </Button>
      </div>
      {showBrowser && (
        <FileBrowser
          initialPath={value || project?.path || "/scratch/gpfs"}
          selectDirectory
          onSelect={(path) => {
            onChange(path);
            setShowBrowser(false);
          }}
        />
      )}
    </div>
  );
}
