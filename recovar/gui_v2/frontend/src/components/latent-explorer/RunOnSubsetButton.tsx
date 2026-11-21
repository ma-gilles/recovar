import { useMutation } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { Play } from "lucide-react";
import { getSubsetPipelineParams } from "../../lib/api/client";
import { Button } from "../ui/button";

/**
 * Opens the pipeline form pre-filled to rerun the source pipeline on a subset:
 * same particles/poses/ctf/mask/settings, restricted with the subset's .ind.
 */
export function RunOnSubsetButton({ subsetId }: { subsetId: string }): React.JSX.Element {
  const navigate = useNavigate();
  const mutation = useMutation({
    mutationFn: () => getSubsetPipelineParams(subsetId),
    onSuccess: (data) => {
      navigate({
        to: "/jobs/new",
        search: {
          type: "pipeline",
          params: JSON.stringify(data.params),
          result_dir: undefined,
          density: undefined,
          input: undefined,
          particles: undefined,
        },
      });
    },
  });

  return (
    <span className="inline-flex items-center gap-2">
      <Button size="sm" onClick={() => mutation.mutate()} loading={mutation.isPending}>
        <Play className="h-3.5 w-3.5" />
        Run pipeline on this subset
      </Button>
      {mutation.isError && (
        <span className="text-xs text-red-400">{(mutation.error as Error).message}</span>
      )}
    </span>
  );
}
