import { useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, RefreshCw, XCircle } from "lucide-react";
import { getPreflight, type Preflight } from "../../lib/api/client";

/**
 * Lists everything about the server's machine that will make jobs fail
 * (process limits, wrong partition, missing CUDA kernel, ...) together with the
 * command that fixes it. Renders nothing when every check passes.
 */
export function PreflightBanner(): React.JSX.Element | null {
  const queryClient = useQueryClient();
  const { data, isFetching } = useQuery<Preflight>({
    queryKey: ["preflight"],
    queryFn: () => getPreflight(),
    staleTime: Infinity,
  });

  const problems = (data?.checks ?? []).filter((c) => c.level !== "ok");
  if (problems.length === 0) return null;

  const recheck = async (): Promise<void> => {
    queryClient.setQueryData(["preflight"], await getPreflight(true));
  };

  return (
    <div className="rounded-lg border border-amber-900/60 bg-amber-950/30 p-4" role="alert">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-amber-200">
          This machine needs attention before jobs will run
        </h2>
        <button
          type="button"
          onClick={recheck}
          disabled={isFetching}
          className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-300 hover:bg-zinc-800"
        >
          <RefreshCw className="h-3 w-3" />
          Check again
        </button>
      </div>
      <ul className="mt-3 space-y-3">
        {problems.map((c) => (
          <li key={c.id} className="flex gap-2 text-sm">
            {c.level === "error" ? (
              <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-400" />
            ) : (
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
            )}
            <div className="min-w-0">
              <p className="text-zinc-200">
                <span className="font-medium">{c.title}:</span> {c.detail}
              </p>
              {c.fix && (
                <pre className="mt-1 overflow-x-auto whitespace-pre-wrap rounded bg-zinc-950 px-2 py-1 font-mono text-xs text-zinc-300">
                  {c.fix}
                </pre>
              )}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
