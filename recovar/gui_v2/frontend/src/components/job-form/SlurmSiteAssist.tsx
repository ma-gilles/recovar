import { useMutation, useQuery } from "@tanstack/react-query";
import { CheckCircle2, FlaskConical, XCircle } from "lucide-react";
import { getSlurmOptions, testSlurmSettings, type SlurmOptions } from "../../lib/api/client";

export const PARTITION_LIST_ID = "slurm-partitions";
export const ACCOUNT_LIST_ID = "slurm-accounts";

export function useSlurmOptions(): SlurmOptions | undefined {
  return useQuery<SlurmOptions>({
    queryKey: ["slurm-options"],
    queryFn: getSlurmOptions,
    staleTime: Infinity,
  }).data;
}

/**
 * Suggestions for the Partition and Account inputs, read from this cluster.
 * Attach with `list={PARTITION_LIST_ID}` / `list={ACCOUNT_LIST_ID}`; typing a
 * value that is not listed still works.
 */
export function SlurmDatalists(): React.JSX.Element | null {
  const options = useSlurmOptions();
  if (!options) return null;
  return (
    <>
      <datalist id={PARTITION_LIST_ID}>
        {options.partitions
          .filter((p) => p.gpus && p.up)
          .map((p) => (
            <option key={p.name} value={p.name}>
              {`GPU nodes, up to ${Math.floor(p.max_node_memory_mb / 1024)}G memory${p.is_default ? " (default)" : ""}`}
            </option>
          ))}
      </datalist>
      <datalist id={ACCOUNT_LIST_ID}>
        {options.accounts.map((a) => (
          <option key={a} value={a} />
        ))}
      </datalist>
    </>
  );
}

/** One line saying what was found to work on this cluster, if anything was. */
export function SlurmDetectedNote(): React.JSX.Element | null {
  const detected = useSlurmOptions()?.detected ?? {};
  const parts = Object.entries(detected).map(([k, v]) => `${k} ${v}`);
  if (parts.length === 0) return null;
  return (
    <p className="text-xs text-zinc-500">
      Detected for this cluster: <span className="font-mono text-zinc-400">{parts.join(", ")}</span>. Used
      wherever you leave a field blank.
    </p>
  );
}

/** Asks the scheduler whether it accepts these settings; nothing is queued. */
export function SlurmTestButton({
  partition,
  account,
  memory,
}: {
  partition: string;
  account: string;
  memory: string;
}): React.JSX.Element {
  const mutation = useMutation({ mutationFn: () => testSlurmSettings({ partition, account, memory }) });
  const result = mutation.data;
  return (
    <div className="space-y-1">
      <button
        type="button"
        onClick={() => mutation.mutate()}
        disabled={mutation.isPending}
        className="flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
      >
        <FlaskConical className="h-3 w-3" />
        {mutation.isPending ? "Asking the scheduler..." : "Test these settings (nothing is submitted)"}
      </button>
      {result?.accepted === true && (
        <p className="flex items-center gap-1 text-xs text-emerald-400">
          <CheckCircle2 className="h-3 w-3" /> The scheduler accepts a 1-GPU job with these settings.
        </p>
      )}
      {result?.accepted === false && (
        <p className="flex items-start gap-1 text-xs text-red-400">
          <XCircle className="mt-0.5 h-3 w-3 shrink-0" />
          <span>The scheduler refuses these settings: {result.message}</span>
        </p>
      )}
      {result && result.accepted === null && (
        <p className="text-xs text-amber-400">Could not ask the scheduler: {result.message}</p>
      )}
      {mutation.isError && <p className="text-xs text-red-400">{(mutation.error as Error).message}</p>}
    </div>
  );
}
