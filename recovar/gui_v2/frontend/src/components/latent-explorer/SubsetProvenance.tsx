import { useQuery } from "@tanstack/react-query";
import { Info } from "lucide-react";
import { useState, useCallback } from "react";
import { getSubsetProvenance, type SubsetProvenance as SubsetProvenanceData } from "../../lib/api/client";
import { Dialog } from "../ui/dialog";
import { Button } from "../ui/button";
import { Spinner } from "../ui/spinner";

interface SubsetProvenanceButtonProps {
  subsetId: string;
}

/**
 * Small info button that opens a provenance dialog for a subset.
 */
export function SubsetProvenanceButton({ subsetId }: SubsetProvenanceButtonProps): React.JSX.Element {
  const [open, setOpen] = useState(false);

  const handleOpen = useCallback(() => setOpen(true), []);
  const handleClose = useCallback(() => setOpen(false), []);

  return (
    <>
      <Button
        variant="ghost"
        size="icon"
        onClick={handleOpen}
        title="Show provenance"
        aria-label="Show subset provenance"
      >
        <Info className="h-3.5 w-3.5 text-zinc-400" />
      </Button>
      {open && (
        <SubsetProvenanceDialog subsetId={subsetId} onClose={handleClose} />
      )}
    </>
  );
}

interface SubsetProvenanceDialogProps {
  subsetId: string;
  onClose: () => void;
}

function SubsetProvenanceDialog({ subsetId, onClose }: SubsetProvenanceDialogProps): React.JSX.Element {
  const { data, isLoading, error } = useQuery({
    queryKey: ["subset-provenance", subsetId],
    queryFn: () => getSubsetProvenance(subsetId),
  });

  return (
    <Dialog open onClose={onClose}>
      <div className="space-y-4">
        <h3 className="text-sm font-semibold text-zinc-100">Subset Provenance</h3>
        {isLoading && <Spinner label="Loading provenance..." />}
        {error && (
          <p className="text-xs text-red-400">
            Failed to load provenance: {(error as Error).message}
          </p>
        )}
        {data && <ProvenanceDetails data={data} />}
      </div>
    </Dialog>
  );
}

function ProvenanceDetails({ data }: { data: SubsetProvenanceData }): React.JSX.Element {
  const method = data.method as Record<string, unknown> | null | undefined;
  const selectionTool = method?.type as string | undefined;
  const created = new Date(data.created);

  return (
    <div className="space-y-3 text-xs">
      <Row label="Name" value={data.name} />
      <Row label="Created" value={created.toLocaleString()} />
      <Row label="Particles" value={data.n_particles.toLocaleString()} />
      {data.zdim != null && <Row label="zdim" value={String(data.zdim)} />}
      {selectionTool && <Row label="Selection tool" value={selectionTool} />}
      {Array.isArray(method?.axes) && (
        <Row
          label="PCA axes"
          value={`PC${(method.axes as number[])[0] + 1}, PC${(method.axes as number[])[1] + 1}`}
        />
      )}
      {data.source_job_id && (
        <Row
          label="Source job"
          value={data.source_job_name ?? data.source_job_id}
          href={`/jobs/${data.source_job_id}`}
        />
      )}
      <Row label=".ind file" value={data.ind_path} mono />
      {data.star_exports.length > 0 && (
        <div>
          <span className="text-zinc-500">.star exports:</span>
          <div className="mt-1 space-y-0.5">
            {data.star_exports.map((p) => (
              <div key={p} className="font-mono text-zinc-300 break-all">{p}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Row({
  label,
  value,
  href,
  mono,
}: {
  label: string;
  value: string;
  href?: string;
  mono?: boolean;
}): React.JSX.Element {
  return (
    <div className="flex gap-2">
      <span className="shrink-0 text-zinc-500">{label}:</span>
      {href ? (
        <a href={href} className="text-blue-400 hover:underline break-all">
          {value}
        </a>
      ) : (
        <span className={mono ? "font-mono text-zinc-300 break-all" : "text-zinc-300"}>
          {value}
        </span>
      )}
    </div>
  );
}

/**
 * Inline provenance summary shown after subset creation.
 */
export function SubsetProvenanceSummary({
  name,
  nParticles,
  zdim,
  selectionTool,
  sourceJobId,
}: {
  name: string;
  nParticles: number;
  zdim?: number | null;
  selectionTool?: string | null;
  sourceJobId?: string | null;
}): React.JSX.Element {
  return (
    <span className="text-xs text-emerald-400/80">
      {name} -- {nParticles.toLocaleString()} particles
      {selectionTool && ` via ${selectionTool}`}
      {zdim != null && ` (zdim=${zdim})`}
      {sourceJobId && (
        <>
          {" from "}
          <a href={`/jobs/${sourceJobId}`} className="text-blue-400 hover:underline">
            job
          </a>
        </>
      )}
    </span>
  );
}
