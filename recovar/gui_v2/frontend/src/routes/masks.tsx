import React, { Suspense, useMemo, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Wand2, GitMerge, FolderOpen, Loader2 } from "lucide-react";
import { useProject } from "../lib/project-context";
import {
  listProjectMasks,
  maskBooleanOp,
  ApiError,
  type MaskInfo,
} from "../lib/api/client";
import { Button } from "../components/ui/button";
import { Spinner } from "../components/ui/spinner";

const VtkViewer = React.lazy(() =>
  import("../components/volume-viewer/VtkViewer").then((m) => ({ default: m.VtkViewer }))
);

const OPS = [
  { value: "union", label: "Union (A ∪ B)", desc: "max(A, B)" },
  { value: "intersect", label: "Intersect (A ∩ B)", desc: "min(A, B)" },
  { value: "subtract", label: "Subtract (A − B)", desc: "A * (1 - B)" },
] as const;

type OpValue = (typeof OPS)[number]["value"];

export function MasksPage(): React.JSX.Element {
  const { project } = useProject();
  const queryClient = useQueryClient();
  const [activePath, setActivePath] = useState<string | null>(null);
  const [selectedA, setSelectedA] = useState<string>("");
  const [selectedB, setSelectedB] = useState<string>("");
  const [op, setOp] = useState<OpValue>("union");
  const [outputName, setOutputName] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [lastSaved, setLastSaved] = useState<MaskInfo | null>(null);

  const { data: masks, isLoading } = useQuery<MaskInfo[]>({
    queryKey: ["project-masks", project?.id],
    queryFn: () => listProjectMasks(project!.id),
    enabled: !!project?.id,
  });

  const opMutation = useMutation({
    mutationFn: () =>
      maskBooleanOp({
        project_id: project!.id,
        mask_a: selectedA,
        mask_b: selectedB,
        op,
        output_name: outputName.trim(),
      }),
    onSuccess: (info) => {
      setLastSaved(info);
      setError(null);
      // Clear the output name so a repeated op doesn't silently collide
      // with the just-saved mask (the server-side default name differs).
      setOutputName("");
      queryClient.invalidateQueries({ queryKey: ["project-masks", project?.id] });
    },
    onError: (e) => {
      setError(e instanceof ApiError ? e.message : String(e));
    },
  });

  // Both masks picked and distinct. (Shape compatibility is validated
  // server-side and surfaced via `error`.)
  const bothSelected = !!selectedA && !!selectedB && selectedA !== selectedB;
  const canRun = bothSelected && outputName.trim().length > 0 && !opMutation.isPending;

  // Auto-derive a default output name when both masks are picked.
  const previewName = useMemo(() => {
    if (!selectedA || !selectedB) return "";
    const stem = (p: string) => p.split("/").pop()!.replace(/\.mrc$/i, "");
    const sym = op === "union" ? "or" : op === "intersect" ? "and" : "minus";
    return `${stem(selectedA)}_${sym}_${stem(selectedB)}`;
  }, [selectedA, selectedB, op]);

  if (!project) {
    return (
      <div className="space-y-4">
        <h1 className="text-xl font-semibold">Masks</h1>
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-8 text-center">
          <FolderOpen className="mx-auto h-10 w-10 text-zinc-600" />
          <p className="mt-3 text-zinc-400">Open a project to see its masks.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Wand2 className="h-5 w-5 text-emerald-400" />
        <h1 className="text-xl font-semibold">Masks</h1>
        <span className="text-xs text-zinc-500">
          {masks?.length ?? 0} mask{masks?.length === 1 ? "" : "s"} in {project.path}/Masks/
        </span>
      </div>

      {isLoading && <Spinner label="Loading masks..." />}

      {!isLoading && masks && masks.length === 0 && (
        <div className="rounded-lg border border-dashed border-zinc-800 p-8 text-center text-sm text-zinc-500">
          No masks yet. Generate one from any volume by clicking the green
          wand icon on a job's Volumes tab.
        </div>
      )}

      {masks && masks.length > 0 && (
        <div className="grid grid-cols-2 gap-6 lg:grid-cols-3">
          {/* Left: list */}
          <div className="space-y-1 lg:col-span-1">
            <h2 className="text-xs font-medium uppercase tracking-wider text-zinc-500">
              Available
            </h2>
            <div className="space-y-1 rounded-lg border border-zinc-800 bg-zinc-950 p-2">
              {masks.map((m) => (
                <button
                  key={m.path}
                  onClick={() => setActivePath(m.path)}
                  className={
                    "block w-full truncate rounded px-2 py-1 text-left text-sm " +
                    (m.path === activePath
                      ? "bg-emerald-500/15 text-emerald-200"
                      : "text-zinc-300 hover:bg-zinc-800")
                  }
                  title={m.path}
                >
                  {m.name}
                  <span className="ml-2 text-[10px] text-zinc-500">
                    {(m.size_bytes / 1e6).toFixed(1)} MB
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Middle: preview */}
          <div className="space-y-1 lg:col-span-2">
            <h2 className="text-xs font-medium uppercase tracking-wider text-zinc-500">
              Preview
            </h2>
            <div
              className="rounded-lg border border-zinc-800 bg-black"
              style={{ height: 360 }}
            >
              {activePath ? (
                <Suspense fallback={<Spinner label="Loading 3D viewer..." />}>
                  <VtkViewer
                    activeVolume={activePath}
                    activeSigma={0.5}
                    pinnedVolumes={[
                      {
                        path: activePath,
                        name: activePath.split("/").pop() ?? "mask",
                        threshold: 0.5,
                        opacity: 0.85,
                        visible: true,
                        colorIndex: 2,
                      },
                    ]}
                  />
                </Suspense>
              ) : (
                <div className="flex h-full items-center justify-center text-xs text-zinc-500">
                  Select a mask from the list to preview it.
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Boolean ops */}
      {masks && masks.length >= 2 && (
        <div className="space-y-3 rounded-lg border border-zinc-800 bg-zinc-900 p-5">
          <h2 className="flex items-center gap-2 text-sm font-medium uppercase tracking-wider text-zinc-500">
            <GitMerge className="h-4 w-4" /> Boolean Operations
          </h2>
          <p className="text-xs text-zinc-500">
            Combine two existing masks. Both must have the same shape.
          </p>
          <div className="grid grid-cols-3 gap-3">
            <div className="space-y-1">
              <label className="text-xs text-zinc-500">Mask A</label>
              <select
                value={selectedA}
                onChange={(e) => {
                  setSelectedA(e.target.value);
                  setLastSaved(null);
                }}
                className="w-full rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-sm text-zinc-200"
              >
                <option value="">— pick a mask —</option>
                {masks.map((m) => (
                  <option key={m.path} value={m.path}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>
            <div className="space-y-1">
              <label className="text-xs text-zinc-500">Operation</label>
              <select
                value={op}
                onChange={(e) => {
                  setOp(e.target.value as OpValue);
                  setLastSaved(null);
                }}
                className="w-full rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-sm text-zinc-200"
              >
                {OPS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
              <p className="text-[11px] text-zinc-600">
                {OPS.find((o) => o.value === op)?.desc}
              </p>
            </div>
            <div className="space-y-1">
              <label className="text-xs text-zinc-500">Mask B</label>
              <select
                value={selectedB}
                onChange={(e) => {
                  setSelectedB(e.target.value);
                  setLastSaved(null);
                }}
                className="w-full rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-sm text-zinc-200"
              >
                <option value="">— pick a mask —</option>
                {masks.map((m) => (
                  <option key={m.path} value={m.path}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="flex items-end gap-3">
            <div className="flex-1 space-y-1">
              <label className="text-xs text-zinc-500">Output name</label>
              <input
                value={outputName}
                onChange={(e) => setOutputName(e.target.value)}
                placeholder={previewName || "result_mask"}
                className="w-full rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-sm text-zinc-200 placeholder-zinc-600"
              />
              <p className="text-[11px] text-zinc-600">
                Saved to <code>&lt;project&gt;/Masks/</code>
              </p>
            </div>
            <Button onClick={() => opMutation.mutate()} disabled={!canRun}>
              {opMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" /> Running…
                </>
              ) : (
                <>Run</>
              )}
            </Button>
          </div>
          {error && <p className="text-xs text-red-400">{error}</p>}
          {lastSaved && (
            <p className="text-xs text-emerald-400">
              Saved <code>{lastSaved.name}</code> ({(lastSaved.size_bytes / 1e6).toFixed(1)} MB)
            </p>
          )}
        </div>
      )}
    </div>
  );
}
