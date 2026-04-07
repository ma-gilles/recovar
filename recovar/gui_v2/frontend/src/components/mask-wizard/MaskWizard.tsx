import { useCallback, useEffect, useRef, useState } from "react";
import { Save, Loader2 } from "lucide-react";
import { Dialog } from "../ui/dialog";
import { Button } from "../ui/button";
import { Spinner } from "../ui/spinner";
import {
  previewMask,
  saveMask,
  ApiError,
  type MaskParams,
} from "../../lib/api/client";

interface MaskWizardProps {
  open: boolean;
  onClose: () => void;
  sourcePath: string;
  sourceName: string;
  projectId: string;
  onSaved?: (path: string) => void;
}

interface WizardState {
  thresholdMode: "auto" | "manual";
  threshold: number;
  extend: number;
  softEdge: number;
  lowpassMode: "auto" | "manual";
  lowpass: number;
  cleanup: boolean;
}

const DEFAULT_STATE: WizardState = {
  thresholdMode: "auto",
  threshold: 0.02,
  extend: 3,
  softEdge: 6,
  lowpassMode: "auto",
  lowpass: 2,
  cleanup: true,
};

function buildParams(s: WizardState, sourcePath: string): MaskParams {
  return {
    source_path: sourcePath,
    threshold: s.thresholdMode === "auto" ? null : s.threshold,
    lowpass_sigma: s.lowpassMode === "auto" ? null : s.lowpass,
    extend: s.extend,
    soft_edge: s.softEdge,
    cleanup: s.cleanup,
  };
}

/**
 * Guided mask creation modal. Wraps recovar.core.mask.make_mask through
 * the /api/masks/preview and /api/masks/save endpoints.
 */
export function MaskWizard({
  open,
  onClose,
  sourcePath,
  sourceName,
  projectId,
  onSaved,
}: MaskWizardProps): React.JSX.Element {
  const [state, setState] = useState<WizardState>(DEFAULT_STATE);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [coverage, setCoverage] = useState<number | null>(null);
  const [shape, setShape] = useState<number[] | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);

  const [outputName, setOutputName] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savedPath, setSavedPath] = useState<string | null>(null);

  const [axis, setAxis] = useState<0 | 1 | 2>(2);
  const [sliceIdx, setSliceIdx] = useState<number | null>(null);

  // Debounced preview generation
  const previewTokenRef = useRef(0);
  const triggerPreview = useCallback(async () => {
    if (!open) return;
    const myToken = ++previewTokenRef.current;
    setPreviewLoading(true);
    setPreviewError(null);
    try {
      const params = buildParams(state, sourcePath);
      const result = await previewMask({
        ...params,
        axis,
        idx: sliceIdx,
      });
      if (myToken !== previewTokenRef.current) {
        URL.revokeObjectURL(result.url);
        return;
      }
      setPreviewUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return result.url;
      });
      setCoverage(result.coverage);
      setShape(result.shape);
    } catch (e) {
      if (myToken !== previewTokenRef.current) return;
      const msg = e instanceof ApiError ? e.message : String(e);
      setPreviewError(msg);
    } finally {
      if (myToken === previewTokenRef.current) setPreviewLoading(false);
    }
  }, [open, state, sourcePath, axis, sliceIdx]);

  // Auto-preview on open and on parameter changes (debounced 250ms).
  useEffect(() => {
    if (!open) return;
    const t = setTimeout(triggerPreview, 250);
    return () => clearTimeout(t);
  }, [open, triggerPreview]);

  // Reset state when modal opens with a new source
  useEffect(() => {
    if (open) {
      setState(DEFAULT_STATE);
      setOutputName(deriveDefaultName(sourceName));
      setSavedPath(null);
      setSaveError(null);
      setSliceIdx(null);
    }
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, sourcePath]);

  const handleSave = useCallback(async () => {
    if (!outputName.trim()) {
      setSaveError("Enter an output name");
      return;
    }
    setSaving(true);
    setSaveError(null);
    try {
      const result = await saveMask({
        ...buildParams(state, sourcePath),
        project_id: projectId,
        output_name: outputName.trim(),
      });
      setSavedPath(result.path);
      onSaved?.(result.path);
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : String(e);
      setSaveError(msg);
    } finally {
      setSaving(false);
    }
  }, [outputName, state, sourcePath, projectId, onSaved]);

  const maxSlice = shape ? shape[axis] - 1 : 0;
  const currentIdx = sliceIdx ?? (shape ? Math.floor(shape[axis] / 2) : 0);

  return (
    <Dialog open={open} onClose={onClose} className="!max-w-2xl">
      <h2 className="mb-1 text-lg font-semibold text-zinc-100">Create Mask</h2>
      <p className="mb-4 text-xs text-zinc-500">
        From volume: <span className="text-zinc-300">{sourceName}</span>
      </p>

      <div className="grid grid-cols-2 gap-5">
        {/* Left: preview */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium uppercase tracking-wider text-zinc-500">
              Preview
            </span>
            <div className="flex gap-1 rounded-md border border-zinc-700 p-0.5">
              {(["X", "Y", "Z"] as const).map((label, i) => (
                <button
                  key={label}
                  onClick={() => {
                    setAxis(i as 0 | 1 | 2);
                    setSliceIdx(null);
                  }}
                  className={
                    "rounded px-2 py-0.5 text-xs " +
                    (axis === i ? "bg-zinc-700 text-zinc-50" : "text-zinc-400")
                  }
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
          <div
            className="relative flex items-center justify-center rounded-lg border border-zinc-800 bg-black"
            style={{ aspectRatio: "1 / 1", minHeight: 220 }}
          >
            {previewError ? (
              <p className="px-3 text-center text-xs text-red-400">{previewError}</p>
            ) : previewUrl ? (
              <img
                src={previewUrl}
                alt="Mask preview"
                className="h-full w-full object-contain"
                style={{ imageRendering: "pixelated" }}
              />
            ) : (
              <Spinner label="Generating..." />
            )}
            {previewLoading && previewUrl && (
              <div className="absolute right-2 top-2">
                <Loader2 className="h-4 w-4 animate-spin text-zinc-400" />
              </div>
            )}
          </div>
          {shape && (
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-xs text-zinc-500">
                <span className="w-10">Slice</span>
                <input
                  type="range"
                  min={0}
                  max={maxSlice}
                  value={currentIdx}
                  onChange={(e) => setSliceIdx(parseInt(e.target.value))}
                  className="flex-1"
                />
                <span className="w-12 text-right">{currentIdx} / {maxSlice}</span>
              </div>
              <div className="flex items-center justify-between text-xs text-zinc-500">
                <span>Shape: {shape.join(" x ")}</span>
                {coverage !== null && (
                  <span>Coverage: {(coverage * 100).toFixed(1)}%</span>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Right: parameters */}
        <div className="space-y-3 text-sm">
          {/* Threshold */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium uppercase tracking-wider text-zinc-500">
                Threshold
              </label>
              <div className="flex gap-1 rounded-md border border-zinc-700 p-0.5">
                {(["auto", "manual"] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setState((s) => ({ ...s, thresholdMode: m }))}
                    className={
                      "rounded px-2 py-0.5 text-xs " +
                      (state.thresholdMode === m
                        ? "bg-zinc-700 text-zinc-50"
                        : "text-zinc-400")
                    }
                  >
                    {m}
                  </button>
                ))}
              </div>
            </div>
            {state.thresholdMode === "manual" ? (
              <div className="flex items-center gap-2 text-xs text-zinc-500">
                <input
                  type="range"
                  min={-0.5}
                  max={2}
                  step={0.001}
                  value={state.threshold}
                  onChange={(e) =>
                    setState((s) => ({ ...s, threshold: parseFloat(e.target.value) }))
                  }
                  className="flex-1"
                />
                <span className="w-14 text-right text-zinc-300">
                  {state.threshold.toFixed(3)}
                </span>
              </div>
            ) : (
              <p className="text-xs text-zinc-600">Otsu auto-threshold</p>
            )}
          </div>

          {/* Extend (dilate) */}
          <div className="space-y-1">
            <label className="text-xs font-medium uppercase tracking-wider text-zinc-500">
              Dilation: {state.extend} voxels
            </label>
            <input
              type="range"
              min={0}
              max={20}
              step={1}
              value={state.extend}
              onChange={(e) =>
                setState((s) => ({ ...s, extend: parseInt(e.target.value) }))
              }
              className="w-full"
            />
          </div>

          {/* Soft edge */}
          <div className="space-y-1">
            <label className="text-xs font-medium uppercase tracking-wider text-zinc-500">
              Soft edge: {state.softEdge.toFixed(1)} voxels
            </label>
            <input
              type="range"
              min={0}
              max={20}
              step={0.5}
              value={state.softEdge}
              onChange={(e) =>
                setState((s) => ({ ...s, softEdge: parseFloat(e.target.value) }))
              }
              className="w-full"
            />
          </div>

          {/* Lowpass */}
          <div className="space-y-1">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium uppercase tracking-wider text-zinc-500">
                Lowpass smoothing
              </label>
              <div className="flex gap-1 rounded-md border border-zinc-700 p-0.5">
                {(["auto", "manual"] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setState((s) => ({ ...s, lowpassMode: m }))}
                    className={
                      "rounded px-2 py-0.5 text-xs " +
                      (state.lowpassMode === m
                        ? "bg-zinc-700 text-zinc-50"
                        : "text-zinc-400")
                    }
                  >
                    {m}
                  </button>
                ))}
              </div>
            </div>
            {state.lowpassMode === "manual" && (
              <div className="flex items-center gap-2 text-xs text-zinc-500">
                <input
                  type="range"
                  min={0}
                  max={10}
                  step={0.5}
                  value={state.lowpass}
                  onChange={(e) =>
                    setState((s) => ({ ...s, lowpass: parseFloat(e.target.value) }))
                  }
                  className="flex-1"
                />
                <span className="w-12 text-right text-zinc-300">
                  σ {state.lowpass.toFixed(1)}
                </span>
              </div>
            )}
          </div>

          {/* Cleanup */}
          <div className="flex items-center gap-2">
            <input
              id="mask-cleanup"
              type="checkbox"
              checked={state.cleanup}
              onChange={(e) => setState((s) => ({ ...s, cleanup: e.target.checked }))}
              className="h-3.5 w-3.5"
            />
            <label htmlFor="mask-cleanup" className="text-xs text-zinc-400">
              Cleanup (fill holes, keep largest component)
            </label>
          </div>

          {/* Output name */}
          <div className="border-t border-zinc-800 pt-3 space-y-1">
            <label className="text-xs font-medium uppercase tracking-wider text-zinc-500">
              Save as
            </label>
            <input
              type="text"
              value={outputName}
              onChange={(e) => setOutputName(e.target.value)}
              placeholder="mask_name"
              className="w-full rounded border border-zinc-700 bg-zinc-950 px-2 py-1 text-sm text-zinc-200 placeholder-zinc-600 focus:border-blue-500 focus:outline-none"
            />
            <p className="text-xs text-zinc-600">
              Saved to <code>&lt;project&gt;/Masks/</code>
            </p>
          </div>

          {/* Status */}
          {saveError && (
            <p className="text-xs text-red-400" role="alert">
              {saveError}
            </p>
          )}
          {savedPath && (
            <p className="text-xs text-emerald-400">
              Saved: {savedPath.split("/").pop()}
            </p>
          )}

          {/* Actions */}
          <div className="flex gap-2 pt-2">
            <Button variant="outline" size="sm" onClick={onClose}>
              Close
            </Button>
            <Button
              variant="default"
              size="sm"
              onClick={handleSave}
              disabled={saving || previewLoading}
              className="ml-auto"
            >
              {saving ? (
                <>
                  <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                  Saving
                </>
              ) : (
                <>
                  <Save className="mr-1.5 h-3.5 w-3.5" />
                  Save Mask
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </Dialog>
  );
}

function deriveDefaultName(sourceName: string): string {
  const stem = sourceName.replace(/\.mrc$/i, "");
  return `mask_from_${stem}`.replace(/[^A-Za-z0-9_.-]/g, "_");
}
