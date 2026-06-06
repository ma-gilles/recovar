import { useEffect, useRef, useCallback } from "react";

// D3 categorical 10 colors for k-means clusters
const CLUSTER_COLORS = [
  [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
  [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
  [188, 189, 34], [23, 190, 207],
];

interface HistogramPanelProps {
  /** Interleaved xy coords from PCA projection: [x0,y0,x1,y1,...] — only x values are used */
  values: Float32Array;
  /** Per-point cluster labels (optional) */
  labels?: Int32Array | null;
  /** X axis label */
  xLabel: string;
  /** Title */
  title: string;
  /** Currently selected indices */
  selectedIndices?: Set<number>;
  /** Point click callback — selects the nearest particle to the clicked bin */
  onPointClick?: (index: number, coords: [number, number]) => void;
}

const NUM_BINS = 80;

/**
 * Canvas-based histogram panel for zdim=1.
 * Draws a distribution of PC1 values instead of a degenerate X=Y scatter plot.
 */
export function HistogramPanel({
  values,
  labels,
  xLabel,
  title,
  selectedIndices,
  onPointClick,
}: HistogramPanelProps): React.JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const n = values.length / 2;

  // Extract only x-values (PC1) from the interleaved array
  const getXValues = useCallback((): Float32Array => {
    const xs = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      xs[i] = values[i * 2];
    }
    return xs;
  }, [values, n]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (n === 0) return;

    const xs = getXValues();

    // Compute bounds
    let minX = Infinity, maxX = -Infinity;
    for (let i = 0; i < n; i++) {
      if (xs[i] < minX) minX = xs[i];
      if (xs[i] > maxX) maxX = xs[i];
    }
    const padX = (maxX - minX) * 0.05 || 1;
    minX -= padX;
    maxX += padX;
    const binWidth = (maxX - minX) / NUM_BINS;

    // Build bins: if we have labels, build per-cluster stacked bins
    const hasLabels = labels != null;
    // Compute the max label with a loop — spreading ~200k entries into
    // Math.max(...) overflows V8's argument limit (RangeError) on large datasets.
    let maxLabel = 0;
    if (hasLabels) {
      for (let i = 0; i < labels!.length; i++) {
        if (labels![i] > maxLabel) maxLabel = labels![i];
      }
    }
    const nClusters = hasLabels ? maxLabel + 1 : 1;
    // bins[cluster][bin]
    const bins: number[][] = Array.from({ length: nClusters }, () => new Array(NUM_BINS).fill(0));
    const selectedBins = new Array(NUM_BINS).fill(0);

    for (let i = 0; i < n; i++) {
      const bin = Math.min(Math.floor((xs[i] - minX) / binWidth), NUM_BINS - 1);
      const cluster = hasLabels ? labels![i] : 0;
      bins[cluster][bin]++;
      if (selectedIndices?.has(i)) {
        selectedBins[bin]++;
      }
    }

    // Find max total bin height
    const totals = new Array(NUM_BINS).fill(0);
    for (let b = 0; b < NUM_BINS; b++) {
      for (let c = 0; c < nClusters; c++) {
        totals[b] += bins[c][b];
      }
    }
    const maxCount = Math.max(...totals, 1);

    // Drawing margins
    const marginBottom = 24;
    const marginTop = 8;
    const marginLeft = 4;
    const marginRight = 4;
    const plotW = w - marginLeft - marginRight;
    const plotH = h - marginBottom - marginTop;

    const barW = plotW / NUM_BINS;
    const scaleY = plotH / maxCount;

    // Draw stacked bars
    for (let b = 0; b < NUM_BINS; b++) {
      const x = marginLeft + b * barW;
      let yOffset = 0;
      for (let c = 0; c < nClusters; c++) {
        const count = bins[c][b];
        if (count === 0) continue;
        const barH = count * scaleY;
        const color = CLUSTER_COLORS[c % CLUSTER_COLORS.length];
        ctx.fillStyle = `rgba(${color[0]},${color[1]},${color[2]},0.7)`;
        ctx.fillRect(x, h - marginBottom - yOffset - barH, barW - 1, barH);
        yOffset += barH;
      }

      // Overlay selected particles highlight
      if (selectedBins[b] > 0) {
        const selH = selectedBins[b] * scaleY;
        ctx.fillStyle = "rgba(59, 130, 246, 0.5)";
        ctx.fillRect(x, h - marginBottom - selH, barW - 1, selH);
      }
    }

    // Draw baseline
    ctx.strokeStyle = "rgba(113, 113, 122, 0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(marginLeft, h - marginBottom);
    ctx.lineTo(w - marginRight, h - marginBottom);
    ctx.stroke();
  }, [values, n, labels, selectedIndices, getXValues]);

  useEffect(() => {
    draw();
  }, [draw]);

  // Resize observer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;

    const ro = new ResizeObserver(() => {
      canvas.width = parent.clientWidth;
      canvas.height = parent.clientHeight;
      draw();
    });
    ro.observe(parent);
    return () => ro.disconnect();
  }, [draw]);

  // Click handler: find particle nearest to clicked x position
  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!onPointClick) return;
      const canvas = canvasRef.current!;
      const rect = canvas.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const w = canvas.width;
      const marginLeft = 4;
      const marginRight = 4;
      const plotW = w - marginLeft - marginRight;

      // Map click x to data x
      const xs = getXValues();
      let minX = Infinity, maxX = -Infinity;
      for (let i = 0; i < n; i++) {
        if (xs[i] < minX) minX = xs[i];
        if (xs[i] > maxX) maxX = xs[i];
      }
      const padX = (maxX - minX) * 0.05 || 1;
      minX -= padX;
      maxX += padX;

      const dataX = minX + ((cx - marginLeft) / plotW) * (maxX - minX);

      // Find nearest particle
      let bestDist = Infinity;
      let bestIdx = -1;
      for (let i = 0; i < n; i++) {
        const d = Math.abs(xs[i] - dataX);
        if (d < bestDist) {
          bestDist = d;
          bestIdx = i;
        }
      }

      if (bestIdx >= 0) {
        onPointClick(bestIdx, [values[bestIdx * 2], values[bestIdx * 2 + 1]]);
      }
    },
    [onPointClick, values, n, getXValues]
  );

  return (
    <div className="flex flex-col">
      <div className="mb-1 text-xs font-medium text-zinc-400">{title}</div>
      <div
        className="relative rounded-md border border-zinc-800 bg-black"
        style={{ height: 350 }}
      >
        <canvas
          ref={canvasRef}
          className="w-full h-full cursor-pointer"
          onClick={handleClick}
        />
        {/* Axis label */}
        <div className="absolute bottom-1 left-1/2 -translate-x-1/2 text-[10px] text-zinc-600">
          {xLabel}
        </div>
        <div
          className="absolute left-1 top-1/2 -translate-y-1/2 -rotate-90 text-[10px] text-zinc-600"
        >
          Count
        </div>
      </div>
    </div>
  );
}
