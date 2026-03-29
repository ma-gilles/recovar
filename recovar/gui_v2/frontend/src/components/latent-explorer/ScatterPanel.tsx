import { useEffect, useRef, useCallback, useState } from "react";
import { clsx } from "clsx";

// D3 categorical 10 colors for k-means clusters
const CLUSTER_COLORS = [
  [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
  [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
  [188, 189, 34], [23, 190, 207],
];

interface ScatterPanelProps {
  /** Interleaved xy coords: [x0,y0,x1,y1,...] */
  points: Float32Array;
  /** Per-point cluster labels (optional) */
  labels?: Int32Array | null;
  /** Special marker positions: interleaved [x0,y0,x1,y1,...] */
  markers?: Float32Array | null;
  /** Axis labels */
  xLabel: string;
  yLabel: string;
  /** Title */
  title: string;
  /** Lasso selection callback */
  onLasso?: (indices: number[]) => void;
  /** Point click callback */
  onPointClick?: (index: number, coords: [number, number]) => void;
  /** Currently selected indices */
  selectedIndices?: Set<number>;
  /** Panel ID for cross-linking */
  panelId: string;
}

/**
 * Canvas-based scatter plot panel.
 * Uses 2D canvas for broad compatibility; regl-scatterplot can be swapped in
 * when the npm package is installed.
 */
export function ScatterPanel({
  points,
  labels,
  markers,
  xLabel,
  yLabel,
  title,
  onLasso,
  onPointClick,
  selectedIndices,
  panelId: _panelId,
}: ScatterPanelProps): React.JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isLassoing, setIsLassoing] = useState(false);
  const lassoPointsRef = useRef<[number, number][]>([]);
  const transformRef = useRef({ scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 0 });

  const n = points.length / 2;

  // Compute bounds
  const getBounds = useCallback(() => {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = points[i * 2], y = points[i * 2 + 1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const padX = (maxX - minX) * 0.05 || 1;
    const padY = (maxY - minY) * 0.05 || 1;
    return { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY };
  }, [points, n]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (n === 0) return;

    const bounds = getBounds();
    const scaleX = w / (bounds.maxX - bounds.minX);
    const scaleY = h / (bounds.maxY - bounds.minY);
    transformRef.current = { scaleX, scaleY, offsetX: bounds.minX, offsetY: bounds.minY };

    const toCanvasX = (x: number) => (x - bounds.minX) * scaleX;
    const toCanvasY = (y: number) => h - (y - bounds.minY) * scaleY; // flip Y

    // Draw points
    const pointSize = n > 500000 ? 0.5 : n > 100000 ? 1 : n > 10000 ? 1.5 : 2;
    const alpha = n > 100000 ? 0.15 : n > 10000 ? 0.3 : 0.6;

    for (let i = 0; i < n; i++) {
      const x = toCanvasX(points[i * 2]);
      const y = toCanvasY(points[i * 2 + 1]);

      if (selectedIndices?.has(i)) {
        ctx.fillStyle = "rgba(59, 130, 246, 0.9)"; // blue highlight
      } else if (labels) {
        const c = CLUSTER_COLORS[labels[i] % CLUSTER_COLORS.length];
        ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
      } else {
        ctx.fillStyle = `rgba(148, 163, 184, ${alpha})`; // zinc-400
      }

      ctx.fillRect(x - pointSize / 2, y - pointSize / 2, pointSize, pointSize);
    }

    // Draw markers
    if (markers && markers.length >= 2) {
      const markerColors = ["#3b82f6", "#ef4444", "#22c55e", "#eab308"];
      for (let i = 0; i < markers.length / 2; i++) {
        const mx = toCanvasX(markers[i * 2]);
        const my = toCanvasY(markers[i * 2 + 1]);
        ctx.beginPath();
        ctx.arc(mx, my, 6, 0, Math.PI * 2);
        ctx.fillStyle = markerColors[i % markerColors.length];
        ctx.fill();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }

    // Draw lasso path
    if (lassoPointsRef.current.length > 1) {
      ctx.beginPath();
      ctx.moveTo(lassoPointsRef.current[0][0], lassoPointsRef.current[0][1]);
      for (let i = 1; i < lassoPointsRef.current.length; i++) {
        ctx.lineTo(lassoPointsRef.current[i][0], lassoPointsRef.current[i][1]);
      }
      ctx.strokeStyle = "rgba(59, 130, 246, 0.8)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [points, n, labels, markers, selectedIndices, getBounds]);

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

  // Mouse handlers for lasso and click
  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.shiftKey && onLasso) {
      setIsLassoing(true);
      const rect = canvasRef.current!.getBoundingClientRect();
      lassoPointsRef.current = [[e.clientX - rect.left, e.clientY - rect.top]];
    }
  }, [onLasso]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isLassoing) return;
    const rect = canvasRef.current!.getBoundingClientRect();
    lassoPointsRef.current.push([e.clientX - rect.left, e.clientY - rect.top]);
    draw();
  }, [isLassoing, draw]);

  const handleMouseUp = useCallback(() => {
    if (!isLassoing || !onLasso) {
      setIsLassoing(false);
      return;
    }
    setIsLassoing(false);

    const lasso = lassoPointsRef.current;
    if (lasso.length < 3) {
      lassoPointsRef.current = [];
      draw();
      return;
    }

    // Point-in-polygon test for all particles
    const canvas = canvasRef.current!;
    const h = canvas.height;
    const t = transformRef.current;
    const selected: number[] = [];

    for (let i = 0; i < n; i++) {
      const cx = (points[i * 2] - t.offsetX) * t.scaleX;
      const cy = h - (points[i * 2 + 1] - t.offsetY) * t.scaleY;
      if (pointInPolygon(cx, cy, lasso)) {
        selected.push(i);
      }
    }

    onLasso(selected);
    lassoPointsRef.current = [];
    draw();
  }, [isLassoing, onLasso, points, n, draw]);

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!onPointClick || isLassoing) return;
      const canvas = canvasRef.current!;
      const rect = canvas.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      const h = canvas.height;
      const t = transformRef.current;

      // Find nearest point
      let bestDist = 100; // max pixel distance
      let bestIdx = -1;
      for (let i = 0; i < n; i++) {
        const px = (points[i * 2] - t.offsetX) * t.scaleX;
        const py = h - (points[i * 2 + 1] - t.offsetY) * t.scaleY;
        const d = Math.sqrt((px - cx) ** 2 + (py - cy) ** 2);
        if (d < bestDist) {
          bestDist = d;
          bestIdx = i;
        }
      }

      if (bestIdx >= 0) {
        onPointClick(bestIdx, [points[bestIdx * 2], points[bestIdx * 2 + 1]]);
      }
    },
    [onPointClick, points, n, isLassoing]
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
          className={clsx("w-full h-full", isLassoing && "cursor-crosshair")}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onClick={handleClick}
        />
        {/* Axis labels */}
        <div className="absolute bottom-1 left-1/2 -translate-x-1/2 text-[10px] text-zinc-600">
          {xLabel}
        </div>
        <div
          className="absolute left-1 top-1/2 -translate-y-1/2 -rotate-90 text-[10px] text-zinc-600"
        >
          {yLabel}
        </div>
      </div>
      <div className="mt-0.5 text-[10px] text-zinc-600">
        Shift+drag to lasso select
      </div>
    </div>
  );
}

/** Ray casting point-in-polygon test */
function pointInPolygon(x: number, y: number, polygon: [number, number][]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i][0], yi = polygon[i][1];
    const xj = polygon[j][0], yj = polygon[j][1];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}
