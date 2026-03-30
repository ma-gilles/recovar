import { useEffect, useRef, useCallback, useState } from "react";
import { clsx } from "clsx";
import type { SelectionTool } from "./SelectionToolbar";

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
  /** Selection callback (from any tool: lasso, rectangle, polygon) */
  onSelect?: (indices: number[]) => void;
  /** Point click callback */
  onPointClick?: (index: number, coords: [number, number]) => void;
  /** Currently selected indices */
  selectedIndices?: Set<number>;
  /** Panel ID for cross-linking */
  panelId: string;
  /** Active selection tool (null = no selection mode, just click) */
  activeTool?: SelectionTool | null;
}

/**
 * Canvas-based scatter plot panel with lasso, rectangle, and polygon selection.
 */
export function ScatterPanel({
  points,
  labels,
  markers,
  xLabel,
  yLabel,
  title,
  onSelect,
  onPointClick,
  selectedIndices,
  panelId: _panelId,
  activeTool = null,
}: ScatterPanelProps): React.JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  // Lasso: freehand path; Rectangle: [start, current]; Polygon: vertices
  const shapePointsRef = useRef<[number, number][]>([]);
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

    // Draw selection shape overlay
    const pts = shapePointsRef.current;
    if (pts.length > 0) {
      ctx.strokeStyle = "rgba(59, 130, 246, 0.8)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);

      if (activeTool === "lasso" && pts.length > 1) {
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) {
          ctx.lineTo(pts[i][0], pts[i][1]);
        }
        ctx.stroke();
      } else if (activeTool === "rectangle" && pts.length === 2) {
        const [start, end] = pts;
        const rx = Math.min(start[0], end[0]);
        const ry = Math.min(start[1], end[1]);
        const rw = Math.abs(end[0] - start[0]);
        const rh = Math.abs(end[1] - start[1]);
        ctx.strokeRect(rx, ry, rw, rh);
        // Semi-transparent fill
        ctx.fillStyle = "rgba(59, 130, 246, 0.08)";
        ctx.fillRect(rx, ry, rw, rh);
      } else if (activeTool === "polygon" && pts.length > 0) {
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) {
          ctx.lineTo(pts[i][0], pts[i][1]);
        }
        ctx.stroke();
        // Draw vertices as small circles
        ctx.setLineDash([]);
        for (const pt of pts) {
          ctx.beginPath();
          ctx.arc(pt[0], pt[1], 3, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(59, 130, 246, 0.9)";
          ctx.fill();
        }
      }

      ctx.setLineDash([]);
    }
  }, [points, n, labels, markers, selectedIndices, getBounds, activeTool]);

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

  // Clear shape when tool changes
  useEffect(() => {
    shapePointsRef.current = [];
    setIsDrawing(false);
  }, [activeTool]);

  /** Find all point indices inside the given polygon (canvas coords) */
  const selectPointsInPolygon = useCallback((polygon: [number, number][]): number[] => {
    const canvas = canvasRef.current!;
    const h = canvas.height;
    const t = transformRef.current;
    const selected: number[] = [];

    for (let i = 0; i < n; i++) {
      const cx = (points[i * 2] - t.offsetX) * t.scaleX;
      const cy = h - (points[i * 2 + 1] - t.offsetY) * t.scaleY;
      if (pointInPolygon(cx, cy, polygon)) {
        selected.push(i);
      }
    }
    return selected;
  }, [points, n]);

  /** Find all point indices inside the given rectangle (canvas coords) */
  const selectPointsInRect = useCallback((start: [number, number], end: [number, number]): number[] => {
    const canvas = canvasRef.current!;
    const h = canvas.height;
    const t = transformRef.current;
    const selected: number[] = [];

    const minCx = Math.min(start[0], end[0]);
    const maxCx = Math.max(start[0], end[0]);
    const minCy = Math.min(start[1], end[1]);
    const maxCy = Math.max(start[1], end[1]);

    for (let i = 0; i < n; i++) {
      const cx = (points[i * 2] - t.offsetX) * t.scaleX;
      const cy = h - (points[i * 2 + 1] - t.offsetY) * t.scaleY;
      if (cx >= minCx && cx <= maxCx && cy >= minCy && cy <= maxCy) {
        selected.push(i);
      }
    }
    return selected;
  }, [points, n]);

  // --- Lasso handlers ---
  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    // Legacy shift+drag lasso support (backwards compatible)
    if (e.shiftKey && onSelect && !activeTool) {
      setIsDrawing(true);
      const rect = canvasRef.current!.getBoundingClientRect();
      shapePointsRef.current = [[e.clientX - rect.left, e.clientY - rect.top]];
      return;
    }

    if (!activeTool || !onSelect) return;
    const rect = canvasRef.current!.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    if (activeTool === "lasso") {
      setIsDrawing(true);
      shapePointsRef.current = [[px, py]];
    } else if (activeTool === "rectangle") {
      setIsDrawing(true);
      shapePointsRef.current = [[px, py], [px, py]];
    }
    // Polygon uses click (not mousedown) — handled in handleCanvasClick
  }, [onSelect, activeTool]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    const rect = canvasRef.current!.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    const currentTool = activeTool ?? (e.shiftKey ? "lasso" : null);

    if (currentTool === "lasso") {
      shapePointsRef.current.push([px, py]);
      draw();
    } else if (currentTool === "rectangle") {
      shapePointsRef.current[1] = [px, py];
      draw();
    }
  }, [isDrawing, activeTool, draw]);

  const handleMouseUp = useCallback(() => {
    if (!isDrawing || !onSelect) {
      setIsDrawing(false);
      return;
    }

    const currentTool = activeTool ?? "lasso"; // shift+drag fallback
    setIsDrawing(false);

    if (currentTool === "lasso") {
      const pts = shapePointsRef.current;
      if (pts.length < 3) {
        shapePointsRef.current = [];
        draw();
        return;
      }
      const selected = selectPointsInPolygon(pts);
      onSelect(selected);
      shapePointsRef.current = [];
      draw();
    } else if (currentTool === "rectangle") {
      const pts = shapePointsRef.current;
      if (pts.length < 2) {
        shapePointsRef.current = [];
        draw();
        return;
      }
      const [start, end] = pts;
      // Ignore tiny drags (< 4px)
      if (Math.abs(end[0] - start[0]) < 4 && Math.abs(end[1] - start[1]) < 4) {
        shapePointsRef.current = [];
        draw();
        return;
      }
      const selected = selectPointsInRect(start, end);
      onSelect(selected);
      shapePointsRef.current = [];
      draw();
    }
  }, [isDrawing, onSelect, activeTool, draw, selectPointsInPolygon, selectPointsInRect]);

  // --- Polygon + point click handler ---
  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (isDrawing) return;

      if (activeTool === "polygon" && onSelect) {
        const rect = canvasRef.current!.getBoundingClientRect();
        const px = e.clientX - rect.left;
        const py = e.clientY - rect.top;
        shapePointsRef.current = [...shapePointsRef.current, [px, py]];
        draw();
        return;
      }

      // Default: point click
      if (!onPointClick || activeTool === "lasso" || activeTool === "rectangle") return;
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
    [onPointClick, onSelect, points, n, isDrawing, activeTool, draw]
  );

  // --- Polygon double-click to close ---
  const handleDoubleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (activeTool !== "polygon" || !onSelect) return;
      e.preventDefault();

      const pts = shapePointsRef.current;
      if (pts.length < 3) {
        shapePointsRef.current = [];
        draw();
        return;
      }

      const selected = selectPointsInPolygon(pts);
      onSelect(selected);
      shapePointsRef.current = [];
      draw();
    },
    [activeTool, onSelect, draw, selectPointsInPolygon]
  );

  const isSelecting = activeTool !== null || isDrawing;

  return (
    <div className="flex flex-col">
      <div className="mb-1 text-xs font-medium text-zinc-400">{title}</div>
      <div
        className="relative rounded-md border border-zinc-800 bg-black"
        style={{ height: 350 }}
      >
        <canvas
          ref={canvasRef}
          className={clsx("w-full h-full", isSelecting && "cursor-crosshair")}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onClick={handleCanvasClick}
          onDoubleClick={handleDoubleClick}
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
