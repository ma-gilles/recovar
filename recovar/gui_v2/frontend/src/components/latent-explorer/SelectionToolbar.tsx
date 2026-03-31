import { PenTool, Square, Hexagon, X } from "lucide-react";
import { clsx } from "clsx";

/** Selection tool types available in the scatter panel */
export type SelectionTool = "lasso" | "rectangle" | "polygon";

interface SelectionToolbarProps {
  activeTool: SelectionTool | null;
  onToolChange: (tool: SelectionTool | null) => void;
  onClearSelection: () => void;
  hasSelection: boolean;
  /** Live count of particles inside the in-progress selection shape (updated on mousemove). */
  liveSelectionCount?: number | null;
}

const tools: { id: SelectionTool; label: string; Icon: typeof PenTool }[] = [
  { id: "lasso", label: "Lasso", Icon: PenTool },
  { id: "rectangle", label: "Rectangle", Icon: Square },
  { id: "polygon", label: "Polygon", Icon: Hexagon },
];

/**
 * Toolbar for choosing the active selection tool on scatter plots.
 * Only one tool is active at a time (radio-button behavior).
 * Clicking the active tool deactivates it.
 */
export function SelectionToolbar({
  activeTool,
  onToolChange,
  onClearSelection,
  hasSelection,
  liveSelectionCount,
}: SelectionToolbarProps): React.JSX.Element {
  return (
    <div className="flex items-center gap-2">
      <div className="flex items-center rounded-md border border-zinc-800 bg-zinc-900 p-0.5">
        {tools.map(({ id, label, Icon }) => (
          <button
            key={id}
            type="button"
            title={label}
            aria-label={`${label} selection tool`}
            aria-pressed={activeTool === id}
            className={clsx(
              "inline-flex items-center gap-1.5 rounded px-2 py-1 text-xs transition-colors outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-1 focus-visible:ring-offset-zinc-950",
              activeTool === id
                ? "bg-blue-600 text-white"
                : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
            )}
            onClick={() => onToolChange(activeTool === id ? null : id)}
          >
            <Icon className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">{label}</span>
          </button>
        ))}
      </div>

      {hasSelection && (
        <button
          type="button"
          className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 transition-colors"
          onClick={onClearSelection}
        >
          <X className="h-3.5 w-3.5" />
          Clear Selection
        </button>
      )}

      {activeTool && (
        <span className="text-[11px] text-zinc-500">
          {activeTool === "lasso" && "Draw freehand on the scatter plot to select particles"}
          {activeTool === "rectangle" && "Click and drag to draw a rectangle selection"}
          {activeTool === "polygon" && "Click to add vertices, double-click to close and select"}
        </span>
      )}

      {liveSelectionCount != null && liveSelectionCount > 0 && (
        <span className="ml-auto rounded bg-blue-600/20 px-2 py-0.5 text-[11px] font-medium tabular-nums text-blue-300">
          {liveSelectionCount.toLocaleString()} particles
        </span>
      )}
    </div>
  );
}
