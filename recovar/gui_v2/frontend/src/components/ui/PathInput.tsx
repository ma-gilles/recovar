/**
 * PathInput -- Input field with filesystem path autocomplete.
 *
 * As the user types a path, this component debounces requests to
 * `/api/files/browse?path=<parent_dir>` and shows matching entries
 * in a dropdown.  Clicking a result fills the input.
 *
 * Drop-in replacement for <Input> on any path field.
 */

import {
  useState,
  useRef,
  useEffect,
  useCallback,
  forwardRef,
  type InputHTMLAttributes,
} from "react";
import { clsx } from "clsx";
import { browseFiles, type FileEntry } from "../../lib/api/client";

interface PathInputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, "onChange" | "accept"> {
  /** Current value (controlled). */
  value: string;
  /** Called with the new path string. */
  onChange: (value: string) => void;
  /** If true, only show directories in suggestions. */
  directoryOnly?: boolean;
  /** File extensions to filter (e.g., [".star", ".mrc"]). If empty, show all. */
  accept?: string[];
}

/** Debounce delay for autocomplete requests (ms). */
const DEBOUNCE_MS = 300;

/**
 * Extract the parent directory and typed prefix from a path string.
 * For "/scratch/gpfs/foo/ba" -> { parent: "/scratch/gpfs/foo", prefix: "ba" }
 * For "/scratch/gpfs/foo/"   -> { parent: "/scratch/gpfs/foo", prefix: "" }
 */
function splitPath(path: string): { parent: string; prefix: string } {
  if (!path || path === "/") return { parent: "/", prefix: "" };
  const lastSlash = path.lastIndexOf("/");
  if (lastSlash < 0) return { parent: "/", prefix: path };
  const parent = path.slice(0, lastSlash) || "/";
  const prefix = path.slice(lastSlash + 1);
  return { parent, prefix };
}

export const PathInput = forwardRef<HTMLInputElement, PathInputProps>(
  ({ value, onChange, directoryOnly, accept, className, ...props }, ref) => {
    const [suggestions, setSuggestions] = useState<FileEntry[]>([]);
    const [showDropdown, setShowDropdown] = useState(false);
    const [highlightIdx, setHighlightIdx] = useState(-1);
    const [loading, setLoading] = useState(false);
    const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    // Track the last browsed parent to avoid redundant fetches.
    const lastParentRef = useRef<string>("");

    const fetchSuggestions = useCallback(
      async (rawPath: string) => {
        const { parent, prefix } = splitPath(rawPath);
        // Skip if parent hasn't changed and we already have results
        if (parent === lastParentRef.current && suggestions.length > 0) {
          // Just re-filter existing suggestions
          const filtered = filterEntries(suggestions, prefix, directoryOnly, accept);
          setSuggestions(filtered.length > 0 ? suggestions : []);
          setShowDropdown(filtered.length > 0);
          return;
        }

        lastParentRef.current = parent;
        setLoading(true);
        try {
          const entries = await browseFiles(parent);
          setSuggestions(entries);
          const filtered = filterEntries(entries, prefix, directoryOnly, accept);
          setShowDropdown(filtered.length > 0);
          setHighlightIdx(-1);
        } catch {
          // Directory doesn't exist or access denied -- hide dropdown
          setSuggestions([]);
          setShowDropdown(false);
        } finally {
          setLoading(false);
        }
      },
      // eslint-disable-next-line react-hooks/exhaustive-deps
      [directoryOnly, accept]
    );

    const handleChange = useCallback(
      (e: React.ChangeEvent<HTMLInputElement>) => {
        const newVal = e.target.value;
        onChange(newVal);

        // Debounce the browse request
        if (timerRef.current) clearTimeout(timerRef.current);
        if (newVal.length < 2) {
          setShowDropdown(false);
          return;
        }
        timerRef.current = setTimeout(() => fetchSuggestions(newVal), DEBOUNCE_MS);
      },
      [onChange, fetchSuggestions]
    );

    const selectEntry = useCallback(
      (entry: FileEntry) => {
        const newVal = entry.is_dir ? entry.path + "/" : entry.path;
        onChange(newVal);
        setShowDropdown(false);
        lastParentRef.current = "";
        // If directory was selected, immediately fetch its contents
        if (entry.is_dir) {
          setTimeout(() => fetchSuggestions(newVal), 50);
        }
      },
      [onChange, fetchSuggestions]
    );

    const handleKeyDown = useCallback(
      (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (!showDropdown) return;
        const { prefix } = splitPath(value);
        const filtered = filterEntries(suggestions, prefix, directoryOnly, accept);

        if (e.key === "ArrowDown") {
          e.preventDefault();
          setHighlightIdx((prev) => Math.min(prev + 1, filtered.length - 1));
        } else if (e.key === "ArrowUp") {
          e.preventDefault();
          setHighlightIdx((prev) => Math.max(prev - 1, 0));
        } else if (e.key === "Enter" && highlightIdx >= 0 && highlightIdx < filtered.length) {
          e.preventDefault();
          selectEntry(filtered[highlightIdx]);
        } else if (e.key === "Escape") {
          setShowDropdown(false);
        } else if (e.key === "Tab" && highlightIdx >= 0 && highlightIdx < filtered.length) {
          e.preventDefault();
          selectEntry(filtered[highlightIdx]);
        }
      },
      [showDropdown, value, suggestions, directoryOnly, accept, highlightIdx, selectEntry]
    );

    // Close dropdown when clicking outside
    useEffect(() => {
      function handleClickOutside(e: MouseEvent): void {
        if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
          setShowDropdown(false);
        }
      }
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }, []);

    // Cleanup timer
    useEffect(() => {
      return () => {
        if (timerRef.current) clearTimeout(timerRef.current);
      };
    }, []);

    const { prefix } = splitPath(value);
    const filtered = filterEntries(suggestions, prefix, directoryOnly, accept);
    // Cap display at 15 items
    const displayed = filtered.slice(0, 15);

    return (
      <div ref={containerRef} className="relative">
        <input
          ref={ref}
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          onFocus={() => {
            if (value.length >= 2 && filtered.length > 0) {
              setShowDropdown(true);
            }
          }}
          className={clsx(
            "w-full rounded-md border border-zinc-600 bg-zinc-800 px-3 py-2 text-sm text-zinc-50",
            "placeholder:text-zinc-500",
            "focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500",
            "disabled:opacity-50 disabled:cursor-not-allowed",
            className
          )}
          autoComplete="off"
          {...props}
        />
        {showDropdown && displayed.length > 0 && (
          <div className="absolute z-50 mt-1 max-h-60 w-full overflow-auto rounded-md border border-zinc-700 bg-zinc-900 shadow-lg">
            {loading && (
              <div className="px-3 py-1.5 text-xs text-zinc-500">Loading...</div>
            )}
            {displayed.map((entry, idx) => (
              <button
                key={entry.path}
                type="button"
                className={clsx(
                  "flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm hover:bg-zinc-800",
                  idx === highlightIdx && "bg-zinc-800"
                )}
                onMouseDown={(e) => {
                  e.preventDefault(); // Prevent blur before click registers
                  selectEntry(entry);
                }}
                onMouseEnter={() => setHighlightIdx(idx)}
              >
                <span className={clsx(
                  "shrink-0 text-xs",
                  entry.is_dir ? "text-blue-400" : "text-zinc-500"
                )}>
                  {entry.is_dir ? "DIR" : extLabel(entry.name)}
                </span>
                <span className="truncate font-mono text-zinc-300">
                  {entry.name}{entry.is_dir ? "/" : ""}
                </span>
                {!entry.is_dir && entry.size > 0 && (
                  <span className="ml-auto shrink-0 text-xs text-zinc-600">
                    {formatSize(entry.size)}
                  </span>
                )}
              </button>
            ))}
            {filtered.length > 15 && (
              <div className="px-3 py-1 text-xs text-zinc-600">
                ...and {filtered.length - 15} more
              </div>
            )}
          </div>
        )}
      </div>
    );
  }
);
PathInput.displayName = "PathInput";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function filterEntries(
  entries: FileEntry[],
  prefix: string,
  directoryOnly?: boolean,
  accept?: string[]
): FileEntry[] {
  let filtered = entries;
  if (directoryOnly) {
    filtered = filtered.filter((e) => e.is_dir);
  }
  if (accept && accept.length > 0) {
    filtered = filtered.filter(
      (e) => e.is_dir || accept.some((ext) => e.name.toLowerCase().endsWith(ext.toLowerCase()))
    );
  }
  if (prefix) {
    const lowerPrefix = prefix.toLowerCase();
    filtered = filtered.filter((e) => e.name.toLowerCase().startsWith(lowerPrefix));
  }
  return filtered;
}

function extLabel(name: string): string {
  const dot = name.lastIndexOf(".");
  if (dot < 0) return "";
  return name.slice(dot + 1).toUpperCase();
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}
