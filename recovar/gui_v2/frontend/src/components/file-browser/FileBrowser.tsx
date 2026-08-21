import { useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Folder,
  File,
  ChevronRight,
  ArrowUp,
  Database,
  FileImage,
} from "lucide-react";
import { clsx } from "clsx";
import { ApiError, browseFiles, validateStar, type FileEntry } from "../../lib/api/client";
import { Spinner } from "../ui/spinner";

const typeIcons: Record<string, typeof File> = {
  star: Database,
  mrc: FileImage,
  mrcs: FileImage,
  cs: Database,
};

/**
 * Format a byte count using the same binary (base-1024) convention as
 * PathInput's formatSize, so both file pickers report identical sizes.
 */
function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

interface FileBrowserProps {
  initialPath: string;
  accept?: string[];
  /** If true, the browser is for selecting a directory, not a file. */
  selectDirectory?: boolean;
  onSelect: (path: string) => void;
  onValidation?: (result: { valid: boolean | null; n_particles?: number; box_size?: number; error?: string }) => void;
}

export function FileBrowser({ initialPath, accept, selectDirectory, onSelect, onValidation }: FileBrowserProps): React.JSX.Element {
  const [currentPath, setCurrentPath] = useState(initialPath);

  const { data: entries, isLoading, error } = useQuery({
    queryKey: ["browse", currentPath],
    queryFn: () => browseFiles(currentPath),
  });

  const breadcrumbs = currentPath.split("/").filter(Boolean);

  const navigateTo = useCallback((path: string) => {
    setCurrentPath(path);
  }, []);

  const goUp = useCallback(() => {
    const parent = currentPath.split("/").slice(0, -1).join("/") || "/";
    setCurrentPath(parent);
  }, [currentPath]);

  const handleClick = useCallback(
    async (entry: FileEntry) => {
      if (entry.is_dir) {
        navigateTo(entry.path);
        return;
      }
      // File selected
      onSelect(entry.path);

      // Validate .star files. For any other selected file, signal that
      // validation does not apply so the caller can clear stale info from
      // a previously selected .star (otherwise it would be misattributed
      // to the newly selected non-star file).
      if (onValidation) {
        if (entry.type === "star") {
          try {
            const result = await validateStar(entry.path);
            onValidation(result);
          } catch (err) {
            onValidation({
              valid: false,
              error: err instanceof Error ? err.message : "Validation failed",
            });
          }
        } else {
          onValidation({ valid: null });
        }
      }
    },
    [navigateTo, onSelect, onValidation]
  );

  const filteredEntries = entries?.filter((e) => {
    if (e.is_dir) return true;
    if (!accept || accept.length === 0) return true;
    return accept.some((ext) => e.name.endsWith(ext));
  });

  return (
    <div className="rounded-lg border border-zinc-700 bg-zinc-900">
      {/* Breadcrumb */}
      <div className="flex items-center gap-1 border-b border-zinc-800 px-3 py-2 text-xs text-zinc-400">
        <button onClick={() => navigateTo("/")} className="hover:text-zinc-50">
          /
        </button>
        {breadcrumbs.map((part, i) => {
          const path = "/" + breadcrumbs.slice(0, i + 1).join("/");
          return (
            <span key={path} className="flex items-center gap-1">
              <ChevronRight className="h-3 w-3" />
              <button onClick={() => navigateTo(path)} className="hover:text-zinc-50">
                {part}
              </button>
            </span>
          );
        })}
      </div>

      {/* Go up button */}
      {currentPath !== "/" && (
        <button
          onClick={goUp}
          className="flex w-full items-center gap-2 border-b border-zinc-800 px-3 py-1.5 text-sm text-zinc-400 hover:bg-zinc-800 hover:text-zinc-50"
        >
          <ArrowUp className="h-3.5 w-3.5" />
          Parent directory
        </button>
      )}

      {/* Select current directory button */}
      {selectDirectory && (
        <button
          onClick={() => onSelect(currentPath)}
          className="flex w-full items-center gap-2 border-b border-zinc-800 bg-blue-600/10 px-3 py-2 text-sm text-blue-400 hover:bg-blue-600/20"
        >
          <Folder className="h-3.5 w-3.5" />
          Select this directory
          <span className="ml-auto font-mono text-xs text-zinc-500">{currentPath}</span>
        </button>
      )}

      {/* File list */}
      <div className="max-h-64 overflow-y-auto">
        {isLoading && (
          <div className="flex items-center justify-center py-8">
            <Spinner label="Loading..." />
          </div>
        )}
        {error && (
          <div className="px-3 py-4 text-center text-sm text-red-400">
            <p>
              {error instanceof ApiError && error.status === 403
                ? "This directory is outside the allowed locations"
                : "Failed to browse directory"}
            </p>
            <p className="mt-1 break-words text-xs text-red-400/80">
              {error instanceof Error ? error.message : String(error)}
            </p>
            {currentPath !== initialPath && (
              <button
                onClick={() => navigateTo(initialPath)}
                className="mt-2 rounded border border-zinc-700 px-2 py-1 text-xs text-zinc-300 hover:bg-zinc-800 hover:text-zinc-50"
              >
                Back to start
              </button>
            )}
            <p className="mt-2 text-xs text-zinc-500">
              Use the breadcrumb or Parent directory above to navigate elsewhere.
            </p>
          </div>
        )}
        {filteredEntries?.length === 0 && !isLoading && (
          <div className="px-3 py-4 text-center text-sm text-zinc-500">
            No matching files
          </div>
        )}
        {filteredEntries?.map((entry) => {
          const Icon = entry.is_dir
            ? Folder
            : typeIcons[entry.type] ?? File;
          return (
            <button
              key={entry.path}
              onClick={() => handleClick(entry)}
              className={clsx(
                "flex w-full items-center gap-2 px-3 py-1.5 text-sm",
                "hover:bg-zinc-800",
                entry.is_dir ? "text-zinc-300" : "text-zinc-400"
              )}
            >
              <Icon className="h-4 w-4 shrink-0" />
              <span className="truncate">{entry.name}</span>
              {!entry.is_dir && (
                <span className="ml-auto shrink-0 text-xs text-zinc-600">
                  {formatSize(entry.size)}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
