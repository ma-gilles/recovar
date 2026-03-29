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
import { browseFiles, validateStar, type FileEntry } from "../../lib/api/client";
import { Spinner } from "../ui/spinner";

const typeIcons: Record<string, typeof File> = {
  star: Database,
  mrc: FileImage,
  mrcs: FileImage,
  cs: Database,
};

interface FileBrowserProps {
  initialPath: string;
  accept?: string[];
  onSelect: (path: string) => void;
  onValidation?: (result: { valid: boolean | null; n_particles?: number; box_size?: number; error?: string }) => void;
}

export function FileBrowser({ initialPath, accept, onSelect, onValidation }: FileBrowserProps): React.JSX.Element {
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

      // Validate .star files
      if (onValidation && entry.type === "star") {
        try {
          const result = await validateStar(entry.path);
          onValidation(result);
        } catch {
          onValidation({ valid: false, error: "Validation failed" });
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

      {/* File list */}
      <div className="max-h-64 overflow-y-auto">
        {isLoading && (
          <div className="flex items-center justify-center py-8">
            <Spinner label="Loading..." />
          </div>
        )}
        {error && (
          <div className="px-3 py-4 text-center text-sm text-red-400">
            Failed to browse directory
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
              onDoubleClick={() => entry.is_dir && navigateTo(entry.path)}
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
                  {(entry.size / 1e6).toFixed(1)} MB
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
