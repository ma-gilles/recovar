import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Search, Box, FlaskConical, Settings as SettingsIcon, Wand2, Layers } from "lucide-react";
import { useProject } from "../../lib/project-context";
import { getProject, listProjectMasks, type ProjectDetail, type MaskInfo } from "../../lib/api/client";

interface CommandItem {
  id: string;
  label: string;
  sub?: string;
  icon: React.ReactNode;
  navigate: () => void;
}

/**
 * VS Code / Linear-style spotlight palette. Open with Cmd+K (Mac) or
 * Ctrl+K. Fuzzy-substring search across the active project's jobs and
 * masks plus a few static actions (new job, dashboard, etc.).
 */
export function CommandPalette(): React.JSX.Element | null {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const { project } = useProject();

  // Global Cmd+K / Ctrl+K listener.
  useEffect(() => {
    function onKey(e: KeyboardEvent): void {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      } else if (e.key === "Escape" && open) {
        setOpen(false);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  // Reset state when opened.
  useEffect(() => {
    if (open) {
      setQuery("");
      setActiveIndex(0);
      // Focus on next tick so the input is mounted.
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [open]);

  const { data: projectDetail } = useQuery<ProjectDetail>({
    queryKey: ["project", project?.id, "for-palette"],
    queryFn: () => getProject(project!.id),
    enabled: open && !!project?.id,
  });

  const { data: masks } = useQuery<MaskInfo[]>({
    queryKey: ["project-masks", project?.id, "for-palette"],
    queryFn: () => listProjectMasks(project!.id),
    enabled: open && !!project?.id,
  });

  // Build the master item list.
  const allItems: CommandItem[] = useMemo(() => {
    const items: CommandItem[] = [];
    items.push({
      id: "go-dashboard",
      label: "Dashboard",
      sub: "Project overview",
      icon: <SettingsIcon className="h-4 w-4 text-zinc-400" />,
      navigate: () => navigate({ to: "/" }),
    });
    items.push({
      id: "new-job",
      label: "New Job",
      sub: "Submit a job",
      icon: <FlaskConical className="h-4 w-4 text-blue-400" />,
      navigate: () => navigate({ to: "/jobs/new", search: {} as never }),
    });
    if (projectDetail?.jobs) {
      for (const job of projectDetail.jobs) {
        items.push({
          id: `job:${job.id}`,
          label: `${job.type} — ${job.output_dir.split("/").slice(-2).join("/")}`,
          sub: `${job.status} · ${new Date(job.created).toLocaleDateString()}`,
          icon: <Box className="h-4 w-4 text-sky-400" />,
          navigate: () => navigate({ to: "/jobs/$jobId", params: { jobId: job.id } }),
        });
        if (job.type === "Analyze" && job.status === "completed") {
          items.push({
            id: `explore:${job.id}`,
            label: `Explore latent space — ${job.output_dir.split("/").slice(-2).join("/")}`,
            sub: "Open scatter plot for this analyze job",
            icon: <Layers className="h-4 w-4 text-emerald-400" />,
            navigate: () => navigate({ to: "/explore/$jobId", params: { jobId: job.id } }),
          });
        }
      }
    }
    if (masks) {
      for (const m of masks) {
        items.push({
          id: `mask:${m.path}`,
          label: m.name,
          sub: `Mask · ${(m.size_bytes / 1e6).toFixed(1)} MB`,
          icon: <Wand2 className="h-4 w-4 text-emerald-500" />,
          // Masks don't have a dedicated detail page yet — copy path to clipboard
          navigate: () => {
            navigator.clipboard?.writeText(m.path).catch(() => undefined);
          },
        });
      }
    }
    return items;
  }, [projectDetail, masks, navigate]);

  // Substring filter (case-insensitive).
  const filtered = useMemo(() => {
    if (!query.trim()) return allItems.slice(0, 50);
    const q = query.toLowerCase();
    return allItems
      .filter((it) => it.label.toLowerCase().includes(q) || it.sub?.toLowerCase().includes(q))
      .slice(0, 50);
  }, [allItems, query]);

  // Clamp the active index when the filter changes.
  useEffect(() => {
    if (activeIndex >= filtered.length) setActiveIndex(0);
  }, [filtered.length, activeIndex]);

  if (!open) return null;

  function runItem(item: CommandItem): void {
    item.navigate();
    setOpen(false);
  }

  return (
    <div
      className="fixed inset-0 z-[100] flex items-start justify-center bg-black/60 pt-[15vh]"
      onClick={(e) => {
        if (e.target === e.currentTarget) setOpen(false);
      }}
    >
      <div
        className="w-full max-w-xl rounded-lg border border-zinc-700 bg-zinc-900 shadow-2xl"
        role="dialog"
        aria-label="Command palette"
      >
        <div className="flex items-center gap-2 border-b border-zinc-800 px-3 py-2">
          <Search className="h-4 w-4 text-zinc-500" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "ArrowDown") {
                e.preventDefault();
                setActiveIndex((i) => Math.min(i + 1, filtered.length - 1));
              } else if (e.key === "ArrowUp") {
                e.preventDefault();
                setActiveIndex((i) => Math.max(i - 1, 0));
              } else if (e.key === "Enter") {
                e.preventDefault();
                if (filtered[activeIndex]) runItem(filtered[activeIndex]);
              }
            }}
            placeholder="Search jobs, masks, actions…"
            className="flex-1 bg-transparent text-sm text-zinc-100 placeholder-zinc-500 outline-none"
          />
          <kbd className="rounded border border-zinc-700 bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-500">
            ESC
          </kbd>
        </div>
        <ul className="max-h-[50vh] overflow-y-auto p-1" role="listbox">
          {filtered.length === 0 ? (
            <li className="px-3 py-6 text-center text-sm text-zinc-500">No matches</li>
          ) : (
            filtered.map((item, i) => (
              <li key={item.id}>
                <button
                  role="option"
                  aria-selected={i === activeIndex}
                  onClick={() => runItem(item)}
                  onMouseEnter={() => setActiveIndex(i)}
                  className={
                    "flex w-full items-center gap-3 rounded px-3 py-2 text-left text-sm " +
                    (i === activeIndex
                      ? "bg-blue-500/15 text-zinc-50"
                      : "text-zinc-300 hover:bg-zinc-800")
                  }
                >
                  <span className="shrink-0">{item.icon}</span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate">{item.label}</span>
                    {item.sub && (
                      <span className="block truncate text-xs text-zinc-500">
                        {item.sub}
                      </span>
                    )}
                  </span>
                </button>
              </li>
            ))
          )}
        </ul>
        <div className="flex items-center justify-between border-t border-zinc-800 px-3 py-1.5 text-[11px] text-zinc-500">
          <span>
            <kbd className="rounded bg-zinc-800 px-1">↑</kbd>
            <kbd className="ml-0.5 rounded bg-zinc-800 px-1">↓</kbd> to navigate
          </span>
          <span>
            <kbd className="rounded bg-zinc-800 px-1">↵</kbd> to open
          </span>
          <span>
            <kbd className="rounded bg-zinc-800 px-1">⌘K</kbd> to toggle
          </span>
        </div>
      </div>
    </div>
  );
}
