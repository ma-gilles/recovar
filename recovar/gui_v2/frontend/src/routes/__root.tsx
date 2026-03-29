import { useCallback } from "react";
import { Outlet } from "@tanstack/react-router";
import { Sidebar } from "../components/sidebar/Sidebar";
import { useProject } from "../lib/project-context";
import { AlertTriangle, X } from "lucide-react";

export function RootLayout(): React.JSX.Element {
  const { project, setProject, staleProjectMessage, dismissStaleMessage } = useProject();

  const handleProjectNotFound = useCallback(() => {
    setProject(null);
  }, [setProject]);

  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-50">
      <Sidebar
        projectId={project?.id}
        onProjectCreated={(p) => setProject(p)}
        onProjectNotFound={handleProjectNotFound}
      />
      <main className="flex-1 overflow-auto p-6">
        <div className="mx-auto max-w-[1400px]">
          {staleProjectMessage && (
            <div className="mb-4 flex items-center gap-3 rounded-lg border border-amber-600/50 bg-amber-950/50 px-4 py-3 text-sm text-amber-200">
              <AlertTriangle className="h-4 w-4 shrink-0 text-amber-500" />
              <span className="flex-1">{staleProjectMessage}</span>
              <button
                onClick={dismissStaleMessage}
                className="shrink-0 rounded p-0.5 text-amber-400 hover:bg-amber-900/50 hover:text-amber-200"
                aria-label="Dismiss"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          )}
          <Outlet />
        </div>
      </main>
    </div>
  );
}
