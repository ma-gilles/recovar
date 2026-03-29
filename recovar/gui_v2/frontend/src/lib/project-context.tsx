import { createContext, useContext, useState, useCallback, useEffect, type ReactNode } from "react";
import { getProject, ApiError } from "./api/client";

interface ProjectState {
  id: string;
  path: string;
  name: string;
}

interface ProjectContextValue {
  project: ProjectState | null;
  setProject: (project: ProjectState | null) => void;
  /** Non-null when a previously saved project was not found on the server. */
  staleProjectMessage: string | null;
  dismissStaleMessage: () => void;
}

const ProjectContext = createContext<ProjectContextValue>({
  project: null,
  setProject: () => {},
  staleProjectMessage: null,
  dismissStaleMessage: () => {},
});

const STORAGE_KEY = "recovar_active_project";

export function ProjectProvider({ children }: { children: ReactNode }): React.JSX.Element {
  const [project, setProjectState] = useState<ProjectState | null>(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? JSON.parse(stored) : null;
    } catch {
      return null;
    }
  });

  const [staleProjectMessage, setStaleProjectMessage] = useState<string | null>(null);

  const setProject = useCallback((p: ProjectState | null) => {
    setProjectState(p);
    setStaleProjectMessage(null);
    if (p) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(p));
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  const dismissStaleMessage = useCallback(() => {
    setStaleProjectMessage(null);
  }, []);

  // On mount: validate the stored project against the server.
  // If it returns 404, clear it immediately (no retry, no loop).
  useEffect(() => {
    if (!project) return;

    let cancelled = false;

    getProject(project.id).catch((err: unknown) => {
      if (cancelled) return;
      if (err instanceof ApiError && err.status === 404) {
        // Project no longer exists on the server.
        localStorage.removeItem(STORAGE_KEY);
        setProjectState(null);
        setStaleProjectMessage(
          "Previous project not found. It may have been moved or deleted."
        );
      }
      // For non-404 errors (network blip, 500, etc.) do nothing —
      // TanStack Query polling will handle recovery.
    });

    return () => {
      cancelled = true;
    };
    // Only run on mount (project.id from localStorage).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <ProjectContext.Provider value={{ project, setProject, staleProjectMessage, dismissStaleMessage }}>
      {children}
    </ProjectContext.Provider>
  );
}

export function useProject(): ProjectContextValue {
  return useContext(ProjectContext);
}
