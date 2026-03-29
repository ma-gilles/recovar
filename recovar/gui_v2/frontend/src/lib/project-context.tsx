import { createContext, useContext, useState, useCallback, type ReactNode } from "react";

interface ProjectState {
  id: string;
  path: string;
  name: string;
}

interface ProjectContextValue {
  project: ProjectState | null;
  setProject: (project: ProjectState | null) => void;
}

const ProjectContext = createContext<ProjectContextValue>({
  project: null,
  setProject: () => {},
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

  const setProject = useCallback((p: ProjectState | null) => {
    setProjectState(p);
    if (p) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(p));
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  return (
    <ProjectContext.Provider value={{ project, setProject }}>
      {children}
    </ProjectContext.Provider>
  );
}

export function useProject(): ProjectContextValue {
  return useContext(ProjectContext);
}
