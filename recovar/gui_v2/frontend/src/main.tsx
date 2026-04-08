import React from "react";
import ReactDOM from "react-dom/client";
import { RouterProvider, createRouter, createRootRoute, createRoute } from "@tanstack/react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "./styles/globals.css";
import { ProjectProvider } from "./lib/project-context";
import { ApiError } from "./lib/api/client";

import { RootLayout } from "./routes/__root";
import { DashboardPage } from "./routes/index";
import { NewJobPage } from "./routes/jobs/new";
import { JobDetailPage } from "./routes/jobs/$jobId";
import { ExplorePage } from "./routes/explore/$jobId";
import { ComparePage } from "./routes/compare";
import { MasksPage } from "./routes/masks";

// Define routes
const rootRoute = createRootRoute({ component: RootLayout });

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: DashboardPage,
});

const newJobRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/jobs/new",
  component: NewJobPage,
  validateSearch: (search: Record<string, unknown>) => ({
    type: (search.type as string) || undefined,
    result_dir: (search.result_dir as string) || undefined,
    density: (search.density as string) || undefined,
    input: (search.input as string) || undefined,
    particles: (search.particles as string) || undefined,
    params: (search.params as string) || undefined,
  }),
});

const jobDetailRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/jobs/$jobId",
  component: JobDetailPage,
});

const exploreRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/explore/$jobId",
  component: ExplorePage,
});

const compareRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/compare",
  component: ComparePage,
  validateSearch: (search: Record<string, unknown>) => ({
    jobs: (search.jobs as string) || "",
  }),
});

const masksRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/masks",
  component: MasksPage,
});

const routeTree = rootRoute.addChildren([
  indexRoute,
  newJobRoute,
  jobDetailRoute,
  exploreRoute,
  compareRoute,
  masksRoute,
]);

const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: (failureCount, error) => {
        // Never retry 4xx errors (client errors like 404 Not Found).
        // Only retry 5xx / network errors, and only once.
        if (error instanceof ApiError && error.status >= 400 && error.status < 500) {
          return false;
        }
        return failureCount < 1;
      },
    },
  },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <ProjectProvider>
        <RouterProvider router={router} />
      </ProjectProvider>
    </QueryClientProvider>
  </React.StrictMode>
);
