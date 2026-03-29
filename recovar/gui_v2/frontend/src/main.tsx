import React from "react";
import ReactDOM from "react-dom/client";
import { RouterProvider, createRouter, createRootRoute, createRoute } from "@tanstack/react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "./styles/globals.css";
import { ProjectProvider } from "./lib/project-context";

import { RootLayout } from "./routes/__root";
import { DashboardPage } from "./routes/index";
import { NewJobPage } from "./routes/jobs/new";
import { JobDetailPage } from "./routes/jobs/$jobId";
import { ExplorePage } from "./routes/explore/$jobId";

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

const routeTree = rootRoute.addChildren([
  indexRoute,
  newJobRoute,
  jobDetailRoute,
  exploreRoute,
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
      retry: 1,
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
