import { useQuery } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { getSystemInfo } from "../lib/api/client";
import type { SystemInfo } from "../lib/api/client";
import { Plus, Server } from "lucide-react";

export function DashboardPage(): React.JSX.Element {
  const { data: sysInfo } = useQuery<SystemInfo>({
    queryKey: ["system-info"],
    queryFn: getSystemInfo,
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">recovar GUI</h1>
        <Link
          to="/jobs/new"
          className="inline-flex items-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-500"
        >
          <Plus className="h-4 w-4" />
          New Job
        </Link>
      </div>

      {sysInfo && (
        <div className="rounded-lg bg-zinc-900 p-4">
          <div className="flex items-center gap-2 text-sm text-zinc-400">
            <Server className="h-4 w-4" />
            <span>{sysInfo.hostname}</span>
            <span className="mx-1">|</span>
            <span>
              {sysInfo.executor_mode === "slurm" ? "Cluster mode" : "Local mode"}
            </span>
            <span className="mx-1">|</span>
            <span>recovar {sysInfo.recovar_version}</span>
            {sysInfo.gpu_count > 0 && (
              <>
                <span className="mx-1">|</span>
                <span>{sysInfo.gpu_count} GPU{sysInfo.gpu_count > 1 ? "s" : ""}</span>
              </>
            )}
          </div>
        </div>
      )}

      <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-8 text-center">
        <p className="text-zinc-400">
          Create a project or open an existing one to get started.
        </p>
        <Link
          to="/jobs/new"
          className="mt-4 inline-flex items-center gap-2 rounded-md border border-zinc-700 px-4 py-2 text-sm hover:bg-zinc-800"
        >
          <Plus className="h-4 w-4" />
          New Job
        </Link>
      </div>
    </div>
  );
}
