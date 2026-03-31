import { useCallback, useEffect, useRef, useState } from "react";
import { Outlet } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { Sidebar } from "../components/sidebar/Sidebar";
import { ErrorBoundary } from "../components/ui/error-boundary";
import { useProject } from "../lib/project-context";
import { getSystemInfo, type SystemInfo } from "../lib/api/client";
import { AlertTriangle, WifiOff, X } from "lucide-react";

/** How often to poll the health endpoint when the server appears reachable (ms). */
const HEALTH_POLL_INTERVAL_MS = 10_000;
/** Escalate from yellow to red banner after this many ms of consecutive failures. */
const RED_BANNER_THRESHOLD_MS = 5 * 60 * 1_000;

type ConnStatus = "ok" | "reconnecting" | "unreachable";

/**
 * Polls /api/system/info to detect server connectivity independently of
 * TanStack Query. Returns:
 *   "ok"           — last poll succeeded
 *   "reconnecting" — failing but < 5 min since first failure
 *   "unreachable"  — failing for >= 5 min (show red banner)
 * Auto-recovers to "ok" as soon as a poll succeeds again.
 */
function useServerHealth(): ConnStatus {
  const [status, setStatus] = useState<ConnStatus>("ok");
  const firstFailureAt = useRef<number | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function poll(): Promise<void> {
      try {
        const resp = await fetch("/api/system/info", {
          method: "GET",
          // Short timeout so a stalled server doesn't block the next poll.
          signal: AbortSignal.timeout(8_000),
        });
        if (!resp.ok) throw new Error("non-ok");
        if (!cancelled) {
          firstFailureAt.current = null;
          setStatus("ok");
        }
      } catch {
        if (cancelled) return;
        const now = Date.now();
        if (firstFailureAt.current === null) {
          firstFailureAt.current = now;
        }
        const elapsed = now - firstFailureAt.current;
        setStatus(elapsed >= RED_BANNER_THRESHOLD_MS ? "unreachable" : "reconnecting");
      }
    }

    void poll();
    const id = setInterval(() => void poll(), HEALTH_POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return status;
}

const FIVE_GB = 5 * 1024 * 1024 * 1024;

export function RootLayout(): React.JSX.Element {
  const { project, setProject, staleProjectMessage, dismissStaleMessage } = useProject();
  const [diskBannerDismissed, setDiskBannerDismissed] = useState(false);
  const connStatus = useServerHealth();

  const { data: sysInfo } = useQuery<SystemInfo>({
    queryKey: ["system-info"],
    queryFn: getSystemInfo,
  });

  const handleProjectNotFound = useCallback(() => {
    setProject(null);
  }, [setProject]);

  const showDiskWarning =
    !diskBannerDismissed &&
    sysInfo?.disk != null &&
    sysInfo.disk.free < FIVE_GB;

  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-50">
      <Sidebar
        projectId={project?.id}
        onProjectCreated={(p) => setProject(p)}
        onProjectNotFound={handleProjectNotFound}
      />
      <main className="relative z-0 flex-1 overflow-auto p-6">
        <ErrorBoundary>
          <div className="mx-auto max-w-[1400px]">
            {/* WebSocket / server connectivity banners */}
            {connStatus === "reconnecting" && (
              <div className="mb-4 flex items-center gap-3 rounded-lg border border-yellow-600/50 bg-yellow-950/50 px-4 py-3 text-sm text-yellow-200">
                <WifiOff className="h-4 w-4 shrink-0 text-yellow-400" />
                <span className="flex-1">Reconnecting to server...</span>
              </div>
            )}
            {connStatus === "unreachable" && (
              <div className="mb-4 flex items-center gap-3 rounded-lg border border-red-600/50 bg-red-950/50 px-4 py-3 text-sm text-red-200">
                <WifiOff className="h-4 w-4 shrink-0 text-red-400" />
                <span className="flex-1">Cannot reach server. Check your SSH tunnel.</span>
              </div>
            )}
            {showDiskWarning && (
              <div className="mb-4 flex items-center gap-3 rounded-lg border border-red-700 bg-red-900/30 px-4 py-3 text-sm text-red-300">
                <AlertTriangle className="h-4 w-4 shrink-0 text-red-400" />
                <span className="flex-1">
                  Less than 5 GB free on {sysInfo!.disk!.path}. Jobs may fail. Free space before submitting.
                </span>
                <button
                  onClick={() => setDiskBannerDismissed(true)}
                  className="shrink-0 rounded p-0.5 text-red-400 hover:bg-red-900/50 hover:text-red-200"
                  aria-label="Dismiss"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
            )}
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
        </ErrorBoundary>
      </main>
    </div>
  );
}
