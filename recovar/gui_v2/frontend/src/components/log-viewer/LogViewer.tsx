import { useEffect, useRef, useState, useCallback } from "react";
import { Download, ArrowDown } from "lucide-react";
import { clsx } from "clsx";
import { JobStream, type JobStreamCallbacks } from "../../lib/api/ws";
import type { LogLineData, StatusChangeData, ProgressData } from "../../lib/ws_types";
import { Button } from "../ui/button";

interface LogViewerProps {
  jobId: string;
  /** Current job status — used to suppress misleading "Reconnecting..." for terminal jobs */
  jobStatus?: string;
  onStatusChange?: (status: string) => void;
}

export function LogViewer({ jobId, jobStatus, onStatusChange }: LogViewerProps): React.JSX.Element {
  const [lines, setLines] = useState<string[]>([]);
  const [connected, setConnected] = useState(false);
  const [progress, setProgress] = useState<{ step: number; total: number; label: string } | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);
  const streamRef = useRef<JobStream | null>(null);

  const scrollToBottom = useCallback(() => {
    if (containerRef.current && autoScroll) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [autoScroll]);

  useEffect(() => {
    const callbacks: JobStreamCallbacks = {
      onLogLine: (data: LogLineData) => {
        setLines((prev) => [...prev, data.line]);
      },
      onStatusChange: (data: StatusChangeData) => {
        onStatusChange?.(data.new);
      },
      onProgress: (data: ProgressData) => {
        setProgress(data);
      },
      onReconnectSync: (data) => {
        setLines(data.log_tail);
        onStatusChange?.(data.status);
      },
      onConnectionChange: (isConnected: boolean) => {
        setConnected(isConnected);
      },
    };

    streamRef.current = new JobStream(jobId, callbacks);
    return () => {
      streamRef.current?.close();
      streamRef.current = null;
    };
  }, [jobId, onStatusChange]);

  useEffect(() => {
    scrollToBottom();
  }, [lines, scrollToBottom]);

  const handleScroll = useCallback(() => {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    // Auto-scroll if near bottom (within 50px)
    setAutoScroll(scrollHeight - scrollTop - clientHeight < 50);
  }, []);

  const downloadLog = useCallback(() => {
    const blob = new Blob([lines.join("\n")], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `job-${jobId}.log`;
    a.click();
    URL.revokeObjectURL(url);
  }, [lines, jobId]);

  return (
    <div className="flex flex-col rounded-lg border border-zinc-800 bg-zinc-950">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-zinc-800 px-3 py-2">
        <div className="flex items-center gap-2 text-xs text-zinc-400">
          {(() => {
            const isTerminal = jobStatus === "completed" || jobStatus === "failed" || jobStatus === "cancelled";
            if (isTerminal) {
              return (
                <>
                  <span className="h-2 w-2 rounded-full bg-zinc-500" />
                  <span>Log complete</span>
                </>
              );
            }
            return (
              <>
                <span
                  className={clsx(
                    "h-2 w-2 rounded-full",
                    connected ? "bg-emerald-500" : "bg-red-500"
                  )}
                />
                {connected ? "Connected" : "Reconnecting..."}
              </>
            );
          })()}
          {progress && (
            <span className="ml-2">
              {progress.label} ({progress.step}/{progress.total})
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {!autoScroll && (
            <Button
              variant="ghost"
              size="icon"
              onClick={() => {
                setAutoScroll(true);
                scrollToBottom();
              }}
              aria-label="Scroll to bottom"
            >
              <ArrowDown className="h-3.5 w-3.5" />
            </Button>
          )}
          <Button variant="ghost" size="icon" onClick={downloadLog} aria-label="Download log">
            <Download className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {/* Progress bar */}
      {progress && progress.total > 0 && (
        <div className="h-1 w-full bg-zinc-800">
          <div
            className="h-1 bg-blue-500 transition-all"
            style={{ width: `${(progress.step / progress.total) * 100}%` }}
          />
        </div>
      )}

      {/* Log output */}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="h-96 overflow-y-auto p-3 font-mono text-xs text-zinc-300"
      >
        {lines.length === 0 ? (
          <div className="flex h-full items-center justify-center text-zinc-500">
            Waiting for log output...
          </div>
        ) : (
          lines.map((line, i) => (
            <div
              key={i}
              className={clsx(
                "whitespace-pre-wrap py-px",
                line.toLowerCase().includes("error") && "text-red-400",
                line.toLowerCase().includes("warning") && "text-amber-400",
                line.toLowerCase().includes("completed") && "text-emerald-400"
              )}
            >
              {line}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
