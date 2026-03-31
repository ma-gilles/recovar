import { clsx } from "clsx";

const statusColors: Record<string, string> = {
  running: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  completed: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
  failed: "bg-red-500/20 text-red-400 border-red-500/30",
  queued: "bg-amber-500/20 text-amber-400 border-amber-500/30",
  cancelled: "bg-zinc-500/20 text-zinc-400 border-zinc-500/30",
  unknown: "bg-zinc-500/20 text-zinc-400 border-zinc-500/30",
};

interface BadgeProps {
  status: string;
  className?: string;
}

export function StatusBadge({ status, className }: BadgeProps): React.JSX.Element {
  return (
    <span
      className={clsx(
        "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium capitalize",
        statusColors[status] ?? statusColors.unknown,
        className
      )}
      aria-live="polite"
    >
      {status}
    </span>
  );
}
