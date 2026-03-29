import { clsx } from "clsx";

interface SpinnerProps {
  className?: string;
  label?: string;
}

export function Spinner({ className, label }: SpinnerProps): React.JSX.Element {
  return (
    <div className="flex flex-col items-center gap-2">
      <svg
        className={clsx("h-6 w-6 animate-spin text-blue-500", className)}
        viewBox="0 0 24 24"
        fill="none"
      >
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
      </svg>
      {label && <span className="text-sm text-zinc-400">{label}</span>}
    </div>
  );
}
