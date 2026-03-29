import { useState } from "react";
import { HelpCircle } from "lucide-react";

interface TooltipIconProps {
  text: string;
}

export function TooltipIcon({ text }: TooltipIconProps): React.JSX.Element {
  const [show, setShow] = useState(false);

  return (
    <span className="relative inline-block">
      <button
        type="button"
        className="text-zinc-500 hover:text-zinc-300 focus:outline-none"
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        onFocus={() => setShow(true)}
        onBlur={() => setShow(false)}
        aria-label="Help"
      >
        <HelpCircle className="h-3.5 w-3.5" />
      </button>
      {show && (
        <div className="absolute bottom-full left-1/2 z-50 mb-2 w-64 -translate-x-1/2 rounded-md bg-zinc-800 p-2 text-xs text-zinc-300 shadow-lg">
          {text}
          <div className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-zinc-800" />
        </div>
      )}
    </span>
  );
}
