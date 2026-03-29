import { forwardRef, type LabelHTMLAttributes } from "react";
import { clsx } from "clsx";

export const Label = forwardRef<HTMLLabelElement, LabelHTMLAttributes<HTMLLabelElement>>(
  ({ className, ...props }, ref) => {
    return (
      <label
        ref={ref}
        className={clsx("text-sm font-medium text-zinc-50", className)}
        {...props}
      />
    );
  }
);
Label.displayName = "Label";
