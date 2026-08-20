import React from "react";

// ---------------------------------------------------------------------------
// ErrorBoundary — safety net around VtkViewer for unhandled render errors.
// Extracted to its own file so it can be imported eagerly while VtkViewer
// (and its heavy vtk.js dependency) is lazy-loaded.
// ---------------------------------------------------------------------------

interface VtkErrorBoundaryProps {
  onWebGLFail: () => void;
  children: React.ReactNode;
}

interface VtkErrorBoundaryState {
  hasError: boolean;
  message: string | null;
}

export class VtkErrorBoundary extends React.Component<VtkErrorBoundaryProps, VtkErrorBoundaryState> {
  constructor(props: VtkErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, message: null };
  }

  static getDerivedStateFromError(error: Error): VtkErrorBoundaryState {
    return { hasError: true, message: error?.message ?? null };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo): void {
    console.error("VtkErrorBoundary caught error:", error, info);
    this.props.onWebGLFail();
  }

  render(): React.ReactNode {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center rounded-lg border border-zinc-800 bg-zinc-900 p-8" style={{ minHeight: 400 }}>
          <p className="text-sm text-amber-400">
            3D rendering failed. Falling back to slice view.
            {this.state.message ? ` (${this.state.message})` : ""}
          </p>
        </div>
      );
    }
    return this.props.children;
  }
}
