import { useCallback, useEffect, useRef, useState } from "react";
import { Play, Pause, SkipBack, SkipForward } from "lucide-react";
import type { VolumeEntry } from "../../lib/api/client";

interface TrajectoryPlayerProps {
  /** Ordered trajectory volumes (sorted by numeric suffix). */
  volumes: VolumeEntry[];
  /** Callback when the displayed frame changes. Receives the volume path. */
  onFrameChange: (path: string) => void;
  /** Path of the currently displayed volume. */
  currentFrame: string;
}

/** Default playback speed in frames per second. */
const DEFAULT_FPS = 2;
const MIN_FPS = 0.5;
const MAX_FPS = 10;
const FPS_STEP = 0.5;

/**
 * TrajectoryPlayer — playback controls for trajectory volume animations.
 *
 * Renders a play/pause button, frame scrubber, frame counter, and speed
 * controls. Manages the animation timer internally and calls `onFrameChange`
 * to request each new frame.
 */
export function TrajectoryPlayer({
  volumes,
  onFrameChange,
  currentFrame,
}: TrajectoryPlayerProps): React.JSX.Element {
  const [playing, setPlaying] = useState(false);
  const [fps, setFps] = useState(DEFAULT_FPS);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Derive the current frame index from the currentFrame path
  const currentIndex = volumes.findIndex((v) => v.path === currentFrame);
  const frameIndex = currentIndex >= 0 ? currentIndex : 0;
  const totalFrames = volumes.length;

  // Keep a ref so the interval callback always sees the latest index
  const frameIndexRef = useRef(frameIndex);
  frameIndexRef.current = frameIndex;

  // Start/stop the interval timer
  useEffect(() => {
    if (playing) {
      timerRef.current = setInterval(() => {
        const nextIndex = (frameIndexRef.current + 1) % volumes.length;
        onFrameChange(volumes[nextIndex].path);
      }, 1000 / fps);
    }
    return () => {
      if (timerRef.current !== null) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [playing, fps, volumes, onFrameChange]);

  const togglePlay = useCallback(() => {
    setPlaying((prev) => !prev);
  }, []);

  const goToFrame = useCallback(
    (index: number) => {
      const clamped = Math.max(0, Math.min(index, totalFrames - 1));
      onFrameChange(volumes[clamped].path);
    },
    [volumes, totalFrames, onFrameChange]
  );

  const stepBack = useCallback(() => {
    setPlaying(false);
    goToFrame(frameIndex - 1 >= 0 ? frameIndex - 1 : totalFrames - 1);
  }, [frameIndex, totalFrames, goToFrame]);

  const stepForward = useCallback(() => {
    setPlaying(false);
    goToFrame((frameIndex + 1) % totalFrames);
  }, [frameIndex, totalFrames, goToFrame]);

  const handleSliderChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const idx = parseInt(e.target.value, 10);
      goToFrame(idx);
    },
    [goToFrame]
  );

  const handleFpsChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setFps(parseFloat(e.target.value));
    },
    []
  );

  return (
    <div className="flex items-center gap-3 rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2">
      {/* Step back */}
      <button
        onClick={stepBack}
        className="text-zinc-400 hover:text-zinc-200"
        aria-label="Previous frame"
      >
        <SkipBack className="h-3.5 w-3.5" />
      </button>

      {/* Play / Pause */}
      <button
        onClick={togglePlay}
        className="flex h-7 w-7 items-center justify-center rounded-full bg-blue-600 text-white hover:bg-blue-500"
        aria-label={playing ? "Pause" : "Play"}
      >
        {playing ? (
          <Pause className="h-3.5 w-3.5" />
        ) : (
          <Play className="h-3.5 w-3.5 ml-0.5" />
        )}
      </button>

      {/* Step forward */}
      <button
        onClick={stepForward}
        className="text-zinc-400 hover:text-zinc-200"
        aria-label="Next frame"
      >
        <SkipForward className="h-3.5 w-3.5" />
      </button>

      {/* Frame scrubber */}
      <input
        type="range"
        min={0}
        max={totalFrames - 1}
        value={frameIndex}
        onChange={handleSliderChange}
        className="flex-1"
        aria-label="Frame scrubber"
      />

      {/* Frame counter */}
      <span className="min-w-[4rem] text-center text-xs text-zinc-300">
        {frameIndex + 1} / {totalFrames}
      </span>

      {/* Speed control */}
      <div className="flex items-center gap-1 border-l border-zinc-700 pl-3">
        <span className="text-xs text-zinc-500">Speed</span>
        <input
          type="range"
          min={MIN_FPS}
          max={MAX_FPS}
          step={FPS_STEP}
          value={fps}
          onChange={handleFpsChange}
          className="w-16"
          aria-label="Playback speed"
        />
        <span className="text-xs text-zinc-400 w-10">{fps.toFixed(1)} fps</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Trajectory detection helpers
// ---------------------------------------------------------------------------

/**
 * Given a list of volume entries, returns the trajectory volumes sorted
 * by their numeric suffix. A trajectory is a set of volumes with
 * `category === "trajectory"`.
 *
 * Returns an empty array if there are fewer than 2 trajectory volumes.
 */
export function getTrajectoryVolumes(volumes: VolumeEntry[]): VolumeEntry[] {
  const trajectoryVols = volumes.filter((v) => v.category === "trajectory");
  if (trajectoryVols.length < 2) return [];

  // Sort by numeric suffix: trajectory_000, trajectory_001, ...
  return trajectoryVols.sort((a, b) => {
    const numA = extractTrailingNumber(a.name);
    const numB = extractTrailingNumber(b.name);
    if (numA !== null && numB !== null) return numA - numB;
    return a.name.localeCompare(b.name);
  });
}

/**
 * Extract the trailing number from a filename like "trajectory_003.mrc" => 3.
 * Returns null if no number is found.
 */
function extractTrailingNumber(name: string): number | null {
  const match = /(\d+)\.\w+$/.exec(name);
  if (!match) return null;
  return parseInt(match[1], 10);
}

/**
 * Returns true if `path` is one of the trajectory volumes.
 */
export function isTrajectoryVolume(
  path: string,
  trajectoryVolumes: VolumeEntry[]
): boolean {
  return trajectoryVolumes.some((v) => v.path === path);
}
