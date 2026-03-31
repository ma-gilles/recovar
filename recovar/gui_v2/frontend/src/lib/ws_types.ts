/**
 * WebSocket message types — keep in sync with backend/ws_types.py.
 */

// Server → Client
export const LOG_LINE = "log_line" as const;
export const STATUS_CHANGE = "status_change" as const;
export const PROGRESS = "progress" as const;
export const RECONNECT_SYNC = "reconnect_sync" as const;
export const PING = "ping" as const;

// Client → Server
export const PONG = "pong" as const;

export interface WsMessage<T = unknown> {
  type: string;
  data: T;
  ts: number;
}

export interface LogLineData {
  line: string;
  offset: number;
}

export interface StatusChangeData {
  old: string;
  new: string;
  error?: string;
}

export interface ProgressData {
  step: number;
  total: number;
  label: string;
}

export interface ReconnectSyncData {
  status: string;
  log_offset: number;
  log_tail: string[];
}
