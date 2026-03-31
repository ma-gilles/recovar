/**
 * WebSocket client for job log/status streaming.
 *
 * Handles reconnection with exponential backoff as specified in PHASE1.md.
 */

import type { WsMessage, LogLineData, StatusChangeData, ProgressData, ReconnectSyncData } from "../ws_types";
import { PONG } from "../ws_types";

export interface JobStreamCallbacks {
  onLogLine?: (data: LogLineData) => void;
  onStatusChange?: (data: StatusChangeData) => void;
  onProgress?: (data: ProgressData) => void;
  onReconnectSync?: (data: ReconnectSyncData) => void;
  onConnectionChange?: (connected: boolean) => void;
}

export class JobStream {
  private ws: WebSocket | null = null;
  private jobId: string;
  private lastOffset: number;
  private callbacks: JobStreamCallbacks;
  private reconnectAttempts = 0;
  private maxReconnectDelay = 30_000;
  private closed = false;

  constructor(jobId: string, callbacks: JobStreamCallbacks, lastOffset = 0) {
    this.jobId = jobId;
    this.callbacks = callbacks;
    this.lastOffset = lastOffset;
    this.connect();
  }

  private connect(): void {
    if (this.closed) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${window.location.host}/api/jobs/${this.jobId}/stream?last_offset=${this.lastOffset}`;

    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      this.callbacks.onConnectionChange?.(true);
    };

    this.ws.onmessage = (event: MessageEvent) => {
      try {
        const msg: WsMessage = JSON.parse(event.data);
        this.handleMessage(msg);
      } catch {
        // Ignore unparseable messages
      }
    };

    this.ws.onclose = () => {
      this.callbacks.onConnectionChange?.(false);
      if (!this.closed) {
        this.scheduleReconnect();
      }
    };

    this.ws.onerror = () => {
      // onclose will fire after onerror
    };
  }

  private handleMessage(msg: WsMessage): void {
    switch (msg.type) {
      case "log_line":
        this.lastOffset = (msg.data as LogLineData).offset;
        this.callbacks.onLogLine?.(msg.data as LogLineData);
        break;
      case "status_change":
        this.callbacks.onStatusChange?.(msg.data as StatusChangeData);
        break;
      case "progress":
        this.callbacks.onProgress?.(msg.data as ProgressData);
        break;
      case "reconnect_sync":
        this.callbacks.onReconnectSync?.(msg.data as ReconnectSyncData);
        break;
      case "ping":
        this.ws?.send(JSON.stringify({ type: PONG, data: {}, ts: Date.now() }));
        break;
    }
  }

  private scheduleReconnect(): void {
    const delay = Math.min(
      1000 * Math.pow(2, this.reconnectAttempts),
      this.maxReconnectDelay
    );
    this.reconnectAttempts++;
    setTimeout(() => this.connect(), delay);
  }

  close(): void {
    this.closed = true;
    this.ws?.close();
    this.ws = null;
  }
}
