"""
Node-level structured logger for AgentMesh.

Features:
- Node start/end/error logging
- LLM call logging (prompt, response, usage)
- Keeps in-memory trace buffer
- Optional SQLite persistence (default: off)
"""

import json
import time
import threading
import sqlite3
from datetime import datetime


class NodeLogger:
    """
    Structured debug logger for AgentMesh nodes.

    Args:
        enabled (bool): Enable logging.
        keep_trace (bool): Maintain in-memory trace list.
        dump_to_sqlite (bool): Persist logs to SQLite.
        sqlite_path (str): Path to SQLite DB file.
    """

    def __init__(
        self,
        enabled=True,
        keep_trace=True,
        dump_to_sqlite=False,
        sqlite_path="agentmesh_traces.db",
    ):
        self.enabled = enabled
        self.keep_trace = keep_trace
        self.dump_to_sqlite = dump_to_sqlite
        self.sqlite_path = sqlite_path

        self.trace_buffer = [] if keep_trace else None
        self.lock = threading.Lock()

        self._sqlite_conn = None
        if self.dump_to_sqlite:
            try:
                self._sqlite_conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
                self._init_sqlite_table()
            except Exception as e:
                print(f"[AgentMesh][logger] SQLite init error: {e}")

    # ------------------------------------------------------------------
    # SQLite
    # ------------------------------------------------------------------
    def _init_sqlite_table(self):
        cur = self._sqlite_conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                event_type TEXT,
                node TEXT,
                payload TEXT
            );
            """
        )
        self._sqlite_conn.commit()

    def _write_sqlite(self, entry: dict):
        if not self._sqlite_conn:
            return
        try:
            cur = self._sqlite_conn.cursor()
            ts = datetime.utcnow().isoformat() + "Z"
            event_type = entry.get("event")
            node = entry.get("node")
            payload = json.dumps(entry, ensure_ascii=False)
            cur.execute(
                "INSERT INTO traces (ts, event_type, node, payload) VALUES (?, ?, ?, ?)",
                (ts, event_type, node, payload),
            )
            self._sqlite_conn.commit()
        except Exception as e:
            print(f"[AgentMesh][logger] SQLite write error: {e}")

    # ------------------------------------------------------------------
    # Internal trace handling
    # ------------------------------------------------------------------
    def _add_trace(self, entry: dict):
        if self.keep_trace:
            with self.lock:
                self.trace_buffer.append(entry)

        if self.dump_to_sqlite:
            try:
                self._write_sqlite(entry)
            except Exception:
                pass

    def capture(self, event):
        """
        Capture a graph event into the trace buffer.
        Newer versions of AgentMesh expect this API.
        """
        if not self.enabled:
            return
        self._add_trace(event)


    def last_event(self):
        if not self.keep_trace or not self.trace_buffer:
            return {}
        return self.trace_buffer[-1]


    # ------------------------------------------------------------------
    # LLM logging
    # ------------------------------------------------------------------
    def log_llm_call(self, node_name: str, prompt: str, response: str, usage=None):
        if not self.enabled:
            return

        entry = {
            "event": "llm_call",
            "node": node_name,
            "prompt": prompt,
            "response": response,
            "usage": usage,
        }

        print(f"[AgentMesh][llm][{node_name}] {json.dumps(entry, indent=2)}")
        self._add_trace(entry)

    # ------------------------------------------------------------------
    # Node wrapper
    # ------------------------------------------------------------------
    def wrap(self, name: str, fn):
        def wrapped(state, *args, **kwargs):
            if not self.enabled:
                return fn(state, *args, **kwargs)

            start = time.time()

            entry_start = {
                "event": "node_start",
                "node": name,
                "state_in": state,
                "ts": start,
            }
            print(f"[AgentMesh][{name}][start] {json.dumps(entry_start, indent=2)}")
            self._add_trace(entry_start)

            try:
                result = fn(state, *args, **kwargs)
                duration = time.time() - start

                entry_end = {
                    "event": "node_end",
                    "node": name,
                    "state_out": result,
                    "duration_s": f"{duration:.3f} s",
                }
                print(f"[AgentMesh][{name}][end] {json.dumps(entry_end, indent=2)}")
                self._add_trace(entry_end)
                return result

            except Exception as e:
                duration = time.time() - start
                entry_err = {
                    "event": "node_error",
                    "node": name,
                    "error": str(e),
                    "duration_s": f"{duration:.3f} s",
                }
                print(f"[AgentMesh][{name}][error] {json.dumps(entry_err, indent=2)}")
                self._add_trace(entry_err)
                raise

        return wrapped
    
    def wrap_async(self, name: str, fn):
        """Asynchronous wrapper for async nodes."""
        async def wrapped(state, *args, **kwargs):
            if not self.enabled:
                return await fn(state, *args, **kwargs)

            start = time.time()

            entry_start = {
                "event": "node_start",
                "node": name,
                "state_in": state,
                "ts": start,
            }
            print(f"[AgentMesh][{name}][start] {json.dumps(entry_start, indent=2)}")
            self._add_trace(entry_start)

            try:
                result = await fn(state, *args, **kwargs)
                duration = time.time() - start

                entry_end = {
                    "event": "node_end",
                    "node": name,
                    "state_out": result,
                    "duration_s": f"{duration:.3f} s",
                }
                print(f"[AgentMesh][{name}][end] {json.dumps(entry_end, indent=2)}")
                self._add_trace(entry_end)
                return result

            except Exception as e:
                duration = time.time() - start
                entry_err = {
                    "event": "node_error",
                    "node": name,
                    "error": str(e),
                    "duration_s": f"{duration:.3f} s",
                }
                print(f"[AgentMesh][{name}][error] {json.dumps(entry_err, indent=2)}")
                self._add_trace(entry_err)
                raise

        return wrapped
    
    def export_chrome_trace(self, path: str = "agentmesh_trace.json"):
        """
        Export the in-memory trace buffer into Chrome Tracing JSON format.
        View via chrome://tracing or https://ui.perfetto.dev
        """
        if not self.keep_trace:
            raise RuntimeError("Trace buffer disabled (keep_trace=False).")

        events = []
        pid = 1
        tid = 1

        # IMPORTANT FIX: use self.trace_buffer
        for entry in self.trace_buffer:
            ts = entry.get("ts", None)
            if ts is None:
                # For events without timestamp (e.g. llm_call), generate one
                ts = time.time()

            ts_us = int(ts * 1_000_000)   # microseconds

            etype = entry.get("event")

            # Node start
            if etype == "node_start":
                events.append({
                    "name": entry["node"],
                    "cat": "node",
                    "ph": "B",
                    "ts": ts_us,
                    "pid": pid,
                    "tid": tid,
                    "args": entry.get("state_in", {})
                })

            # Node end
            elif etype == "node_end":
                events.append({
                    "name": entry["node"],
                    "cat": "node",
                    "ph": "E",
                    "ts": ts_us,
                    "pid": pid,
                    "tid": tid,
                    "args": entry.get("state_out", {})
                })

            # LLM call (instant marker)
            elif etype == "llm_call":
                events.append({
                    "name": f"llm_{entry['node']}",
                    "cat": "llm",
                    "ph": "i",
                    "ts": ts_us,
                    "pid": pid,
                    "tid": tid,
                    "s": "t",
                    "args": {
                        "prompt": entry.get("prompt"),
                        "response": entry.get("response"),
                        "usage": entry.get("usage", {})
                    }
                })

        # Wrap container for Chrome Trace Viewer
        out = {"traceEvents": events}

        with open(path, "w") as f:
            json.dump(out, f, indent=2)

        return path
