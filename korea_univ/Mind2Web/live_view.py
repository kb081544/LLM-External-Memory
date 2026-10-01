"""Tiny local server that shows the most recently written Mind2Web screenshot as
run.py works, auto-refreshing every few seconds. Mind2Web is offline (see run.py's
module docstring) so this isn't a live browser -- it just streams "what was most
recently highlighted" to a page you can leave open while a run is in progress.

Usage:
    python live_view.py --results-dir results_shopping_memory_gemini-3.5-flash-lite results_travel_memory_gemini-3.5-flash-lite
Then open http://localhost:8765 in a browser.
"""

from __future__ import annotations

import argparse
import glob
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RESULTS_DIRS: list[Path] = []
REFRESH_SECONDS = 3


def find_latest_screenshot() -> Path | None:
    candidates: list[Path] = []
    for d in RESULTS_DIRS:
        candidates.extend(Path(p) for p in glob.glob(str(d / "mind2web.*" / "screenshot_step_*.png")))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def screenshot_id(shot: Path | None) -> str:
    if shot is None:
        return ""
    return f"{shot.parent.name}/{shot.name}:{shot.stat().st_mtime}"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # quiet
        pass

    def do_GET(self):
        if self.path.startswith("/latest.png"):
            shot = find_latest_screenshot()
            if shot is None:
                self.send_response(404)
                self.end_headers()
                return
            data = shot.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path.startswith("/status"):
            shot = find_latest_screenshot()
            sid = screenshot_id(shot)
            if shot is not None:
                info = f"{shot.parent.name} / step {shot.stem.replace('screenshot_step_', '')}"
            else:
                info = "(no screenshots yet)"
            import json
            data = json.dumps({"id": sid, "info": info}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        # static shell -- JS below polls /status and only swaps the <img> src (no
        # page reload / flicker) when the underlying screenshot has actually changed
        html = """<!doctype html><html><head><meta charset="utf-8">
<title>Mind2Web live view</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee}
header{padding:10px 16px;background:#1b1b1b;border-bottom:1px solid #333;font-family:ui-monospace,monospace;font-size:13px;display:flex;justify-content:space-between}
#dot{width:8px;height:8px;border-radius:50%;background:#3fb950;display:inline-block;margin-right:6px;transition:opacity .15s}
img{display:block;max-width:100%;margin:0 auto}
</style></head><body>
<header><span id="info">connecting...</span><span><span id="dot"></span>live</span></header>
<img id="shot" src="">
<script>
let lastId = null;
async function poll() {
  try {
    const r = await fetch('/status?t=' + Date.now());
    const s = await r.json();
    document.getElementById('info').textContent = s.info;
    if (s.id && s.id !== lastId) {
      lastId = s.id;
      document.getElementById('shot').src = '/latest.png?t=' + Date.now();
      const dot = document.getElementById('dot');
      dot.style.opacity = 0.2;
      setTimeout(() => dot.style.opacity = 1, 150);
    }
  } catch (e) {}
}
poll();
setInterval(poll, 1500);
</script>
</body></html>"""
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    RESULTS_DIRS.extend(args.results_dir)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Serving on http://localhost:{args.port} -- watching: {', '.join(str(d) for d in RESULTS_DIRS)}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
