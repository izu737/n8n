"""Local web app: drop in a video, watch it transcribe, click any word to jump there.

Standard library only. Binds to localhost by default and processes one job at a time,
because transcription already uses every CPU core.
"""

from __future__ import annotations

import json
import mimetypes
import queue
import re
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .formats import RENDERERS, to_json
from .pipeline import Options, transcribe

ID_RE = re.compile(r"^[0-9a-f]{32}$")
ALLOWED_EXPORTS = {"txt": "text/plain", "md": "text/markdown", "srt": "application/x-subrip",
                   "vtt": "text/vtt", "json": "application/json"}


@dataclass
class Job:
    id: str
    name: str
    path: Path
    options: Options
    state: str = "queued"  # queued | running | done | error
    stage: str = "queued"
    progress: float = 0.0
    message: str = "Waiting"
    error: str = ""
    transcript: object = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class App:
    def __init__(self, workdir: Optional[Path] = None):
        self.workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="verbatim_"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.queue: "queue.Queue[Job]" = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self) -> None:
        while True:
            job = self.queue.get()
            job.state = "running"

            def report(stage: str, frac: float, msg: str, job=job) -> None:
                job.stage, job.progress, job.message = stage, frac, msg

            try:
                job.transcript = transcribe(job.path, job.options, report)
                job.state, job.progress, job.message = "done", 1.0, "Done"
            except Exception as exc:  # shown to the user, not swallowed
                job.state, job.error, job.message = "error", str(exc), "Failed"

    def submit(self, name: str, data_stream, length: int, options: Options) -> Job:
        jid = uuid.uuid4().hex
        suffix = Path(name).suffix[:12] if re.fullmatch(r"\.[A-Za-z0-9]{1,10}", Path(name).suffix) else ""
        folder = self.workdir / jid
        folder.mkdir()
        path = folder / f"media{suffix}"
        remaining = length
        with open(path, "wb") as fh:
            while remaining > 0:
                block = data_stream.read(min(1 << 20, remaining))
                if not block:
                    break
                fh.write(block)
                remaining -= len(block)
        if remaining > 0:
            shutil.rmtree(folder, ignore_errors=True)
            raise ValueError("Upload was cut short")
        job = Job(id=jid, name=name or "upload", path=path, options=options)
        self.jobs[jid] = job
        self.queue.put(job)
        return job


def _make_handler(app: App, allowed_hosts: set[str]):
    index_html = resources.files("verbatim").joinpath("web/index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "verbatim"

        def log_message(self, fmt, *args):  # quiet
            pass

        # -- helpers --
        def _json(self, obj, status=HTTPStatus.OK):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _host_ok(self) -> bool:
            # Defends against DNS rebinding: only answer to the names we are bound to.
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            return host in allowed_hosts

        def _job(self, jid: str) -> Optional[Job]:
            return app.jobs.get(jid) if ID_RE.match(jid) else None

        # -- routes --
        def do_GET(self):
            if not self._host_ok():
                return self._json({"error": "bad host"}, HTTPStatus.FORBIDDEN)
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            if parts == ["favicon.ico"]:
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            if not parts:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(index_html)))
                self.end_headers()
                self.wfile.write(index_html)
                return
            if len(parts) >= 3 and parts[:2] == ["api", "jobs"]:
                job = self._job(parts[2])
                if not job:
                    return self._json({"error": "unknown job"}, HTTPStatus.NOT_FOUND)
                tail = parts[3] if len(parts) > 3 else ""
                if tail == "":
                    data = {"id": job.id, "name": job.name, "state": job.state, "stage": job.stage,
                            "progress": job.progress, "message": job.message, "error": job.error}
                    if job.state == "done":
                        data["transcript"] = json.loads(to_json(job.transcript))
                    return self._json(data)
                if tail == "media":
                    return self._serve_media(job)
                if tail == "export" and job.state == "done":
                    fmt = parse_qs(url.query).get("fmt", ["txt"])[0]
                    if fmt not in ALLOWED_EXPORTS:
                        return self._json({"error": "bad format"}, HTTPStatus.BAD_REQUEST)
                    body = RENDERERS[fmt](job.transcript).encode("utf-8")
                    stem = re.sub(r"[^\w.\- ]", "_", Path(job.name).stem) or "transcript"
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", ALLOWED_EXPORTS[fmt] + "; charset=utf-8")
                    self.send_header("Content-Disposition", f'attachment; filename="{stem}.{fmt}"')
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            if not self._host_ok():
                return self._json({"error": "bad host"}, HTTPStatus.FORBIDDEN)
            # A custom header cannot be set by a cross site HTML form, so this blocks CSRF uploads.
            if self.headers.get("X-Verbatim") != "1":
                return self._json({"error": "missing header"}, HTTPStatus.FORBIDDEN)
            url = urlparse(self.path)
            if url.path != "/api/jobs":
                return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            q = parse_qs(url.query)
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0:
                return self._json({"error": "empty upload"}, HTTPStatus.BAD_REQUEST)
            speakers = q.get("speakers", [""])[0]
            glossary_text = q.get("glossary", [""])[0]
            opts = Options(
                language=q.get("language", ["en"])[0] or "en",
                diarize=q.get("diarize", ["1"])[0] != "0",
                speakers=int(speakers) if speakers.isdigit() and int(speakers) > 0 else None,
            )
            try:
                job = app.submit(q.get("name", ["upload"])[0], self.rfile, length, opts)
            except ValueError as exc:
                return self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            if glossary_text.strip():
                gpath = job.path.parent / "glossary.txt"
                gpath.write_text(glossary_text[:20000], encoding="utf-8")
                job.options.glossary = gpath
            self._json({"id": job.id}, HTTPStatus.ACCEPTED)

        def _serve_media(self, job: Job):
            path = job.path
            size = path.stat().st_size
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            start, end = 0, size - 1
            status = HTTPStatus.OK
            rng = self.headers.get("Range")
            if rng:
                m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
                if not m or (not m.group(1) and not m.group(2)):
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:  # suffix range: last N bytes
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                if start > end or start >= size:
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                status = HTTPStatus.PARTIAL_CONTENT
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with open(path, "rb") as fh:
                    fh.seek(start)
                    left = end - start + 1
                    while left > 0:
                        block = fh.read(min(1 << 20, left))
                        if not block:
                            break
                        self.wfile.write(block)
                        left -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser closed the connection while seeking

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    app = App()
    allowed = {"localhost", "127.0.0.1", "::1", host}
    httpd = ThreadingHTTPServer((host, port), _make_handler(app, allowed))
    print(f"verbatim is running at http://{host}:{port}  (Ctrl+C to stop)")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("Warning: this address is reachable by other machines and has no login.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        shutil.rmtree(app.workdir, ignore_errors=True)
