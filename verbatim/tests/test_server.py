"""HTTP layer tests. A finished job is injected so no model is needed."""

import http.client
import json
import threading
import uuid
from http.server import ThreadingHTTPServer

import pytest

from verbatim.assemble import build_segments
from verbatim.pipeline import Options
from verbatim.server import App, Job, _make_handler
from verbatim.types import Transcript, Word

MEDIA = bytes(range(256)) * 40  # 10240 bytes


@pytest.fixture
def server(tmp_path):
    app = App(tmp_path / "work")
    words = [Word("Hello", 0.0, 0.4, 1.0, 0), Word("world.", 0.5, 1.0, 0.5, 0)]
    transcript = Transcript(words=words, segments=build_segments(words), duration=2.0, language="en",
                            engine="test", speaker_count=1, timings={"total": 1.0}, source="clip.mp4")
    jid = uuid.uuid4().hex
    folder = app.workdir / jid
    folder.mkdir()
    (folder / "media.mp4").write_bytes(MEDIA)
    app.jobs[jid] = Job(id=jid, name="clip.mp4", path=folder / "media.mp4", options=Options(),
                        state="done", transcript=transcript)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(app, {"127.0.0.1", "localhost"}))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1], jid
    httpd.shutdown()
    httpd.server_close()


def req(port, method, path, headers=None, body=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse()
    data = r.read()
    return r.status, dict(r.getheaders()), data


def test_index_page_served(server):
    status, headers, body = req(server[0], "GET", "/")
    assert status == 200 and b"Verbatim" in body and headers["Content-Type"].startswith("text/html")


def test_job_status_includes_transcript_when_done(server):
    port, jid = server
    status, _, body = req(port, "GET", f"/api/jobs/{jid}")
    data = json.loads(body)
    assert status == 200 and data["state"] == "done"
    assert data["transcript"]["segments"][0]["words"][1]["confidence"] == 0.5


def test_unknown_or_malformed_job_ids_are_404(server):
    port, _ = server
    assert req(port, "GET", "/api/jobs/" + "0" * 32)[0] == 404
    assert req(port, "GET", "/api/jobs/..%2F..%2Fetc%2Fpasswd")[0] == 404


@pytest.mark.parametrize("fmt,needle", [("txt", b"Hello world."), ("srt", b"-->"), ("vtt", b"WEBVTT"),
                                        ("md", b"# clip.mp4"), ("json", b'"segments"')])
def test_exports(server, fmt, needle):
    port, jid = server
    status, headers, body = req(port, "GET", f"/api/jobs/{jid}/export?fmt={fmt}")
    assert status == 200 and needle in body
    assert f'filename="clip.{fmt}"' in headers["Content-Disposition"]


def test_bad_export_format_rejected(server):
    port, jid = server
    assert req(port, "GET", f"/api/jobs/{jid}/export?fmt=exe")[0] == 400


def test_media_full_and_range_requests(server):
    port, jid = server
    status, headers, body = req(port, "GET", f"/api/jobs/{jid}/media")
    assert status == 200 and body == MEDIA and headers["Accept-Ranges"] == "bytes"

    status, headers, body = req(port, "GET", f"/api/jobs/{jid}/media", {"Range": "bytes=100-199"})
    assert status == 206 and body == MEDIA[100:200]
    assert headers["Content-Range"] == f"bytes 100-199/{len(MEDIA)}"

    status, _, body = req(port, "GET", f"/api/jobs/{jid}/media", {"Range": "bytes=10000-"})
    assert status == 206 and body == MEDIA[10000:]

    status, _, body = req(port, "GET", f"/api/jobs/{jid}/media", {"Range": "bytes=-50"})
    assert status == 206 and body == MEDIA[-50:]

    assert req(port, "GET", f"/api/jobs/{jid}/media", {"Range": "bytes=99999-"})[0] == 416
    assert req(port, "GET", f"/api/jobs/{jid}/media", {"Range": "bytes=abc"})[0] == 416


def test_post_requires_custom_header(server):
    port, _ = server
    assert req(port, "POST", "/api/jobs?name=a.mp4", {"Content-Length": "3"}, b"abc")[0] == 403


def test_foreign_host_header_is_refused(server):
    port, _ = server
    assert req(port, "GET", "/", {"Host": "evil.example.com"})[0] == 403


def test_empty_upload_rejected(server):
    port, _ = server
    status, _, _ = req(port, "POST", "/api/jobs?name=a.mp4", {"X-Verbatim": "1", "Content-Length": "0"})
    assert status == 400


def test_upload_is_stored_with_safe_name_and_queued(server, tmp_path):
    port, _ = server
    status, _, body = req(port, "POST", "/api/jobs?name=../../evil.mp4", {"X-Verbatim": "1"}, b"not really a video")
    assert status == 202
    jid = json.loads(body)["id"]
    status, _, body = req(port, "GET", f"/api/jobs/{jid}")
    assert status == 200 and json.loads(body)["state"] in ("queued", "running", "error")
