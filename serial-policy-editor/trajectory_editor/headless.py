"""Single-owner, local HTTP adapter for SPE. No terminal interaction required."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets
from urllib.parse import urlsplit, parse_qs

from .backend_factory import create_backend, BACKEND_NAMES
from .decoder import KV_CACHE_TYPES, LlamaCppSettings
from .domain import EditorError, SamplingConfig
from .episode_actions import action_from_dict, Hold, Finish
from .episode_engine import EpisodeEngine
from .episode_lifecycle import (_create_episode, _restore_engine, _rewind_episode,
                                _fork_engine)
from .episode_policy import EpisodeRunner
from .episode_store import EpisodeStore


class Conflict(EditorError):
    pass


class Session:
    """All calls must run on the same owner thread as the backend and store.

    Revisions include a process nonce. Request receipts last for this process;
    old revisions cannot accidentally become valid after a restart.
    """
    def __init__(self, backend, store, provenance):
        self.backend, self.store, self.provenance = backend, store, provenance
        self.engine = None
        self.episode_id = None
        self.epoch = secrets.token_hex(12)
        self.sequence = 0
        self.receipts = {}
        self.notices = []

    @property
    def revision(self):
        return f"{self.epoch}:{self.sequence}"

    def state(self):
        result = {"revision": self.revision, "episode": None, "notices": list(self.notices)}
        if self.engine is not None:
            e = self.engine
            result.update(episode=self.store.get_episode(self.episode_id),
                          boundary=e.boundary, remaining=e.remaining,
                          ended=e.ended, checkpointed=e.checkpointed,
                          sampling=e.sampling.to_dict())
        return result

    def observation(self, start=1, count=12):
        if self.engine is None:
            raise EditorError("Open an episode first")
        if not 1 <= count <= 100 or not 1 <= start <= self.backend.vocabulary_size():
            raise EditorError("Invalid candidate range (maximum count is 100)")
        o = self.engine.observe()
        return {"revision": self.revision, "boundary": o.boundary,
                "proposal": {"token_id": o.proposal_token_id, "text": o.proposal_text,
                             "rank": o.proposal_raw_rank},
                "candidates": [c.to_dict() for c in self.engine.candidates(o, start_rank=start, count=count)]}

    def mutate(self, operation, payload):
        if not isinstance(payload, dict):
            raise EditorError("Request must be a JSON object")
        rid = payload.get("request_id")
        if not isinstance(rid, str) or not 1 <= len(rid) <= 128:
            raise EditorError("A request_id of 1–128 characters is required")
        signature = json.dumps([operation, payload], sort_keys=True)
        if rid in self.receipts:
            previous, result, error = self.receipts[rid]
            if previous != signature:
                raise Conflict("Request ID was already used for a different request")
            if error:
                raise Conflict(error)
            return result
        if payload.get("revision") != self.revision:
            raise Conflict("The session changed. Refresh before making this choice.")
        # Bound memory without evicting IDs that could otherwise execute twice.
        if len(self.receipts) >= 10000:
            raise Conflict("Session receipt limit reached; restart the service")
        self.notices = []
        self.sequence += 1
        try:
            result = self._execute(operation, payload)
            response = {**self.state(), "result": result}
        except Exception as exc:
            # Backend errors can leave an in-memory prefix partially changed.
            # Drop active ownership; saved evidence remains available for recovery.
            if not isinstance(exc, (EditorError, TypeError, ValueError)):
                self.engine = None
                self.episode_id = None
            self.receipts[rid] = (signature, None, str(exc))
            raise
        self.receipts[rid] = (signature, response, None)
        return response

    def _execute(self, operation, p):
        if operation == "open":
            source = p.get("episode_id")
            if source:
                identifier = self.store.resolve_id(str(source))
                saved = self.store.get_episode(identifier)
                for field in ("backend", "model_path", "model_sha256", "vocabulary_size"):
                    if saved["backend"].get(field) is not None and self.provenance.get(field) != saved["backend"][field]:
                        raise EditorError("Episode uses a different model; launch the service with that model")
                engine = _restore_engine(self.store, identifier, self.backend,
                                         max_tokens=None, sampling_override=None, notice=self.notices.append)
            else:
                prompt = p.get("prompt")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise EditorError("Enter a prompt")
                engine = EpisodeEngine(self.backend, sampling=SamplingConfig(), initial_text=prompt)
                identifier = _create_episode(self.store, engine, backend_provenance=self.provenance,
                                             metadata={"client": "http"})
            self.engine, self.episode_id = engine, identifier
            return None
        if self.engine is None:
            raise EditorError("Open an episode first")
        e, identifier = self.engine, self.episode_id
        runner = EpisodeRunner(e, self.store, identifier)
        if operation == "actions":
            raw = p.get("action")
            if not isinstance(raw, dict):
                raise EditorError("action must be an object")
            action = action_from_dict(raw)
            if isinstance(action, Finish):
                raise EditorError("Use a bounded hold to generate, or /end to seal the episode")
            if isinstance(action, Hold) and action.limit > 256:
                raise EditorError("This HTTP preview limits each hold to 256 tokens")
            if e.ended or e.checkpointed:
                raise EditorError("This episode has no live decision; resume its budget or open another episode")
            class Once:
                def choose(self, engine, observation):
                    return action
            result = runner.run(live_policy=Once(), max_live_actions=1)
            return {"outcomes": [asdict(x) for x in result.outcomes],
                    "handoff_reason": result.handoff_reason}
        if operation in {"rewind", "fork"}:
            boundary = p.get("boundary")
            if type(boundary) is not int or not 0 <= boundary <= e.boundary:
                raise EditorError(f"Boundary must be an integer between 0 and {e.boundary}")
            if operation == "rewind":
                if e.ended:
                    raise EditorError("Fork a sealed episode instead of rewinding it")
                return _rewind_episode(self.store, identifier, e, boundary, notice=self.notices.append)
            branch = _fork_engine(self.store, identifier, e, boundary, backend=self.backend, max_tokens=None, notice=self.notices.append)
            child = _create_episode(self.store, branch, backend_provenance=self.provenance,
                                    parent_episode_id=identifier, fork_boundary=boundary, mode="fork")
            self.engine, self.episode_id = branch, child
            return None
        if operation == "settings":
            values = p.get("sampling", {})
            if not isinstance(values, dict) or values.keys() - asdict(e.sampling).keys():
                raise EditorError("Unknown sampler setting")
            sampling = replace(e.sampling, **values)
            if e.ended:
                raise EditorError("Fork a sealed episode to continue")
            allowance = p.get("max_tokens", "keep")
            if allowance != "keep" and allowance is not None and (type(allowance) is not int or allowance < 1):
                raise EditorError("Token allowance must be positive or null")
            e.resume(max_tokens=allowance, sampling=sampling)
            self.store.record_sampling_segment(identifier, start_boundary=e.boundary,
                sampling=e.sampling, stream_fingerprint=e.stream_fingerprint, coordinate_offset=e.coordinate_offset)
            self.store.record_budget(identifier, e.boundary, e.max_tokens, e.checkpoint_boundary)
            self.store.update_episode(identifier, visible_text=e.backend.render(e.visible_token_ids),
                                      max_tokens=e.max_tokens, status="open")
            return None
        if operation == "end":
            e.terminate("menu-end")
            self.store.finish_episode(identifier, visible_text=e.backend.render(e.visible_token_ids),
                terminal_token_id=e.terminal_token_id, terminal_reason=e.terminal_reason)
            return None
        if operation == "close":
            self.engine = self.episode_id = None
            return None
        raise EditorError("Unknown operation")


def make_handler(session):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value, content_type="application/json"):
            data = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)

        def trusted(self):
            host = self.headers.get("Host", "")
            expected = f"127.0.0.1:{self.server.server_port}"
            if host not in {expected, f"localhost:{self.server.server_port}"}:
                self.reply(403, {"error": "Local host required"})
                return False
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{host}":
                self.reply(403, {"error": "Same-origin requests required"})
                return False
            return True

        def do_GET(self):
            if not self.trusted():
                return
            try:
                url = urlsplit(self.path)
                if url.path == "/api/session":
                    return self.reply(200, session.state())
                if url.path == "/api/episodes":
                    return self.reply(200, {"episodes": [{**row, "initial_text": session.store.get_episode(row["episode_id"])["initial_text"][:80]} for row in session.store.list_episodes()]})
                if url.path == "/api/observation":
                    q = parse_qs(url.query)
                    return self.reply(200, session.observation(int(q.get("start", [1])[0]), int(q.get("count", [12])[0])))
                names = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
                types = {"/": "text/html; charset=utf-8", "/app.js": "text/javascript", "/style.css": "text/css"}
                if url.path in names:
                    return self.reply(200, (Path(__file__).parent / "web" / names[url.path]).read_bytes(), types[url.path])
                self.reply(404, {"error": "Not found"})
            except (EditorError, ValueError, TypeError) as exc:
                self.reply(400, {"error": str(exc), "revision": session.revision})

        def do_POST(self):
            if not self.trusted():
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self.reply(415, {"error": "Use application/json"})
            operations = {"/api/session": "open", **{f"/api/session/{x}": x for x in ("actions", "fork", "rewind", "settings", "close", "end")}}
            if self.path not in operations:
                return self.reply(404, {"error": "Not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    return self.reply(413, {"error": "Request must be between 1 and 65536 bytes"})
                payload = json.loads(self.rfile.read(length))
                self.reply(200, session.mutate(operations[self.path], payload))
            except Conflict as exc:
                self.reply(409, {"error": str(exc), "revision": session.revision})
            except (EditorError, ValueError, TypeError) as exc:
                self.reply(400, {"error": str(exc), "revision": session.revision})
            except Exception:
                import traceback
                traceback.print_exc()
                self.reply(500, {"error": "Execution failed. Refresh and reopen the saved episode.", "revision": session.revision})

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local SPE browser editor and HTTP API")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", choices=BACKEND_NAMES, default="llama.cpp")
    parser.add_argument("--workspace", type=Path, default=Path("headless-episodes.sqlite3"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    for component in ("k", "v"):
        parser.add_argument(f"--cache-type-{component}", dest=f"type_{component}", choices=KV_CACHE_TYPES)
    args = parser.parse_args(argv)
    backend = create_backend(args.backend, args.model,
        llama_settings=LlamaCppSettings(n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers,
            type_k=args.type_k, type_v=args.type_v))
    provenance = dict(backend.provenance(include_model_sha256=False))
    provenance["model_path"] = str(args.model.resolve())
    with EpisodeStore(args.workspace) as store:
        session = Session(backend, store, provenance)
        with HTTPServer(("127.0.0.1", args.port), make_handler(session)) as server:
            server.timeout = 1
            print(f"SPE headless: http://127.0.0.1:{server.server_port}", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
