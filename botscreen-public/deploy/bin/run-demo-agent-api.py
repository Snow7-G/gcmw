#!/usr/bin/env python3
"""Pinned demo Agent API for the packaged Electron renderer (loopback only).

This is a thin, *deployment-side* entry point. It deliberately does NOT
reimplement anything: it runs the repository's own demo assembly
(``scripts/demo_showcase.py``) and wraps ``create_app`` so the packaged renderer
(which has the opaque ``Origin: null``) can read the API.

Why a wrapper instead of a code change: the packaged renderer's origin cannot be
predicted by the server library, and the demo assembly intentionally only adds
CORS for local dev frontends. Keeping the extra middleware here means the
shipped server keeps its own, stricter posture and this file can be reviewed as
deployment configuration.

Contract notes (verified by tests/test_deploy_assets.py):
- binds 127.0.0.1 ONLY — the demo API has fixed low-entropy credentials and must
  never be reachable from the LAN;
- the origin allowlist is exact: no ``*``, and credentials are allowed because
  the origins are explicit (per CORS, ``*`` + credentials is invalid);
- authentication is untouched: an unknown/missing device credential is still
  rejected by the API's own entry guard, CORS or not.
"""

from __future__ import annotations

import argparse
import os
import runpy
import signal
import sys
import threading
from pathlib import Path

#: Where the (tracked) server directory lives on the target host. Overridable so
#: the same file works for a rehearsal checkout and for the pinned release tree.
DEFAULT_SERVER_DIR = "@@GCMW_SERVER_DIR@@"

#: Packaged Electron sends an opaque origin; the other two are the local dev
#: frontends. Exact strings only — never a wildcard.
ALLOWED_ORIGINS: tuple[str, ...] = (
    "null",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)
ALLOWED_METHODS: tuple[str, ...] = ("GET", "POST", "DELETE", "OPTIONS")
ALLOWED_HEADERS: tuple[str, ...] = ("Authorization", "Content-Type", "Last-Event-ID")

#: Loopback only. ``start_server`` also refuses anything else, by design.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8001

DEMO_SCRIPT_RELATIVE = Path("scripts") / "demo_showcase.py"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-dir",
        default=os.environ.get("GCMW_SERVER_DIR", DEFAULT_SERVER_DIR),
        help="directory containing scripts/demo_showcase.py",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.host != DEFAULT_HOST:
        print(
            f"refusing to bind {args.host!r}: the demo Agent API is loopback only",
            file=sys.stderr,
        )
        return 2

    demo_script = Path(args.server_dir) / DEMO_SCRIPT_RELATIVE
    if not demo_script.is_file():
        print(f"demo assembly not found at {demo_script}", file=sys.stderr)
        return 2

    namespace = runpy.run_path(str(demo_script))
    original_create_app = namespace["create_app"]

    def create_app_for_packaged_renderer(*call_args, **call_kwargs):
        from fastapi.middleware.cors import CORSMiddleware

        app = original_create_app(*call_args, **call_kwargs)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(ALLOWED_ORIGINS),
            allow_credentials=True,
            allow_methods=list(ALLOWED_METHODS),
            allow_headers=list(ALLOWED_HEADERS),
        )
        return app

    namespace["start_server"].__globals__["create_app"] = (
        create_app_for_packaged_renderer
    )
    handle = namespace["start_server"](
        host=args.host, port=args.port, cors_for_dev_frontends=False
    )

    stopping = threading.Event()

    def request_stop(_signum, _frame):
        stopping.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        while handle.thread.is_alive() and not stopping.wait(1):
            pass
    finally:
        namespace["stop_server"](handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
