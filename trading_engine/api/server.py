"""Serveur HTTP minimal du dashboard (README §36), sans dépendance.

    GET /              dashboard (HTML)
    GET /api/state     état complet (JSON)
    GET /api/history   série temporelle par intervalle (JSON)
    GET /api/decisions dernières décisions (JSON)
    GET /healthz       ok

**Lecture seule** : aucune route ne modifie le moteur (pas de reset du
Safety Engine, pas d'ordre). Écoute sur 127.0.0.1 par défaut ; l'exposer sur
le réseau (0.0.0.0) n'a aucune authentification : à éviter.
"""

from __future__ import annotations

import asyncio
import json
import logging
from importlib import resources
from urllib.parse import urlsplit

from trading_engine.api.state import build_state, to_jsonable

logger = logging.getLogger(__name__)


def dashboard_html() -> str:
    return resources.files("trading_engine.dashboard").joinpath("index.html").read_text(encoding="utf-8")


class DashboardServer:
    def __init__(self, engine, host: str = "127.0.0.1", port: int = 8050) -> None:
        self.engine = engine
        self.host = host
        self.port = port
        self._server: asyncio.base_events.Server | None = None
        if host not in ("127.0.0.1", "localhost", "::1"):
            logger.warning("dashboard exposed on %s without authentication (read-only)", host)

    def _route(self, path: str) -> tuple[int, str, bytes]:
        if path in ("/", "/index.html"):
            return 200, "text/html; charset=utf-8", dashboard_html().encode("utf-8")
        if path == "/healthz":
            return 200, "text/plain; charset=utf-8", b"ok"
        if path == "/api/state":
            body = build_state(self.engine)
        elif path == "/api/history":
            body = to_jsonable(list(self.engine.history))
        elif path == "/api/decisions":
            body = to_jsonable(list(self.engine.decisions)[-200:])
        else:
            return 404, "application/json", b'{"error":"not found"}'
        return 200, "application/json", json.dumps(body, ensure_ascii=False).encode("utf-8")

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=5)
            while (await asyncio.wait_for(reader.readline(), timeout=5)) not in (b"\r\n", b"\n", b""):
                pass                                            # en-têtes ignorés
            parts = request_line.decode("latin-1").split()
            if len(parts) < 2:
                return
            method, target = parts[0], parts[1]
            if method != "GET":
                status, ctype, body = 405, "text/plain", b"read-only"
            else:
                try:
                    status, ctype, body = self._route(urlsplit(target).path)
                except Exception:
                    logger.exception("dashboard route failed")
                    status, ctype, body = 500, "text/plain", b"internal error"
            reason = {200: "OK", 404: "Not Found", 405: "Method Not Allowed", 500: "Internal Server Error"}[status]
            writer.write(
                f"HTTP/1.1 {status} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {len(body)}\r\n"
                "Cache-Control: no-store\r\nConnection: close\r\n\r\n".encode("latin-1") + body
            )
            await writer.drain()
        except (asyncio.TimeoutError, ConnectionError):
            pass
        finally:
            writer.close()

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self.host, self.port)
        sock = self._server.sockets[0].getsockname()
        self.port = sock[1]
        logger.info("dashboard on http://%s:%s", self.host, self.port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None


def export_static(engine, path) -> None:
    """Dashboard autonome (un seul fichier HTML, état embarqué) : pour analyser
    une simulation ou un replay après coup, sans serveur."""
    state = json.dumps(build_state(engine), ensure_ascii=False).replace("</", "<\\/")
    html = dashboard_html().replace(
        "<script id=\"embedded-state\" type=\"application/json\">null</script>",
        f"<script id=\"embedded-state\" type=\"application/json\">{state}</script>",
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
