import asyncio
import io
import logging
import os
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp.abc import AbstractResolver, ResolveResult

import cognee
from cognee import SearchType

logger = logging.getLogger(__name__)

_RECALL_TYPE_MAP = {
    "CHUNKS": SearchType.CHUNKS,
    "GRAPH_COMPLETION": SearchType.GRAPH_COMPLETION,
    "GRAPH_SUMMARY_COMPLETION": SearchType.GRAPH_SUMMARY_COMPLETION,
}

_cloud_client: "CloudMemoryClient | None" = None
_graph_empty_logged = False


@dataclass(frozen=True)
class MemorySettings:
    openai_api_key: str
    cognee_llm_model: str
    cognee_root_dir: str
    sessions_dataset: str
    recall_type: SearchType
    cognee_base_url: str = ""
    cognee_api_key: str = ""
    recall_timeout: float = 0.4
    turn_recall_timeout: float = 8.0

    @property
    def use_cloud(self) -> bool:
        return bool(self.cognee_base_url and self.cognee_api_key)


def _strip_env(value: str) -> str:
    return value.strip().strip('"').strip("'")


def load_memory_settings() -> MemorySettings:
    api_key = _strip_env(os.environ.get("OPENAI_API_KEY", ""))
    recall_name = _strip_env(os.environ.get("COGNEE_RECALL_TYPE", "CHUNKS")).upper()
    recall_type = _RECALL_TYPE_MAP.get(recall_name, SearchType.CHUNKS)
    cognee_base_url = _strip_env(
        os.environ.get("COGNEE_BASE_URL") or os.environ.get("COGNEE_SERVICE_URL") or ""
    ).rstrip("/")
    cognee_api_key = _strip_env(os.environ.get("COGNEE_API_KEY", ""))
    default_timeout = "8.0" if cognee_base_url and cognee_api_key else "0.4"
    recall_timeout = float(os.environ.get("COGNEE_RECALL_TIMEOUT", default_timeout))
    turn_recall_timeout = float(
        os.environ.get("COGNEE_TURN_RECALL_TIMEOUT", str(recall_timeout))
    )
    default_dataset = "default_dataset" if cognee_base_url and cognee_api_key else "nyra_sessions"
    return MemorySettings(
        openai_api_key=api_key,
        cognee_llm_model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        cognee_root_dir=os.environ.get("COGNEE_SYSTEM_ROOT_DIRECTORY", ".cognee_system"),
        sessions_dataset=os.environ.get("SESSIONS_DATASET", default_dataset),
        recall_type=recall_type,
        cognee_base_url=cognee_base_url,
        cognee_api_key=cognee_api_key,
        recall_timeout=recall_timeout,
        turn_recall_timeout=turn_recall_timeout,
    )


def _parse_lookup_output(stdout: str) -> list[str]:
    ips: list[str] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if " has address " in line:
            ips.append(line.rsplit(" ", 1)[-1])
        elif line[0].isdigit():
            ips.append(line.split()[0])
    return ips


def _pick_ip(candidates: list[str]) -> str:
    public = [ip for ip in candidates if not ip.startswith(("100.", "10.", "192.168.", "172."))]
    return (public or candidates)[0]


def _resolve_host_sync(host: str) -> str:
    try:
        return _pick_ip([socket.gethostbyname(host)])
    except OSError:
        pass

    for cmd in (["dig", "+short", host, "A"], ["host", "-t", "A", host]):
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

        ips = _parse_lookup_output(proc.stdout)
        if ips:
            return _pick_ip(ips)

    raise socket.gaierror(f"Could not resolve {host}")


class DigFallbackResolver(AbstractResolver):
    """aiohttp resolver that falls back to dig/host when getaddrinfo fails."""

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: socket.AddressFamily = socket.AF_INET,
    ) -> list[ResolveResult]:
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(
                host,
                port,
                type=socket.SOCK_STREAM,
                family=family,
            )
        except OSError:
            ip = await asyncio.to_thread(_resolve_host_sync, host)
            return [
                ResolveResult(
                    hostname=host,
                    host=ip,
                    port=port,
                    family=socket.AF_INET,
                    proto=socket.IPPROTO_TCP,
                    flags=socket.AI_NUMERICHOST,
                )
            ]

        results: list[ResolveResult] = []
        for info_family, _, proto, _, address in infos:
            if info_family == socket.AF_INET6:
                continue
            host_addr, addr_port = address[:2]
            results.append(
                ResolveResult(
                    hostname=host,
                    host=host_addr,
                    port=addr_port,
                    family=info_family,
                    proto=proto,
                    flags=socket.AI_NUMERICHOST,
                )
            )
        if results:
            return results

        ip = await asyncio.to_thread(_resolve_host_sync, host)
        return [
            ResolveResult(
                hostname=host,
                host=ip,
                port=port,
                family=socket.AF_INET,
                proto=socket.IPPROTO_TCP,
                flags=socket.AI_NUMERICHOST,
            )
        ]

    async def close(self) -> None:
        return None


class CloudMemoryClient:
    def __init__(self, settings: MemorySettings) -> None:
        self._settings = settings
        self._session: aiohttp.ClientSession | None = None
        self._dataset_name = settings.sessions_dataset
        self._graph_ready = False

    @property
    def base_url(self) -> str:
        return self._settings.cognee_base_url

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    async def start(self) -> None:
        connector = aiohttp.TCPConnector(resolver=DigFallbackResolver())
        self._session = aiohttp.ClientSession(
            headers={"X-Api-Key": self._settings.cognee_api_key},
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=120, sock_connect=10),
        )

        try:
            assert self._session is not None
            async with self._session.get(f"{self.base_url}/health") as resp:
                if resp.status != 200:
                    logger.warning(
                        "[memory] Cognee Cloud health check returned %s",
                        resp.status,
                    )
        except Exception as exc:
            logger.warning("[memory] Cognee Cloud health check failed: %s", exc)

        self._dataset_name = await self._resolve_dataset(self._settings.sessions_dataset)
        self._graph_ready = await self._probe_graph_ready()
        if not self._graph_ready:
            await self._bootstrap_if_configured()

    async def list_datasets(self) -> list[dict[str, Any]]:
        if self._session is None:
            return []
        async with self._session.get(f"{self.base_url}/api/v1/datasets/") as resp:
            if resp.status >= 400:
                return []
            data = await resp.json()
            return data if isinstance(data, list) else []

    async def _resolve_dataset(self, preferred: str) -> str:
        datasets = await self.list_datasets()
        names = {item.get("name") for item in datasets if item.get("name")}
        if preferred in names:
            return preferred
        if "default_dataset" in names:
            logger.info(
                "[memory] dataset '%s' not found on cloud tenant; using default_dataset",
                preferred,
            )
            return "default_dataset"
        if names:
            chosen = sorted(names)[0]
            logger.info(
                "[memory] dataset '%s' not found on cloud tenant; using '%s'",
                preferred,
                chosen,
            )
            return chosen
        logger.warning(
            "[memory] no datasets found on cloud tenant; using configured name '%s'",
            preferred,
        )
        return preferred

    async def _probe_graph_ready(self) -> bool:
        try:
            await self.recall("healthcheck", top_k=1)
            return True
        except RuntimeError as exc:
            if "Recall prerequisites not met" in str(exc):
                return False
            raise

    async def _bootstrap_if_configured(self) -> None:
        raw_paths = os.environ.get("COGNEE_BOOTSTRAP_PATH", "").strip()
        if not raw_paths:
            logger.info(
                "[memory] cloud graph is empty for dataset '%s'. "
                "Set COGNEE_BOOTSTRAP_PATH or run: uv run python ingest_cloud.py <files>",
                self._dataset_name,
            )
            return

        paths = [Path(p.strip()) for p in raw_paths.split(",") if p.strip()]
        texts: list[str] = []
        for path in paths:
            if not path.exists():
                logger.warning("[memory] bootstrap file not found: %s", path)
                continue
            texts.append(path.read_text(encoding="utf-8", errors="replace"))

        if not texts:
            return

        logger.info(
            "[memory] bootstrapping cloud graph from %d file(s) into '%s' …",
            len(texts),
            self._dataset_name,
        )
        for text in texts:
            await self.remember_text(text)
        self._graph_ready = await self._probe_graph_ready()
        if self._graph_ready:
            logger.info("[memory] cloud graph bootstrap complete")
        else:
            logger.warning("[memory] cloud graph bootstrap finished but recall still unavailable")

    async def remember_text(self, text: str) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("CloudMemoryClient is not started")

        form = aiohttp.FormData()
        form.add_field("datasetName", self._dataset_name)
        form.add_field(
            "data",
            io.BytesIO(text.encode("utf-8")),
            filename="memory.txt",
            content_type="text/plain",
        )
        async with self._session.post(f"{self.base_url}/api/v1/remember", data=form) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote remember failed ({resp.status}): {body}")
            return await resp.json()

    async def recall(self, query_text: str, *, top_k: int = 3) -> list[Any]:
        if self._session is None:
            raise RuntimeError("CloudMemoryClient is not started")

        payload = {
            "query": query_text,
            "search_type": self._settings.recall_type.value,
            "datasets": [self._dataset_name],
            "top_k": top_k,
        }
        async with self._session.post(
            f"{self.base_url}/api/v1/recall",
            json=payload,
        ) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Remote recall failed ({resp.status}): {body}")
            return await resp.json()

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


async def init_cognee(settings: MemorySettings) -> None:
    global _cloud_client

    if settings.use_cloud:
        _cloud_client = CloudMemoryClient(settings)
        await _cloud_client.start()
        logger.info(
            "[memory] connected to Cognee Cloud at %s (dataset: %s)",
            settings.cognee_base_url,
            _cloud_client.dataset_name,
        )
        return

    if settings.openai_api_key:
        cognee.config.set_llm_api_key(settings.openai_api_key)
    cognee.config.set_llm_model(settings.cognee_llm_model)
    cognee.config.system_root_directory(os.path.abspath(settings.cognee_root_dir))
    await cognee.run_migrations()
    logger.info("[memory] using local Cognee at %s", settings.cognee_root_dir)


async def shutdown_cognee(settings: MemorySettings) -> None:
    global _cloud_client

    if settings.use_cloud and _cloud_client is not None:
        await _cloud_client.close()
        _cloud_client = None


async def _existing_datasets(candidates: list[str]) -> list[str]:
    """Return only dataset names that actually exist — avoids DatasetNotFoundError."""
    try:
        all_ds = await cognee.datasets.list_datasets()
        existing = {getattr(d, "name", None) for d in all_ds}
        return [name for name in candidates if name in existing]
    except Exception:
        return []


def _format_recall_results(results: list, *, top_k: int) -> str:
    snippets: list[str] = []
    for item in results[:top_k]:
        if not item:
            continue
        if isinstance(item, dict):
            for key in ("text", "content", "answer", "response"):
                value = item.get(key)
                if value:
                    snippets.append(str(value))
                    break
            else:
                snippets.append(str(item))
        else:
            snippets.append(str(item))
    return "\n".join(snippets) if snippets else ""


async def recall_for_transcript(
    query: str,
    settings: MemorySettings,
    *,
    top_k: int = 3,
    timeout: float | None = None,
) -> str:
    """Graph/vector recall for a stable STT transcript.

    Timeout-guarded so it never stalls the voice loop.
    """
    global _graph_empty_logged

    if not query.strip():
        return ""

    recall_timeout = timeout if timeout is not None else settings.recall_timeout
    logger.info("[memory] recall started (timeout=%.1fs): %s", recall_timeout, query[:80])

    try:
        if settings.use_cloud:
            if _cloud_client is None:
                logger.warning("[memory] cloud client not initialized")
                return ""
            results = await asyncio.wait_for(
                _cloud_client.recall(query, top_k=top_k),
                timeout=recall_timeout,
            )
        else:
            datasets = await _existing_datasets([settings.sessions_dataset])
            if not datasets:
                return ""
            results = await asyncio.wait_for(
                cognee.recall(
                    query_text=query,
                    query_type=settings.recall_type,
                    datasets=datasets,
                    top_k=top_k,
                ),
                timeout=recall_timeout,
            )
        memory_text = _format_recall_results(results or [], top_k=top_k)
        if memory_text:
            logger.info("[memory] recall returned %d chars", len(memory_text))
        else:
            logger.info("[memory] recall returned no matches")
        return memory_text
    except asyncio.TimeoutError:
        logger.warning(
            "[memory] recall timed out after %.1fs for query: %s",
            recall_timeout,
            query[:80],
        )
        return ""
    except (aiohttp.ClientConnectorError, socket.gaierror, OSError) as exc:
        logger.warning("[memory] recall network error: %s", exc)
        return ""
    except RuntimeError as exc:
        if "Recall prerequisites not met" in str(exc):
            if not _graph_empty_logged:
                logger.info(
                    "[memory] cloud graph has no indexed memories yet — "
                    "run ingest_cloud.py or set COGNEE_BOOTSTRAP_PATH"
                )
                _graph_empty_logged = True
            return ""
        logger.warning("[memory] recall failed: %s", exc)
        return ""
    except Exception:
        logger.warning("[memory] recall failed", exc_info=True)
        return ""
