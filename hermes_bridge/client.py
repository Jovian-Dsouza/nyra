import logging
from dataclasses import dataclass
from typing import Any

import httpx

from hermes_bridge.settings import HermesSettings

logger = logging.getLogger(__name__)


class HermesClientError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class SubmitRunResult:
    run_id: str
    status: str


@dataclass(frozen=True)
class RunStatus:
    run_id: str
    status: str
    output: str | None
    error: str | None


@dataclass(frozen=True)
class CreateJobResult:
    job_id: str
    name: str
    schedule: str


class HermesClient:
    def __init__(self, settings: HermesSettings, *, session_key: str) -> None:
        self._settings = settings
        self._session_key = session_key
        self._client = httpx.AsyncClient(
            base_url=settings.api_url,
            timeout=httpx.Timeout(
                connect=settings.connect_timeout,
                read=settings.request_timeout,
                write=settings.request_timeout,
                pool=settings.connect_timeout,
            ),
            headers=self._auth_headers(),
        )

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._settings.api_key:
            headers["Authorization"] = f"Bearer {self._settings.api_key}"
        if self._session_key:
            headers["X-Hermes-Session-Key"] = self._session_key
        return headers

    async def close(self) -> None:
        await self._client.aclose()

    async def health_check(self) -> bool:
        try:
            response = await self._client.get("/health")
            if response.status_code != 200:
                return False
            data = response.json()
            return data.get("status") == "ok"
        except Exception:
            logger.debug("[hermes] health check failed", exc_info=True)
            return False

    async def submit_run(
        self,
        user_input: str,
        *,
        session_id: str | None = None,
        instructions: str | None = None,
    ) -> SubmitRunResult:
        body: dict[str, Any] = {"input": user_input}
        if session_id:
            body["session_id"] = session_id
        if instructions:
            body["instructions"] = instructions

        response = await self._client.post("/v1/runs", json=body)
        if response.status_code == 429:
            raise HermesClientError(
                "Hermes is at capacity — try again in a moment.",
                status_code=429,
            )
        if response.status_code >= 400:
            detail = _extract_error(response)
            raise HermesClientError(detail, status_code=response.status_code)

        data = response.json()
        run_id = data.get("run_id")
        if not run_id:
            raise HermesClientError("Hermes did not return a run_id.")
        return SubmitRunResult(run_id=run_id, status=data.get("status", "started"))

    async def get_run(self, run_id: str) -> RunStatus:
        response = await self._client.get(f"/v1/runs/{run_id}")
        if response.status_code == 404:
            raise HermesClientError("Run not found.", status_code=404)
        if response.status_code >= 400:
            detail = _extract_error(response)
            raise HermesClientError(detail, status_code=response.status_code)

        data = response.json()
        output = data.get("output")
        error = data.get("error")
        if isinstance(error, dict):
            error = error.get("message") or str(error)
        return RunStatus(
            run_id=data.get("run_id", run_id),
            status=data.get("status", "unknown"),
            output=str(output) if output is not None else None,
            error=str(error) if error is not None else None,
        )

    async def stop_run(self, run_id: str) -> None:
        response = await self._client.post(f"/v1/runs/{run_id}/stop")
        if response.status_code == 404:
            raise HermesClientError("Run not found.", status_code=404)
        if response.status_code >= 400:
            detail = _extract_error(response)
            raise HermesClientError(detail, status_code=response.status_code)

    async def create_scheduled_job(
        self,
        *,
        name: str,
        schedule: str,
        prompt: str,
    ) -> CreateJobResult:
        body = {
            "name": name,
            "schedule": schedule,
            "prompt": prompt,
            "deliver": "local",
        }
        response = await self._client.post("/api/jobs", json=body)
        if response.status_code >= 400:
            detail = _extract_error(response)
            raise HermesClientError(detail, status_code=response.status_code)

        data = response.json()
        job = data.get("job") or data
        job_id = job.get("id") or job.get("job_id") or ""
        return CreateJobResult(
            job_id=job_id,
            name=job.get("name", name),
            schedule=job.get("schedule", schedule),
        )


def _extract_error(response: httpx.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, dict):
            if "error" in data:
                err = data["error"]
                if isinstance(err, dict):
                    return str(err.get("message") or err)
                return str(err)
            if "message" in data:
                return str(data["message"])
    except Exception:
        pass
    return f"Hermes request failed with status {response.status_code}"
