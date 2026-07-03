"""Ingest markdown/text files into Cognee Cloud memory."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from cognee_memory import CloudMemoryClient, load_memory_settings


def _collect_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("*.md")) + sorted(path.rglob("*.txt"))
        return [f for f in files if f.is_file()]
    print(f"Error: {path} does not exist", file=sys.stderr)
    sys.exit(1)


async def ingest(paths: list[Path], dataset: str | None) -> None:
    load_dotenv(".env")
    settings = load_memory_settings()
    if not settings.use_cloud:
        print("Error: set COGNEE_BASE_URL and COGNEE_API_KEY in .env", file=sys.stderr)
        sys.exit(1)

    client = CloudMemoryClient(settings)
    await client.start()
    if dataset:
        client._dataset_name = dataset

    print(f"Target dataset: {client.dataset_name}")
    print(f"Ingesting {len(paths)} file(s) …\n")

    for i, path in enumerate(paths, 1):
        text = path.read_text(encoding="utf-8", errors="replace")
        print(f"  [{i}/{len(paths)}] remembering {path} ({len(text)} chars)")
        result = await client.remember_text(text)
        print(f"           status={result.get('status')} elapsed={result.get('elapsed_seconds')}s")

    ready = await client._probe_graph_ready()
    print("\nRecall ready:" if ready else "\nRecall still unavailable:", ready)
    await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest files into Cognee Cloud memory")
    parser.add_argument("path", help="File or directory to ingest")
    parser.add_argument("--dataset", help="Override SESSIONS_DATASET from .env")
    args = parser.parse_args()

    files = _collect_files(Path(args.path))
    if not files:
        print("No files found.")
        sys.exit(0)

    asyncio.run(ingest(files, dataset=args.dataset))


if __name__ == "__main__":
    main()
