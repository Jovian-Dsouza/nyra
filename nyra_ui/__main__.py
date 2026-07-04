"""Entrypoint: `python -m nyra_ui`.

Starts the socket server (background thread) and runs the pygame loop on
this process's main thread. Any startup failure is logged and this process
exits quietly — it must never take down the agent worker that spawned it.
"""

import argparse
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _strip_env(value: str) -> str:
    return value.strip().strip('"').strip("'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=_strip_env(os.environ.get("NYRA_UI_HOST", "127.0.0.1")))
    parser.add_argument("--port", type=int, default=int(_strip_env(os.environ.get("NYRA_UI_PORT", "8790"))))
    args = parser.parse_args()

    try:
        from nyra_ui.app import UIApp
        from nyra_ui.bridge import UIStateServer, UIStateStore
    except ImportError:
        logger.warning("[ui] disabled: pygame is not installed", exc_info=True)
        return

    store = UIStateStore()
    server = UIStateServer(store, host=args.host, port=args.port)
    server.start()

    try:
        UIApp(store).run()
    except Exception:
        logger.warning("[ui] render loop crashed", exc_info=True)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
