"""Standalone visual-iteration entrypoint: `python -m nyra_ui.demo`.

Runs the real `UIApp`/renderer on this process's true main thread, fed by a
`FakeStateDriver` on a background thread — no LiveKit agent, socket, or
credentials involved. Use this to iterate on the window's look and feel.
"""

import logging

logging.basicConfig(level=logging.INFO)


def main() -> None:
    from nyra_ui.app import UIApp
    from nyra_ui.bridge import UIStateStore
    from nyra_ui.driver_fake import FakeStateDriver

    store = UIStateStore()
    driver = FakeStateDriver(store)
    driver.start()
    try:
        UIApp(store).run()
    finally:
        driver.stop()


if __name__ == "__main__":
    main()
