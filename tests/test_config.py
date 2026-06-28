import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from orchestrator.config import (
    Settings,
    _default_hermes_command,
    _default_piper_command,
    _resolve_piper_voice_path,
)


class HermesCommandResolutionTests(unittest.TestCase):
    def test_prefers_local_hermes_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            hermes = home / ".local/bin/hermes"
            hermes.parent.mkdir(parents=True)
            hermes.write_text("#!/bin/sh\n", encoding="utf-8")
            with patch("orchestrator.config.Path.home", return_value=home):
                self.assertEqual(_default_hermes_command(), f"{hermes} acp")

    def test_uses_path_hermes_when_local_cli_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with patch("orchestrator.config.Path.home", return_value=home):
                with patch("orchestrator.config.shutil.which", return_value="/usr/bin/hermes"):
                    self.assertEqual(_default_hermes_command(), "/usr/bin/hermes acp")

    def test_falls_back_to_module_launch_when_cli_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with patch("orchestrator.config.Path.home", return_value=home):
                with patch("orchestrator.config.shutil.which", return_value=None):
                    self.assertEqual(_default_hermes_command(), "python -m acp_adapter.entry")

    def test_env_override_wins(self) -> None:
        with patch.dict(
            os.environ,
            {"NYRA_HERMES_COMMAND": "custom hermes", "NYRA_WAKEWORD_BACKEND": "auto"},
            clear=False,
        ):
            settings = Settings.from_env()
            self.assertEqual(settings.hermes_command, "custom hermes")
            self.assertEqual(settings.wakeword_backend, "auto")

    def test_env_parses_hermes_session_idle_timeout(self) -> None:
        with patch.dict(
            os.environ,
            {"NYRA_HERMES_SESSION_IDLE_TIMEOUT_S": "42.5"},
            clear=False,
        ):
            settings = Settings.from_env()
            self.assertEqual(settings.hermes_session_idle_timeout_s, 42.5)

    def test_resolve_piper_voice_path_auto_picks_single_downloaded_voice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            voice_dir = cwd / "models/tts"
            voice_dir.mkdir(parents=True)
            voice = voice_dir / "en_US-lessac-medium.onnx"
            voice.write_text("fake", encoding="utf-8")
            old_cwd = Path.cwd()
            os.chdir(cwd)
            try:
                self.assertEqual(
                    _resolve_piper_voice_path(Path("models/tts/voice.onnx")),
                    Path("models/tts/en_US-lessac-medium.onnx"),
                )
            finally:
                os.chdir(old_cwd)

    def test_default_piper_command_prefers_venv_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "python"
            exe.write_text("", encoding="utf-8")
            piper = Path(tmp) / "piper"
            piper.write_text("", encoding="utf-8")
            with patch("orchestrator.config.sys.executable", str(exe)):
                self.assertEqual(_default_piper_command(), str(piper))


if __name__ == "__main__":
    unittest.main()
