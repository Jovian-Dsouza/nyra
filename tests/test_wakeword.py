import unittest
from unittest.mock import Mock, patch

from orchestrator.wakeword import (
    _OpenWakeWordDetector,
    _canonical_model_key,
    _log_sounddevice_fallback,
    _resolve_model_score_key,
    _resolve_sounddevice_module,
)

class WakeWordLoggingTests(unittest.TestCase):
    def test_arecord_backend_skips_sounddevice_import(self) -> None:
        self.assertIsNone(_resolve_sounddevice_module("arecord"))

    def test_sounddevice_fallback_is_info_when_arecord_exists(self) -> None:
        with patch("orchestrator.wakeword.shutil.which", return_value="/usr/bin/arecord"):
            with self.assertLogs("orchestrator.wakeword", level="INFO") as logs:
                _log_sounddevice_fallback(RuntimeError("PortAudio library not found"))
        self.assertIn("INFO:orchestrator.wakeword:sounddevice unavailable", logs.output[0])

    def test_sounddevice_fallback_is_warning_when_arecord_missing(self) -> None:
        with patch("orchestrator.wakeword.shutil.which", return_value=None):
            with self.assertLogs("orchestrator.wakeword", level="WARNING") as logs:
                _log_sounddevice_fallback(RuntimeError("PortAudio library not found"))
        self.assertIn("WARNING:orchestrator.wakeword:sounddevice unavailable", logs.output[0])
        self.assertIn("arecord` is also unavailable", logs.output[0])

    def test_model_score_key_matches_versioned_output(self) -> None:
        scores = {"alexa_v0.1": 0.9}
        self.assertEqual(_resolve_model_score_key("alexa", scores), "alexa_v0.1")
        self.assertEqual(_canonical_model_key("alexa_v0.1"), "alexa")

    def test_detector_uses_versioned_output_without_warning(self) -> None:
        detector = _OpenWakeWordDetector(
            model=Mock(),
            np_module=Mock(),
            sd_module=None,
            model_key="alexa",
            threshold=0.5,
            sample_rate=16000,
            frame_samples=1280,
            cooldown_s=1.0,
            input_device=None,
            arecord_device=None,
        )
        score = detector._select_score({"alexa_v0.1": 0.75})
        self.assertEqual(score, 0.75)

    def test_stop_capture_terminates_active_arecord_process(self) -> None:
        detector = _OpenWakeWordDetector(
            model=Mock(),
            np_module=Mock(),
            sd_module=None,
            model_key="alexa",
            threshold=0.5,
            sample_rate=16000,
            frame_samples=1280,
            cooldown_s=1.0,
            input_device=None,
            arecord_device=None,
        )
        proc = Mock()
        proc.poll.return_value = None
        detector._proc = proc

        detector.stop_capture()

        proc.terminate.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
