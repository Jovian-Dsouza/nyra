# Nyra

Local, streaming, offline-first voice assistant for Raspberry Pi 5. See [PRD.md](PRD.md) for product requirements.

## Current implementation status

This repository now contains a runnable v1 orchestrator scaffold:
- `orchestrator.main` service entrypoint with state machine + wake/STT/Hermes/TTS wiring
- ACP streaming client (`orchestrator.hermes_client.ACPHermesClient`)
- Sentence chunking + ordered TTS queue with barge-in interruption support
- systemd service unit and setup script
- unit tests for state transitions, sentence aggregation, and TTS queue behavior

The real audio model backends (openWakeWord + sherpa-onnx + Piper command wiring) are represented behind stable interfaces and can be swapped in without changing orchestrator control flow.

## Hardware bring-up

### Bluetooth speaker (Echo Dot)

Verified on device with Echo Dot-LQH (`34:AF:B3:8C:F0:26`).

| Setting | Value |
|---------|-------|
| PipeWire sink | `bluez_output.34_AF_B3_8C_F0_26.1` |
| Card | `bluez_card.34_AF_B3_8C_F0_26` |
| Profile | `a2dp-sink` |

Use the helper script:

```bash
./scripts/setup-bt-audio.sh
paplay /usr/share/sounds/alsa/Front_Center.wav
wpctl status
```

### I2S microphone (hard gate)

Nyra startup validates capture availability with `arecord -l`. If no capture device is detected, service startup fails (unless `NYRA_REQUIRE_AUDIO=0` is set for development).

Enable and verify before running:
```bash
arecord -l
```

## Setup

```bash
./scripts/setup.sh
```

This creates `.venv`, installs Python dependencies from `requirements.txt`, and performs an audio capture check.
To install openWakeWord runtime deps in the same step:

```bash
NYRA_INSTALL_VOICE_DEPS=1 ./scripts/setup.sh
```

To install STT runtime deps:

```bash
NYRA_INSTALL_STT_DEPS=1 ./scripts/setup.sh
```

Run development console mode:

```bash
.venv/bin/python -m orchestrator.main --dev-console
```

Console commands:
- `wake <prompt>`: simulate wake + final transcript
- `wake`: simulate wake only
- `final <prompt>`: submit final transcript while listening
- `quit`: exit

## Service install

System-wide install example:

```bash
sudo cp scripts/nyra.service /etc/systemd/system/nyra.service
sudo systemctl daemon-reload
sudo systemctl enable --now nyra.service
sudo systemctl status nyra.service
```

## Configuration

Environment variables:
- `NYRA_HERMES_COMMAND` (default: auto-detect `~/.local/bin/hermes acp`, then `hermes acp`, then `~/.hermes/hermes-agent/venv/bin/hermes acp`, else fallback to `python -m acp_adapter.entry`)
- `NYRA_PIPER_COMMAND` (default: `piper`)
- `NYRA_PIPER_VOICE` (default: `models/tts/voice.onnx`)
- `NYRA_TTS_PLAY_COMMAND` (optional; override playback command, e.g. `paplay` or `aplay -q`)
- `NYRA_WAKEWORD_MODEL` (default: `models/wakeword/hey_nyra.onnx`)
- `NYRA_WAKEWORD_BACKEND` (default: `arecord`; options: `arecord`, `auto`, `sounddevice`)
- `NYRA_WAKEWORD_MODEL_KEY` (optional; output key to use from model scores)
- `NYRA_WAKEWORD_THRESHOLD` (default: `0.5`)
- `NYRA_WAKEWORD_SAMPLE_RATE` (default: `16000`)
- `NYRA_WAKEWORD_FRAME_SAMPLES` (default: `1280`)
- `NYRA_WAKEWORD_COOLDOWN_S` (default: `1.2`)
- `NYRA_WAKEWORD_INPUT_DEVICE` (optional ALSA device index for `sounddevice`)
- `NYRA_WAKEWORD_ARECORD_DEVICE` (optional ALSA device string, e.g. `hw:0,0`, used when falling back to `arecord`)
- `NYRA_STT_MODEL` (default: `models/stt/vosk-model-small-en-us-0.15`)
- `NYRA_STT_MODEL_URL` (default: Vosk small English ZIP)
- `NYRA_STT_SAMPLE_RATE` (default: `16000`)
- `NYRA_STT_CHUNK_BYTES` (default: `4000`)
- `NYRA_STT_ARECORD_DEVICE` (optional ALSA device string for STT capture, e.g. `hw:0,0`)
- `NYRA_REQUIRE_AUDIO` (default: `true`)
- `NYRA_LOG_LEVEL` (default: `INFO`)

If `NYRA_WAKEWORD_MODEL` does not exist, Nyra now falls back to built-in `openWakeWord` models.  
Set `NYRA_WAKEWORD_MODEL_KEY` to the specific built-in key you want to use (`alexa`, `hey_jarvis`, `hey_mycroft`, `hey_rhasspy`, `timer`, `weather`).
If the model file is missing, Nyra will auto-download the selected built-in ONNX model to `models/wakeword/` on first run.

Nyra uses ALSA `arecord` for wake-word capture by default on Raspberry Pi.
Set `NYRA_WAKEWORD_BACKEND=auto` to try `sounddevice` first with automatic fallback, or `NYRA_WAKEWORD_BACKEND=sounddevice` to require PortAudio explicitly.
If you see `PortAudio library not found`, Nyra can still fall back to ALSA capture via `arecord`.

TTS now prefers Piper when both `NYRA_PIPER_COMMAND` and `NYRA_PIPER_VOICE` are usable, then plays the synthesized WAV with `aplay` by default. If Piper or the voice model is missing, Nyra logs a warning and falls back to local CLI speech (`espeak-ng`, `espeak`, or `spd-say`), then to the placeholder timing-only path as a last resort.
If `-D hw:X,Y` fails, Nyra retries with `plughw:X,Y` and then default input automatically.

STT now uses Vosk streaming recognition and prints live partial/final text in logs:
- `stt partial: ...`
- `stt final: ...`

For STT, `NYRA_STT_MODEL` must point to a Vosk model directory (not a single ONNX file). If that directory is missing, Nyra will auto-download the ZIP from `NYRA_STT_MODEL_URL` and unpack it into `models/stt/` on first use.

## Testing

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## Project layout

```text
orchestrator/   # asyncio service + pipeline components
models/         # wakeword/stt/tts model artifacts
scripts/        # setup and systemd helpers
tests/          # unit tests
```
