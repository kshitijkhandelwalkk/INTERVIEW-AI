"""
Manages the Hugging Face speech-to-speech pipeline as a subprocess.

The pipeline runs in its own conda env (default: `s2s`) in `--mode websocket`,
exposing a bidirectional 16 kHz int16 PCM WebSocket plus JSON transcript events.
It is restarted per interview so each session gets its own interviewer persona
via --init_chat_prompt.
"""

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

from settings import get_settings

settings = get_settings()


def _find_s2s_python() -> str:
    if settings.S2S_PYTHON:
        return settings.S2S_PYTHON
    # Look for the conda env next to the base install.
    candidates = []
    conda = shutil.which("conda")
    if conda:
        base = Path(conda).parent.parent
        candidates.append(base / "envs" / settings.S2S_CONDA_ENV / "python.exe")
    home = Path.home()
    for root in ("miniconda3", "anaconda3", "mambaforge", "miniforge3"):
        candidates.append(home / root / "envs" / settings.S2S_CONDA_ENV / "python.exe")
    for c in candidates:
        if c.exists():
            return str(c)
    raise RuntimeError(
        f"Could not find python for conda env '{settings.S2S_CONDA_ENV}'. Set S2S_PYTHON in .env."
    )


class S2SManager:
    """Starts/stops the speech-to-speech pipeline subprocess."""

    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.current_prompt: str | None = None
        self._lock = asyncio.Lock()

    @property
    def ws_url(self) -> str:
        return f"ws://{settings.S2S_WS_HOST}:{settings.S2S_WS_PORT}"

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _build_command(self, system_prompt: str) -> list[str]:
        python = _find_s2s_python()
        cmd = [
            python,
            "-m",
            "speech_to_speech.s2s_pipeline",
            "--mode", "websocket",
            "--ws_host", settings.S2S_WS_HOST,
            "--ws_port", str(settings.S2S_WS_PORT),
            "--stt", settings.S2S_STT,
            "--tts", settings.S2S_TTS,
            "--llm_backend", "responses-api",
            "--model_name", settings.LLM_MODEL,
            "--responses_api_base_url", settings.LLM_BASE_URL,
            "--responses_api_api_key", settings.LLM_API_KEY.get_secret_value(),
            "--init_chat_role", "system",
            "--init_chat_prompt", system_prompt,
        ]
        if settings.S2S_TTS == "qwen3" and settings.S2S_TTS_REF_AUDIO:
            if Path(settings.S2S_TTS_REF_AUDIO).exists():
                # Voice cloning needs the Base model; the default CustomVoice
                # model only supports its built-in speakers.
                cmd += [
                    "--qwen3_tts_model_name", "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    "--qwen3_tts_ref_audio", settings.S2S_TTS_REF_AUDIO,
                    "--qwen3_tts_xvec_only", "true",
                ]
            else:
                print(f"[s2s] reference voice not found, using stock speaker: {settings.S2S_TTS_REF_AUDIO}")
        if settings.S2S_DEVICE:
            cmd += ["--device", settings.S2S_DEVICE]
        if settings.S2S_EXTRA_ARGS:
            cmd += settings.S2S_EXTRA_ARGS.split()
        return cmd

    async def start(self, system_prompt: str) -> None:
        """(Re)start the pipeline with the given interviewer persona."""
        async with self._lock:
            if self.is_running() and self.current_prompt == system_prompt:
                return
            self._stop_locked()
            self._kill_stale_port_owner()
            env = os.environ.copy()
            # The user-site regex package shadows the env's; keep user site and
            # any machine-wide PYTHONPATH (e.g. gstreamer's) out of the env.
            env["PYTHONNOUSERSITE"] = "1"
            env.pop("PYTHONPATH", None)
            env["PYTHONUNBUFFERED"] = "1"
            cmd = self._build_command(system_prompt)
            print("[s2s] launching:", " ".join(cmd[:6]), "...")
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=sys.stdout,
                stderr=sys.stderr,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            self.current_prompt = system_prompt

    def _kill_stale_port_owner(self) -> None:
        """
        Kill any orphaned pipeline still holding the WebSocket port (left over
        from a force-killed server). Otherwise the new pipeline can't bind and
        the interview would silently talk to the old persona.
        """
        if os.name != "nt":
            return
        try:
            out = subprocess.run(
                [
                    "powershell", "-NoProfile", "-Command",
                    f"(Get-NetTCPConnection -LocalPort {settings.S2S_WS_PORT} -State Listen "
                    "-ErrorAction SilentlyContinue).OwningProcess",
                ],
                capture_output=True, text=True, timeout=15,
            ).stdout.split()
            for pid in {p for p in out if p.isdigit()}:
                print(f"[s2s] killing stale process {pid} on port {settings.S2S_WS_PORT}")
                subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True, timeout=15)
        except Exception as e:
            print(f"[s2s] stale-port check failed: {e}")

    async def wait_ready(self, timeout: float = 300.0) -> bool:
        """Poll until the WebSocket port accepts connections (model load can be slow)."""
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if not self.is_running():
                return False
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(settings.S2S_WS_HOST, settings.S2S_WS_PORT), timeout=2
                )
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
                return True
            except Exception:
                await asyncio.sleep(1.0)
        return False

    def _stop_locked(self) -> None:
        if self.process is not None and self.process.poll() is None:
            print("[s2s] stopping pipeline")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        self.current_prompt = None

    async def stop(self) -> None:
        async with self._lock:
            self._stop_locked()


manager = S2SManager()
