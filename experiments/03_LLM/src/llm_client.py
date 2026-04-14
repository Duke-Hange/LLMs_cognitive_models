from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen


def _sanitize_model_name(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model).strip("_")


@dataclass
class GenerationResult:
    text: str
    latency_seconds: float
    from_cache: bool
    raw_payload: Dict[str, Any]


class OllamaClient:
    """Small Ollama client with deterministic file cache."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: int = 120,
        temperature: float = 0.0,
        num_predict: int = 32,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = int(timeout_seconds)
        self.temperature = float(temperature)
        self.num_predict = int(num_predict)
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, cache_key: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        model_dir = self.cache_dir / _sanitize_model_name(self.model)
        model_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
        return model_dir / f"{digest}.json"

    def _load_cache(self, cache_key: str) -> Optional[GenerationResult]:
        path = self._cache_path(cache_key)
        if path is None or not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
            if not raw.strip():
                path.unlink(missing_ok=True)
                return None
            payload = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        if not isinstance(payload, dict):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        return GenerationResult(
            text=str(payload.get("text", "")),
            latency_seconds=float(payload.get("latency_seconds", 0.0)),
            from_cache=True,
            raw_payload=payload.get("raw_payload", {}),
        )

    def _save_cache(self, cache_key: str, result: GenerationResult) -> None:
        path = self._cache_path(cache_key)
        if path is None:
            return
        payload = {
            "text": result.text,
            "latency_seconds": result.latency_seconds,
            "raw_payload": result.raw_payload,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def generate(self, prompt: str, cache_key: Optional[str] = None) -> GenerationResult:
        if cache_key:
            hit = self._load_cache(cache_key)
            if hit is not None:
                return hit

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            url=f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        started = time.time()
        try:
            with urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e.code} {e.reason}") from e
        except URLError as e:
            raise RuntimeError(f"Ollama connection error: {e.reason}") from e
        elapsed = time.time() - started

        data = json.loads(raw)
        result = GenerationResult(
            text=str(data.get("response", "")).strip(),
            latency_seconds=elapsed,
            from_cache=False,
            raw_payload=data,
        )
        if cache_key:
            self._save_cache(cache_key, result)
        return result
