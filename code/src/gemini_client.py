"""Thin wrapper around the google-genai SDK: rate limiting, retry with
exponential backoff, structured (schema-constrained) JSON output, and call
usage tracking for the operational-analysis report.
"""

import json
import os
import threading
import time
from typing import List, Type

from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from . import config


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(token in msg for token in ("429", "resource_exhausted", "rate limit", "503", "unavailable", "deadline"))


class RateLimiter:
    """Simple token-bucket limiter: at most `max_per_minute` calls in any rolling 60s window."""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self._lock = threading.Lock()
        self._timestamps: List[float] = []

    def acquire(self) -> None:
        with self._lock:
            while True:
                now = time.time()
                self._timestamps = [t for t in self._timestamps if now - t < 60]
                if len(self._timestamps) < self.max_per_minute:
                    self._timestamps.append(now)
                    return
                sleep_for = 60 - (now - self._timestamps[0]) + 0.05
                time.sleep(max(sleep_for, 0.05))


class GeminiClient:
    def __init__(self, api_key: str = None, max_rpm: int = config.MAX_REQUESTS_PER_MINUTE):
        api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment or a .env file."
            )
        from google import genai  # imported lazily so unit tests can avoid the dependency

        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self._limiter = RateLimiter(max_rpm)

        self.call_count = 0
        self.total_prompt_tokens = 0
        self.total_output_tokens = 0
        self.calls_by_model: dict = {}

    def _record_usage(self, model: str, response) -> None:
        self.call_count += 1
        self.calls_by_model[model] = self.calls_by_model.get(model, 0) + 1
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            self.total_prompt_tokens += getattr(usage, "prompt_token_count", 0) or 0
            self.total_output_tokens += getattr(usage, "candidates_token_count", 0) or 0

    def generate_structured(self, model: str, contents: list, response_schema: Type[BaseModel]) -> BaseModel:
        """Runs one rate-limited, retried, structured-output call and returns a
        validated instance of `response_schema`.
        """

        @retry(
            wait=wait_exponential(multiplier=1, min=config.RETRY_MIN_WAIT_SECONDS, max=config.RETRY_MAX_WAIT_SECONDS),
            stop=stop_after_attempt(config.MAX_RETRY_ATTEMPTS),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        )
        def _call():
            self._limiter.acquire()
            types = self._genai.types
            return self._client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=config.TEMPERATURE,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )

        response = _call()
        self._record_usage(model, response)

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, response_schema):
            return parsed
        # Fall back to manual JSON parsing if the SDK didn't populate `.parsed`.
        data = json.loads(response.text)
        return response_schema.model_validate(data)

    def image_part(self, data: bytes, mime_type: str):
        return self._genai.types.Part.from_bytes(data=data, mime_type=mime_type)
