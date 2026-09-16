"""Thin wrapper over the Anthropic Messages API.

Everything the rest of the app needs is here, so model choices and failure
handling live in one place rather than being scattered through the pipeline.
"""
import json
import logging
from typing import Any, Dict, Iterator, List, Optional

from . import config

log = logging.getLogger(__name__)

# Models without the `effort` parameter -- passing it returns a 400.
_NO_EFFORT_MODELS = ("claude-haiku-4-5",)


class LLMUnavailable(RuntimeError):
    """The model could not be reached, or declined to answer."""


def describe_status_error(status_code):
    # type: (int) -> str
    """Turn an API status code into a sentence that says what to fix.

    Shown to the user, and read by whoever deployed the app. "HTTP 401" is
    honest but useless at 11pm; "the API key was rejected" is actionable.
    """
    if status_code == 401:
        return ("The language model service rejected the API key. Check "
                "ANTHROPIC_API_KEY in the deployment's secrets (or .env locally).")
    if status_code == 403:
        return ("The language model service refused this request (HTTP 403). "
                "The API key may lack access to the configured model.")
    if status_code == 404:
        return ("The configured model was not found (HTTP 404). Check "
                "SQL_MODEL / GUARDRAIL_MODEL.")
    if status_code == 402 or status_code == 400:
        return ("The language model service rejected the request (HTTP %s). "
                "The account may be out of credit or the request malformed."
                % status_code)
    if status_code == 529 or status_code >= 500:
        return ("The language model service is overloaded or down (HTTP %s). "
                "Please retry in a moment." % status_code)
    return "The language model service returned an error (HTTP %s)." % status_code


class LLMClient(object):
    def __init__(self, api_key=None):
        # type: (Optional[str]) -> None
        import anthropic
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(
            api_key=api_key or config.anthropic_api_key(),
            timeout=float(config.LLM_TIMEOUT_SECONDS),
            max_retries=1,
        )
        # Token usage per model since construction, for cost reporting.
        self.usage = {}  # type: Dict[str, Dict[str, int]]

    def _record_usage(self, model, response):
        # type: (str, Any) -> None
        u = getattr(response, "usage", None)
        if u is None:
            return
        tally = self.usage.setdefault(model, {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0})
        tally["input"] += getattr(u, "input_tokens", 0) or 0
        tally["output"] += getattr(u, "output_tokens", 0) or 0
        tally["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
        tally["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0

    # --- internals ----------------------------------------------------------

    def _output_config(self, model, effort=None, schema=None):
        # type: (str, Optional[str], Optional[dict]) -> Dict[str, Any]
        cfg = {}  # type: Dict[str, Any]
        if effort and not model.startswith(_NO_EFFORT_MODELS):
            cfg["effort"] = effort
        if schema is not None:
            cfg["format"] = {"type": "json_schema", "schema": schema}
        return cfg

    @staticmethod
    def _system_blocks(system):
        # type: (str) -> List[Dict[str, Any]]
        """Cache the system prompt.

        The schema context is large and largely stable across turns, so a
        cache breakpoint here is the difference between paying full price for
        it every question and paying ~10%.
        """
        return [{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }]

    @staticmethod
    def _first_text(response):
        # type: (Any) -> str
        """Pull the answer text out of the response.

        Opus 5 thinks by default, so `content` contains thinking blocks whose
        text is empty. Indexing content[0] would return nothing -- iterate and
        match on type instead.
        """
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

    def _check_refusal(self, response):
        # type: (Any) -> None
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMUnavailable(
                "The model declined to answer this request%s."
                % (" (%s)" % category if category else "")
            )

    def _call(self, **kwargs):
        # type: (Any) -> Any
        a = self._anthropic
        try:
            response = self._client.messages.create(**kwargs)
            self._record_usage(kwargs.get("model", "?"), response)
            return response
        except a.RateLimitError as exc:
            raise LLMUnavailable(
                "The assistant is rate limited right now. Please retry in a moment."
            ) from exc
        except a.APITimeoutError as exc:
            raise LLMUnavailable(
                "The language model took too long to respond. Please try again."
            ) from exc
        except a.APIConnectionError as exc:
            raise LLMUnavailable(
                "Could not reach the language model service. Check network connectivity."
            ) from exc
        except a.APIStatusError as exc:
            log.exception("Anthropic API error")
            raise LLMUnavailable(describe_status_error(exc.status_code)) from exc

    # --- public API ---------------------------------------------------------

    def structured(self, system, user, schema, model=None, max_tokens=8000, effort=None):
        # type: (str, str, dict, Optional[str], int, Optional[str]) -> Dict[str, Any]
        """Call the model and get back a dict matching `schema`.

        Uses structured outputs so the response is schema-valid by
        construction -- no regex extraction, no retry-on-parse loop.
        """
        model = model or config.SQL_MODEL
        response = self._call(
            model=model,
            max_tokens=max_tokens,
            system=self._system_blocks(system),
            messages=[{"role": "user", "content": user}],
            output_config=self._output_config(model, effort, schema),
        )
        self._check_refusal(response)
        raw = self._first_text(response)
        if not raw.strip():
            raise LLMUnavailable("The model returned an empty response.")
        try:
            return json.loads(raw)
        except ValueError as exc:
            raise LLMUnavailable("The model returned malformed JSON.") from exc

    def stream_text(self, system, messages, model=None, max_tokens=4000, effort=None):
        # type: (str, List[dict], Optional[str], int, Optional[str]) -> Iterator[str]
        """Stream a prose answer token by token.

        Streaming is a requirement, not a nicety: the assignment caps responses
        at 60 seconds, and visible progress is what keeps a reviewer from
        assuming the app hung.
        """
        model = model or config.SQL_MODEL
        a = self._anthropic
        try:
            with self._client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=self._system_blocks(system),
                messages=messages,
                output_config=self._output_config(model, effort),
            ) as stream:
                for text in stream.text_stream:
                    yield text
                final = stream.get_final_message()
                self._record_usage(model, final)
                self._check_refusal(final)
        except a.APIStatusError as exc:
            log.exception("Anthropic streaming error")
            raise LLMUnavailable(describe_status_error(exc.status_code)) from exc
        except a.APITimeoutError as exc:
            raise LLMUnavailable(
                "The language model took too long to respond. Please try again."
            ) from exc
        except a.APIConnectionError as exc:
            raise LLMUnavailable("Could not reach the language model service.") from exc
