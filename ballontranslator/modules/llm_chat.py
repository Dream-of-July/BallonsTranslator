"""Shared request dispatch for Chat Completions and API/subscription Responses."""

from __future__ import annotations

import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import httpx

from .context.errors import provider_error_message
from .context.token_usage import format_completion_token_usage
from .exceptions import (
    LLMApiKeyRequiredError,
    LLMAuthenticationError,
    LLMOutputLimitError,
    LLMRequestStopped,
    LLMUserActionRequiredError,
)
from ballontranslator.utils.llm_profiles import (
    LLMProfile,
    PROVIDER_DEFAULTS,
    THINKING_AUTO,
    THINKING_DISABLED,
    normalize_thinking_level,
    resolve_api_key,
)

if TYPE_CHECKING:
    from .codex import CodexChatSession, CodexTurnState
    from .openai_responses import OpenAIResponsesSession

OPENAI_MAX_TOKENS_MODELS = frozenset({
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4o",
    "gpt-4o-mini",
})
_CHAT_COMPLETIONS_SUFFIX = '/chat/completions'


def _normalized_base_url(url: str) -> str:
    return str(url or '').strip().rstrip('/')


def _openai_sdk_base_url(url: str) -> str:
    """Return the API base expected by the OpenAI SDK chat resource.

    Provider documentation often presents the complete HTTP endpoint, while
    ``client.chat.completions.create`` adds that endpoint path itself.

    >>> _openai_sdk_base_url('https://example.test/v1/chat/completions/')
    'https://example.test/v1'
    >>> _openai_sdk_base_url('https://example.test/v1')
    'https://example.test/v1'
    """
    base_url = str(url or '').strip()
    endpoint_url = base_url.rstrip('/')
    if endpoint_url.endswith(_CHAT_COMPLETIONS_SUFFIX):
        parent_url = endpoint_url[:-len(_CHAT_COMPLETIONS_SUFFIX)]
        if parent_url:
            return parent_url
    return base_url


class LLMConnectionError(RuntimeError):
    """A connection-check failure with a safe message and a stable error kind."""

    def __init__(self, kind: str, message: str) -> None:
        self.kind = kind
        super().__init__(message)


@dataclass(frozen=True)
class LLMConnectionTestResult:
    ok: bool
    message: str
    kind: str
    models: List[str]


def _profile_connection_request(
    profile: LLMProfile,
    path: str,
    *,
    proxy: str,
    timeout: float,
    payload: Optional[Dict[str, Any]] = None,
) -> httpx.Response:
    api_key = resolve_api_key(profile).strip()
    if profile.require_api_key and not api_key:
        raise LLMApiKeyRequiredError(profile.id, profile.name)
    headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
    base_url = _openai_sdk_base_url(profile.base_url) or PROVIDER_DEFAULTS['OpenAI']['base_url']
    url = f'{base_url.rstrip("/")}/{path}'
    try:
        # A single proxy applies to both HTTP and HTTPS, as in the requester.
        with httpx.Client(proxy=proxy or None, timeout=timeout) as client:
            if payload is None:
                response = client.get(url, headers=headers)
            else:
                response = client.post(url, headers=headers, json=payload)
    except httpx.TimeoutException:
        raise LLMConnectionError('timeout', 'The connection timed out. Try again.') from None
    except (httpx.RequestError, httpx.InvalidURL, ValueError, ImportError):
        # Transport diagnostics can echo credentials or proxy URLs; never forward them.
        raise LLMConnectionError('network', 'Could not connect. Check the base URL, network, and proxy.') from None
    if not response.is_success:
        status = response.status_code
        if status in (401, 403):
            kind, message = 'auth', 'Authentication failed. Check the API key.'
        elif status == 404:
            kind, message = 'not_found', 'Endpoint not found. Check the base URL or enter a model manually.'
        elif status == 429:
            kind, message = 'rate_limited', 'The provider rate limit was reached. Try again later.'
        elif 500 <= status < 600:
            kind, message = 'server', 'The provider has a server error. Try again later.'
        else:
            kind, message = 'invalid_response', 'The provider returned an unexpected response.'
        raise LLMConnectionError(kind, f'HTTP {status}: {message}')
    return response


def list_profile_models(
    profile: LLMProfile,
    *,
    proxy: str = '',
    timeout: float = 15.0,
) -> List[str]:
    """Fetch model IDs with a blocking request; run off the Qt main thread.

    Missing credentials are checked before creating any network client.

    >>> try:
    ...     list_profile_models(LLMProfile())
    ... except LLMApiKeyRequiredError:
    ...     print('API key required')
    API key required
    """
    response = _profile_connection_request(profile, 'models', proxy=proxy, timeout=timeout)
    try:
        entries = response.json()
    except ValueError:
        entries = None
    if isinstance(entries, dict):
        entries = entries.get('data', entries.get('models'))
    if not isinstance(entries, list):
        raise LLMConnectionError(
            'invalid_response',
            f'HTTP {response.status_code}: The provider did not return a model list.',
        )
    models: List[str] = []
    seen = set()
    for entry in entries:
        if isinstance(entry, dict):
            model = entry.get('id')
            if not isinstance(model, str) or not model.strip():
                model = entry.get('name')
        else:
            model = entry
        if isinstance(model, str) and model.strip():
            model = model.strip()
            if model not in seen:
                models.append(model)
                seen.add(model)
    return models


def check_profile_connection(
    profile: LLMProfile,
    *,
    proxy: str = '',
    timeout: float = 15.0,
) -> LLMConnectionTestResult:
    """Check access with blocking requests; run off the Qt main thread.

    If model listing is unavailable or rejects the key, try one minimal chat
    request when a model is configured: some providers (e.g. Anthropic's
    OpenAI-compatible endpoint) authenticate chat and model listing differently,
    so the chat result is the authoritative one. Expected connection failures
    are returned, not raised.

    >>> check_profile_connection(LLMProfile()).kind
    'auth'
    """
    try:
        models = list_profile_models(profile, proxy=proxy, timeout=timeout)
    except LLMApiKeyRequiredError:
        return LLMConnectionTestResult(False, 'API key is missing. Enter an API key.', 'auth', [])
    except LLMConnectionError as error:
        if error.kind not in ('not_found', 'invalid_response', 'auth'):
            return LLMConnectionTestResult(False, str(error), error.kind, [])
        if not profile.model.strip():
            return LLMConnectionTestResult(
                False,
                'This provider does not list models. Enter a model name manually.',
                error.kind,
                [],
            )
    else:
        return LLMConnectionTestResult(True, 'Connection successful.', 'ok', models)
    try:
        _profile_connection_request(
            profile, 'chat/completions', proxy=proxy, timeout=timeout,
            payload={
                'model': profile.model.strip(),
                'messages': [{'role': 'user', 'content': 'Hi'}],
                'max_tokens': 1,
            },
        )
    except LLMApiKeyRequiredError:
        return LLMConnectionTestResult(False, 'API key is missing. Enter an API key.', 'auth', [])
    except LLMConnectionError as error:
        return LLMConnectionTestResult(False, str(error), error.kind, [])
    return LLMConnectionTestResult(True, 'Connection successful.', 'ok', [])


def _uses_provider_base_url(base_url: str, provider: str) -> bool:
    provider_url = _normalized_base_url(
        PROVIDER_DEFAULTS[provider]['base_url']
    ).lower()
    base_url = _normalized_base_url(base_url).lower()
    return base_url == provider_url or base_url.startswith(
        f'{provider_url}/'
    )


def gpt_model_version(model: str) -> Optional[Tuple[int, int]]:
    """Recognize GPT IDs, including gateway namespaces and model suffixes.

    >>> gpt_model_version('openai/gpt-5.6-luna')
    (5, 6)
    >>> gpt_model_version('gpt-4o-mini')
    (4, 0)
    >>> gpt_model_version('custom-gpt-6') is None
    True
    """
    name = str(model or '').strip().rsplit('/', 1)[-1].lower()
    match = re.match(r'^gpt-(\d+)(?:\.(\d+)|o)?(?:[-:]|$)', name)
    return (int(match[1]), int(match[2] or 0)) if match else None


def openai_chat_completion_args(
    profile: LLMProfile,
    model: str,
) -> Dict[str, Any]:
    """Map provider-neutral profile values to OpenAI chat API arguments.

    Native OpenAI models default to ``max_completion_tokens`` so future model
    names do not need version-pattern guesses. Only explicitly listed older
    models and compatibility endpoints retain ``max_tokens``. GPT-5.5+ models
    omit temperature and top_p regardless of the endpoint, including gateways.

    >>> profile = LLMProfile.from_provider('OpenAI')
    >>> openai_chat_completion_args(profile, 'gpt-5.5')
    {'max_completion_tokens': 8192}
    >>> openai_chat_completion_args(profile, 'gpt-4o')['temperature']
    0.1
    """
    if profile.backend == 'codex':
        return {}
    base_url = _normalized_base_url(_openai_sdk_base_url(profile.base_url))
    openai_base_url = _normalized_base_url(
        PROVIDER_DEFAULTS['OpenAI']['base_url']
    )
    model_name = str(model or '').strip().rsplit('/', 1)[-1].lower()
    is_native_openai = not base_url or base_url == openai_base_url
    args: Dict[str, Any] = {}
    gpt_version = gpt_model_version(model)
    if gpt_version is None or gpt_version < (5, 5):
        args['top_p'] = float(profile.top_p)
        args['temperature'] = float(profile.temperature)
    token_limit_key = (
        'max_completion_tokens'
        if is_native_openai and model_name not in OPENAI_MAX_TOKENS_MODELS
        else 'max_tokens'
    )
    args[token_limit_key] = int(profile.max_tokens)

    thinking_level = normalize_thinking_level(profile.thinking_level)
    if thinking_level == THINKING_AUTO:
        return args
    if thinking_level != THINKING_DISABLED:
        args['reasoning_effort'] = thinking_level
        return args
    if _uses_provider_base_url(base_url, 'DeepSeek'):
        args['extra_body'] = {'thinking': {'type': 'disabled'}}
    elif _uses_provider_base_url(base_url, 'OpenRouter'):
        args['extra_body'] = {'reasoning': {'effort': 'none'}}
    else:
        args['reasoning_effort'] = 'none'
    return args


def openai_json_response_format(
    profile: LLMProfile,
    name: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the profile-compatible JSON response format.

    >>> profile = LLMProfile.from_provider('LM Studio')
    >>> openai_json_response_format(profile, 'demo', {'type': 'object'})['type']
    'json_schema'
    >>> profile.json_schema_response_format = False
    >>> openai_json_response_format(profile, 'demo', {})
    {'type': 'json_object'}
    """
    if not profile.json_schema_response_format:
        return {'type': 'json_object'}
    return {
        'type': 'json_schema',
        'json_schema': {
            'name': name,
            'strict': True,
            'schema': schema,
        },
    }


@dataclass(frozen=True)
class LLMChatResult:
    content: str
    usage: Any = None
    finish_reason: str = ''
    prompt_cache_diagnostics: object = None
    codex_response_items: Tuple[Dict[str, object], ...] = ()
    codex_cache_key: str = ''
    codex_account_generation: Optional[int] = None


class LLMChatRequestError(RuntimeError):
    """A normalized provider status error from Chat Completions."""

    def __init__(self, provider_error: Exception) -> None:
        self.provider_error = provider_error
        super().__init__(provider_error_message(provider_error))


class LLMChatRequester:
    """Issue one profile-backed API or Codex chat request.

    Prompt construction and retries stay with the owning Translator or OCR
    module; this boundary owns only transport and provider normalization.

    >>> LLMChatRequester().client is None
    True
    """

    dummy_api_key = 'dummy-key'

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.client: Any = None
        self.client_cache_key: Optional[
            Tuple[str, Optional[str], str, float]
        ] = None
        self.last_request_time = 0.0
        self.request_count_minute = 0
        self.minute_start_time = time.monotonic()
        self.stop_event: Optional[threading.Event] = None
        # Feature owners may share a budget across recovery attempts without
        # changing request_chat_completion's override contract. Keep concurrent
        # callers on this requester isolated as well as separate instances.
        self._request_deadline = threading.local()
        self._codex_sessions: Dict[Tuple[str, str, int], CodexChatSession] = {}
        self._openai_session: Optional[OpenAIResponsesSession] = None
        self._openai_session_identity: Optional[Tuple] = None

    def set_stop_event(
        self,
        stop_event: Optional[threading.Event],
    ) -> None:
        if stop_event is not self.stop_event:
            for session in self._codex_sessions.values():
                session.close()
            self._codex_sessions.clear()
            if self._openai_session is not None:
                self._openai_session.close()
                self._openai_session = None
        self.stop_event = stop_event

    def _wait(self, seconds: float) -> None:
        if not math.isfinite(seconds):
            raise ValueError('LLM wait duration must be finite.')
        if seconds <= 0:
            return
        if self.stop_event is not None:
            if self.stop_event.wait(seconds):
                raise LLMRequestStopped()
            return
        time.sleep(seconds)

    @staticmethod
    def _openai_module() -> Any:
        import openai  # type: ignore

        return openai

    def _http_client(self, proxy: str) -> Any:
        import httpx  # type: ignore

        seconds = self._request_timeout()
        timeout = httpx.Timeout(seconds, connect=min(10.0, seconds))
        if not proxy:
            return httpx.Client(timeout=timeout)
        try:
            mounts = {
                'http://': httpx.HTTPTransport(proxy=proxy),
                'https://': httpx.HTTPTransport(proxy=proxy),
            }
            return httpx.Client(mounts=mounts, timeout=timeout)
        except Exception as error:
            self.logger.error(
                f"Failed to initialize proxy '{proxy}': {error}. "
                'Proceeding without proxy.'
            )
            return httpx.Client(timeout=timeout)

    def _request_timeout(self) -> float:
        """Use the request timeout as a budget, defaulting to 120 seconds.

        >>> LLMChatRequester()._request_timeout()
        120.0
        """
        try:
            seconds = float(self.get_param_value('request timeout'))
        except (AttributeError, AssertionError, KeyError, TypeError, ValueError, OverflowError):
            return 120.0
        return seconds if math.isfinite(seconds) and seconds > 0 else 120.0

    def _bounded_request(
        self,
        request: Callable[[threading.Event], Any],
        close: Callable[[], None],
        stop_event: Optional[threading.Event],
        deadline: float,
    ) -> Any:
        """Bound blocking IO, including trickling streams and custom transports.

        HTTPX timeouts bound individual socket operations, not the whole response.
        An abandoned operation must never write results back into the translator
        or close a replacement transport. Transport close also stays off the
        caller because native IO and session teardown can themselves block.

        >>> LLMChatRequester()._bounded_request(
        ...     lambda cancel: 'ok', lambda: None, None, time.monotonic() + 1)
        'ok'
        """
        self._check_request_current(stop_event)
        if time.monotonic() >= deadline:
            raise TimeoutError('The LLM request exceeded its time limit.')
        done, cancelled = threading.Event(), threading.Event()
        result: List[Any] = []
        errors: List[BaseException] = []

        def run() -> None:
            try:
                result.append(request(cancelled))
            except BaseException as error:
                errors.append(error)
            finally:
                done.set()

        def cleanup() -> None:
            try:
                close()
            except Exception:
                # Transport diagnostics can contain credentials or proxy URLs.
                self.logger.warning('Could not close the abandoned LLM transport.')

        threading.Thread(target=run, daemon=True, name='LLM request').start()
        try:
            while True:
                self._check_request_current(stop_event)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('The LLM request exceeded its time limit.')
                if done.wait(min(0.05, remaining)):
                    self._check_request_current(stop_event)
                    break
        except BaseException as error:
            cancelled.set()
            # Close the captured transport now, even if a peer keeps trickling
            # bytes forever. No retry may reuse this abandoned client/session.
            threading.Thread(target=cleanup, daemon=True, name='LLM request cleanup').start()
            # Existing async transports normally finish cancellation and dispose
            # their loop within a poll. Preserve that lifecycle when possible,
            # without allowing broken IO/teardown to delay Stop indefinitely.
            if isinstance(error, LLMRequestStopped):
                done.wait(0.2)
            raise
        if errors:
            raise errors[0]
        return result[0]

    @staticmethod
    def _api_key_for_profile(profile: LLMProfile) -> str:
        api_key = resolve_api_key(profile).strip()
        if profile.require_api_key and not api_key:
            raise LLMApiKeyRequiredError(profile.id, profile.name)
        return api_key

    def _client_api_key_for_profile(self, profile: LLMProfile) -> str:
        api_key = self._api_key_for_profile(profile)
        if not api_key:
            self.logger.debug(
                f'LLM profile "{profile.name or profile.id}" does not require '
                'an API key; using a dummy API key for OpenAI-compatible '
                'client initialization.'
            )
            return self.dummy_api_key
        return api_key

    def _initialize_client(self, profile: LLMProfile) -> Any:
        api_key = self._client_api_key_for_profile(profile)
        configured_base_url = str(profile.base_url or '').strip()
        base_url = _openai_sdk_base_url(configured_base_url) or None
        proxy = str(self.get_param_value('proxy') or '')
        timeout = self._request_timeout()
        cache_key = (api_key, base_url, proxy, timeout)
        if self.client is not None and self.client_cache_key == cache_key:
            return self.client

        if configured_base_url and base_url != configured_base_url:
            self.logger.warning(
                f'LLM profile "{profile.name or profile.id}" Base URL ends '
                f'with {_CHAT_COMPLETIONS_SUFFIX}; using its parent URL as '
                'the OpenAI-compatible API base.'
            )

        openai = self._openai_module()
        previous_client = self.client
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=self._http_client(proxy),
            timeout=httpx.Timeout(timeout, connect=min(10.0, timeout)),
            # Translation owns the finite retry budget; do not multiply it in the SDK.
            max_retries=0,
        )
        self.client_cache_key = cache_key
        if previous_client is not None:
            previous_client.close()
        return self.client

    def _respect_delay(self) -> None:
        current_time = time.monotonic()
        rpm = self.get_param_value('max requests per minute')
        delay = self.get_param_value('delay')
        if rpm > 0:
            if current_time - self.minute_start_time >= 60:
                self.request_count_minute = 0
                self.minute_start_time = current_time
            if self.request_count_minute >= rpm:
                wait_time = 60.1 - (current_time - self.minute_start_time)
                if wait_time > 0:
                    self.logger.warning(
                        f'Global RPM limit ({rpm}) reached. Waiting '
                        f'{wait_time:.2f} seconds.'
                    )
                    self._wait(wait_time)
                self.request_count_minute = 0
                self.minute_start_time = time.monotonic()

        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < delay:
            self._wait(delay - time_since_last_request)

        self.last_request_time = time.monotonic()
        self.request_count_minute += 1

    def _codex_websocket_enabled(self) -> bool:
        return os.environ.get('BALLOONTRANS_CODEX_WEBSOCKET') != '0'

    def _check_request_current(self, stop_event: Optional[threading.Event]) -> None:
        if stop_event is not self.stop_event or (stop_event is not None and stop_event.is_set()):
            raise LLMRequestStopped()

    def request_chat_completion(
        self,
        profile: LLMProfile,
        api_args: Dict[str, Any],
        *,
        codex_turn: Optional[CodexTurnState] = None,
    ) -> LLMChatResult:
        """Perform one request, optionally sharing a deadline across owner retries."""
        # Throttling and client/session setup can overlap a run change. Never
        # adopt the replacement run's token for this request's existing inputs.
        stop_event = self.stop_event
        deadline: Optional[float] = getattr(self._request_deadline, 'value', None)
        self._check_request_current(stop_event)
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('LLM request timed out.')
        if profile.backend == 'codex':
            if self._openai_session is not None:
                self._openai_session.close()
                self._openai_session = None
            from .codex import CodexChatSession, account, request_chat_completion
            self._respect_delay()
            self._check_request_current(stop_event)
            identity = (profile.id, str(api_args['model']), account.generation)
            proxy = str(self.get_param_value('proxy') or '')
            websocket = self._codex_websocket_enabled()
            session = self._codex_sessions.get(identity)
            if (session is None or session.closed or session.proxy != proxy
                    or session.websocket != websocket):
                for old_identity in tuple(self._codex_sessions):
                    if old_identity[2] != identity[2]:
                        self._codex_sessions.pop(old_identity).close()
                cache_key = session.cache_key if session is not None else str(uuid.uuid4())
                if session is not None:
                    session.close()
                session = self._codex_sessions[identity] = CodexChatSession(
                    cache_key, proxy, account_generation=identity[2], websocket=websocket)
            self._check_request_current(stop_event)
            try:
                result = self._bounded_request(
                    lambda cancel: request_chat_completion(
                        profile, api_args, cancel, session.cache_key, proxy,
                        session=session, turn=codex_turn),
                    session.close, stop_event,
                    deadline if deadline is not None else time.monotonic() + self._request_timeout(),
                )
            except TimeoutError:
                self._codex_sessions.pop(identity, None)
                raise
            self._check_request_current(stop_event)
            return result
        if profile.backend != 'openai':
            raise LLMUserActionRequiredError('This LLM profile backend is unavailable.')
        openai = self._openai_module()
        client = self._initialize_client(profile)
        self._respect_delay()
        self._check_request_current(stop_event)
        if deadline is None:
            deadline = time.monotonic() + self._request_timeout()
        # Only the direct OpenAI endpoint has a known Responses WS contract.
        # Check the SDK's resolved URL, including OPENAI_BASE_URL overrides.
        from .openai_responses import OpenAIResponsesSession
        base_url = str(getattr(client, 'base_url', '')).rstrip('/')
        model = str(api_args.get('model', ''))
        native = base_url == 'https://api.openai.com/v1'
        version = gpt_model_version(model)
        responses_model = ((version or (0, 0)) >= (4, 1) or model.startswith('gpt-4o')
                           or re.match(r'^o[1-9](?:-|$)', model) is not None)
        responses_model = responses_model and model not in ('o1-preview', 'o1-mini')
        use_responses = native and responses_model
        identity = (profile.id, model, self.client_cache_key)
        session = self._openai_session
        cache_key = (session.cache_key if session is not None and identity == self._openai_session_identity
                     else None)
        if session is not None and (not use_responses or session.closed or identity != self._openai_session_identity):
            session.close()
            self._openai_session = session = None
        if use_responses:
            if session is None:
                headers = {key: value for key, value in client.default_headers.items()
                           if isinstance(value, str) and key.lower() not in ('accept', 'content-type')}
                # Newer SDKs add auth during HTTP request preparation, outside
                # default_headers. Our WebSocket bypasses that preparation.
                headers['Authorization'] = f'Bearer {client.api_key}'
                self._openai_session = session = OpenAIResponsesSession(
                    base_url + '/responses', cache_key or str(uuid.uuid4()),
                    str(self.get_param_value('proxy') or ''), headers)
                self._openai_session_identity = identity
            self._check_request_current(stop_event)
            try:
                result = self._bounded_request(
                    lambda cancel: session.request_chat(api_args, profile, cancel),
                    session.close, stop_event, deadline,
                )
            except TimeoutError:
                self._openai_session = None
                raise
            if result is not None:
                self._check_request_current(stop_event)
                return result
        self._check_request_current(stop_event)
        try:
            completion = self._bounded_request(
                lambda cancel: client.chat.completions.create(**api_args),
                lambda: client.close(), stop_event, deadline,
            )
        except Exception as error:
            if isinstance(error, (TimeoutError, LLMRequestStopped)):
                self.client = None
                self.client_cache_key = None
            self._check_request_current(stop_event)
            if isinstance(error, getattr(openai, 'AuthenticationError')):
                raise LLMAuthenticationError(profile.id, profile.name) from error
            if isinstance(error, getattr(openai, 'APIStatusError')):
                raise LLMChatRequestError(error) from error
            raise

        self._check_request_current(stop_event)
        choice = next(iter(getattr(completion, 'choices', ())), None)
        message = getattr(choice, 'message', None)
        content = getattr(message, 'content', None)
        if content is None:
            content = getattr(choice, 'text', '')
        result = LLMChatResult(
            content=str(content or ''),
            usage=getattr(completion, 'usage', None),
            finish_reason=str(
                getattr(choice, 'finish_reason', '') or ''
            ),
            prompt_cache_diagnostics=getattr(completion, 'prompt_cache_diagnostics', None),
        )
        if result.finish_reason.strip().lower() == 'length':
            usage = format_completion_token_usage(result)
            model = str(api_args.get('model', '')).replace(
                '\r', ' '
            ).replace('\n', ' ')
            details = ', '.join(
                part for part in (
                    f'profile_id={profile.id!r}',
                    f'model={model!r}',
                    usage,
                    'finish_reason=length',
                    f'chars={len(result.content)}',
                    f'content={result.content!r}',
                )
                if part
            )
            self.logger.debug(f'LLM output-limited response: {details}')
            raise LLMOutputLimitError(
                profile.id,
                profile.name,
                profile.max_tokens,
                str(profile.thinking_level or THINKING_AUTO),
            )
        return result
