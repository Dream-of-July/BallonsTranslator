import json
import threading
import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest import mock

import httpx

from ballontranslator.modules.exceptions import (
    LLMAuthenticationError,
    LLMApiKeyRequiredError,
    LLMOutputLimitError,
    LLMRequestStopped,
)
from ballontranslator.modules.llm_chat import (
    LLMChatRequester,
    LLMChatRequestError,
    LLMConnectionError,
    list_profile_models,
    check_profile_connection,
    openai_chat_completion_args,
    openai_json_response_format,
)
from ballontranslator.utils.llm_profiles import LLMProfile, PROVIDER_DEFAULTS, default_profile
from ballontranslator.utils.secret_store import SecretStore


class ProfileConnectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = LLMProfile(
            id='custom-test', base_url='https://provider.test/v1',
            api_key='mock-credential-only', model='test-model',
        )
        # Capture the real constructor before narrowly patching the client boundary.
        self.client_type = httpx.Client

    def test_model_listing_accepts_all_supported_shapes_and_deduplicates(self) -> None:
        for body in (
            {'data': [{'id': 'one'}, {'id': 'two'}, {'id': 'one'}, {'id': ''}, {'id': 3}, None]},
            ['one', {'id': 'two'}, 'one', ' ', None, {'id': False}],
            {'models': [{'name': 'one'}, {'id': 'two'}, {'name': 'one'}, {'name': ''}, 9]},
        ):
            with self.subTest(body=body):
                client = self.client_type(transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json=body)
                ))
                with mock.patch('httpx.Client', return_value=client):
                    self.assertEqual(list_profile_models(self.profile), ['one', 'two'])
                self.assertTrue(client.is_closed)

    def test_listing_normalizes_urls_and_passes_auth_proxy_and_timeout(self) -> None:
        for base in ('https://provider.test/v1/', 'https://provider.test/v1/chat/completions/', ''):
            with self.subTest(base=base):
                self.profile.base_url = base
                self.profile.api_key = SecretStore().store(self.profile.id, 'mock-credential-only')
                requests = []

                def respond(request: httpx.Request) -> httpx.Response:
                    requests.append(request)
                    return httpx.Response(200, json={'data': []})

                client = self.client_type(transport=httpx.MockTransport(respond))
                with mock.patch('httpx.Client', return_value=client) as factory:
                    self.assertEqual(list_profile_models(self.profile, proxy='http://proxy.test:8080', timeout=2.5), [])
                factory.assert_called_once_with(proxy='http://proxy.test:8080', timeout=2.5)
                expected_base = 'https://provider.test/v1' if base else PROVIDER_DEFAULTS['OpenAI']['base_url']
                self.assertEqual(str(requests[0].url), expected_base + '/models')
                self.assertEqual(requests[0].headers['Authorization'], 'Bearer mock-credential-only')
                self.assertTrue(client.is_closed)

    def test_local_provider_sends_no_authorization_header(self) -> None:
        self.profile.require_api_key = False
        self.profile.api_key = ''

        def respond(request: httpx.Request) -> httpx.Response:
            self.assertNotIn('Authorization', request.headers)
            return httpx.Response(200, json=[])

        client = self.client_type(transport=httpx.MockTransport(respond))
        with mock.patch('httpx.Client', return_value=client):
            self.assertEqual(list_profile_models(self.profile), [])
        self.assertTrue(client.is_closed)

    def test_missing_key_raises_for_listing_and_returns_auth_for_connection_test(self) -> None:
        self.profile.api_key = ''
        with mock.patch('httpx.Client') as factory:
            with self.assertRaises(LLMApiKeyRequiredError):
                list_profile_models(self.profile)
            result = check_profile_connection(self.profile)
        factory.assert_not_called()
        self.assertFalse(result.ok)
        self.assertEqual(result.kind, 'auth')
        self.assertIn('missing', result.message.lower())
        self.assertEqual(result.models, [])

    def test_status_errors_are_classified_and_never_include_credentials(self) -> None:
        for status, kind in ((401, 'auth'), (403, 'auth'), (404, 'not_found'),
                             (429, 'rate_limited'), (500, 'server'), (503, 'server'),
                             (400, 'invalid_response'), (302, 'invalid_response')):
            with self.subTest(status=status):
                client = self.client_type(transport=httpx.MockTransport(
                    lambda request: httpx.Response(status, text='Authorization: Bearer mock-credential-only')
                ))
                with mock.patch('httpx.Client', return_value=client):
                    with self.assertRaises(LLMConnectionError) as caught:
                        list_profile_models(self.profile)
                self.assertEqual(caught.exception.kind, kind)
                self.assertIn(str(status), str(caught.exception))
                self.assertNotIn('mock-credential-only', str(caught.exception))
                self.assertNotIn('Authorization', str(caught.exception))
                self.assertTrue(client.is_closed)

    def test_transport_errors_are_classified_and_sanitized(self) -> None:
        for error_type, kind in ((httpx.ConnectError, 'network'), (httpx.ProxyError, 'network'),
                                 (httpx.ReadTimeout, 'timeout'), (httpx.ConnectTimeout, 'timeout')):
            with self.subTest(error_type=error_type):
                def respond(request: httpx.Request) -> httpx.Response:
                    raise error_type('Authorization: Bearer mock-credential-only', request=request)

                client = self.client_type(transport=httpx.MockTransport(respond))
                with mock.patch('httpx.Client', return_value=client):
                    with self.assertRaises(LLMConnectionError) as caught:
                        list_profile_models(self.profile)
                self.assertEqual(caught.exception.kind, kind)
                self.assertNotIn('mock-credential-only', str(caught.exception))
                self.assertNotIn('Authorization', str(caught.exception))
                self.assertTrue(client.is_closed)

    def test_invalid_model_listing_is_reported_without_echoing_body(self) -> None:
        for body in ('not-json mock-credential-only', '{"error": "mock-credential-only"}',
                     '{"data": null}', '{"models": {}}', 'null'):
            with self.subTest(body=body):
                client = self.client_type(transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, text=body)
                ))
                with mock.patch('httpx.Client', return_value=client):
                    with self.assertRaises(LLMConnectionError) as caught:
                        list_profile_models(self.profile)
                self.assertEqual(caught.exception.kind, 'invalid_response')
                self.assertIn('200', str(caught.exception))
                self.assertNotIn('mock-credential-only', str(caught.exception))
                self.assertTrue(client.is_closed)

    def test_connection_succeeds_via_listing_and_returns_a_frozen_result(self) -> None:
        client = self.client_type(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={'data': [{'id': 'one'}]})
        ))
        with mock.patch('httpx.Client', return_value=client) as factory:
            result = check_profile_connection(self.profile)
        self.assertTrue(result.ok)
        self.assertEqual(result.kind, 'ok')
        self.assertEqual(result.models, ['one'])
        self.assertTrue(result.message)
        factory.assert_called_once()
        with self.assertRaises(FrozenInstanceError):
            result.ok = False

    def test_connection_falls_back_to_one_minimal_chat_only_for_unavailable_listing(self) -> None:
        for status, body in ((404, ''), (200, 'not-json')):
            with self.subTest(status=status):
                requests = []
                clients = []

                def respond(request: httpx.Request) -> httpx.Response:
                    requests.append(request)
                    if request.method == 'GET':
                        return httpx.Response(status, text=body)
                    return httpx.Response(204)

                def create_client(**kwargs: object) -> httpx.Client:
                    client = self.client_type(transport=httpx.MockTransport(respond))
                    clients.append(client)
                    return client

                self.profile.base_url = 'https://provider.test/v1/chat/completions/'
                with mock.patch('httpx.Client', side_effect=create_client) as factory:
                    result = check_profile_connection(self.profile, proxy='http://proxy.test:8080', timeout=3.0)
                self.assertTrue(result.ok)
                self.assertEqual(result.kind, 'ok')
                self.assertEqual(result.models, [])
                self.assertEqual([request.method for request in requests], ['GET', 'POST'])
                self.assertEqual(str(requests[1].url), 'https://provider.test/v1/chat/completions')
                payload = json.loads(requests[1].content)
                self.assertEqual(payload['model'], self.profile.model)
                self.assertEqual(payload['max_tokens'], 1)
                self.assertEqual(len(payload['messages']), 1)
                self.assertEqual(payload['messages'][0]['role'], 'user')
                self.assertTrue(payload['messages'][0]['content'])
                self.assertEqual(requests[1].headers['Authorization'], 'Bearer mock-credential-only')
                self.assertEqual(factory.call_args_list, [mock.call(proxy='http://proxy.test:8080', timeout=3.0)] * 2)
                self.assertTrue(all(client.is_closed for client in clients))

    def test_fallback_is_skipped_without_a_model_or_after_provider_failure(self) -> None:
        for status, body, model, expected_kind in (
            (404, '', '', 'not_found'), (200, 'garbage', '  ', 'invalid_response'),
            (429, '', 'test-model', 'rate_limited'), (500, '', 'test-model', 'server'),
        ):
            with self.subTest(status=status, model=model):
                self.profile.model = model
                requests = []

                def respond(request: httpx.Request) -> httpx.Response:
                    requests.append(request)
                    return httpx.Response(status, text=body)

                client = self.client_type(transport=httpx.MockTransport(respond))
                with mock.patch('httpx.Client', return_value=client):
                    result = check_profile_connection(self.profile)
                self.assertFalse(result.ok)
                self.assertEqual(result.kind, expected_kind)
                self.assertEqual(result.models, [])
                self.assertEqual([request.method for request in requests], ['GET'])
                if not model.strip():
                    self.assertIn('manually', result.message)
                    self.assertIn('model', result.message)
                self.assertTrue(client.is_closed)

    def test_connection_reports_fallback_and_network_failures(self) -> None:
        for fallback in (False, True):
            for failure in ('auth', 'network', 'timeout'):
                with self.subTest(fallback=fallback, failure=failure):
                    def respond(request: httpx.Request) -> httpx.Response:
                        if fallback and request.method == 'GET':
                            return httpx.Response(404)
                        if failure == 'auth':
                            return httpx.Response(401, text='mock-credential-only')
                        error_type = httpx.ConnectError if failure == 'network' else httpx.ReadTimeout
                        raise error_type('mock-credential-only', request=request)

                    clients = []

                    def create_client(**kwargs: object) -> httpx.Client:
                        client = self.client_type(transport=httpx.MockTransport(respond))
                        clients.append(client)
                        return client

                    with mock.patch('httpx.Client', side_effect=create_client):
                        result = check_profile_connection(self.profile)
                    self.assertFalse(result.ok)
                    self.assertEqual(result.kind, failure)
                    self.assertNotIn('mock-credential-only', result.message)
                    self.assertEqual(result.models, [])
                    self.assertTrue(all(client.is_closed for client in clients))

    def test_invalid_proxy_setup_is_a_safe_network_error(self) -> None:
        with mock.patch('httpx.Client', side_effect=ValueError('mock-credential-only')):
            result = check_profile_connection(self.profile, proxy='invalid-proxy')
        self.assertFalse(result.ok)
        self.assertEqual(result.kind, 'network')
        self.assertNotIn('mock-credential-only', result.message)


class FakeAuthError(Exception):
    pass


class FakeStatusError(Exception):
    def __init__(self, message: str = 'provider says no') -> None:
        self.response = SimpleNamespace(
            json=lambda: {'error': {'message': message}},
            text='raw',
        )
        super().__init__('status')


class FakeOpenAI:
    AuthenticationError = FakeAuthError
    APIStatusError = FakeStatusError


class RequesterHarness(LLMChatRequester):
    def __init__(self) -> None:
        super().__init__()
        self.values = {
            'delay': 0.0,
            'max requests per minute': 0,
            'proxy': '',
        }
        self.logs = []
        self.logger = SimpleNamespace(
            debug=self.logs.append,
            error=self.logs.append,
            warning=self.logs.append,
        )

    def get_param_value(self, name: str):
        return self.values[name]


class LLMChatRequesterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.requester = RequesterHarness()
        self.profile = default_profile('OpenAI')
        self.profile.api_key = 'sk-demo'

    def test_provider_args_preserve_native_openai_compatibility(self):
        self.assertEqual(
            openai_chat_completion_args(self.profile, 'gpt-5.5'),
            {'max_completion_tokens': 8192},
        )
        self.assertEqual(
            openai_chat_completion_args(self.profile, 'gpt-4o'),
            {
                'top_p': 1.0,
                'temperature': 0.1,
                'max_tokens': 8192,
            },
        )

        self.profile.base_url = 'https://openrouter.ai/api/v1'
        self.assertEqual(
            openai_chat_completion_args(self.profile, 'openai/gpt-5.5'),
            {'max_tokens': 8192},
        )

        self.profile.base_url = (
            'https://api.openai.com/v1/chat/completions'
        )
        self.assertEqual(
            openai_chat_completion_args(self.profile, 'gpt-5.5'),
            {'max_completion_tokens': 8192},
        )

    def test_gpt_sampling_omission_is_endpoint_independent(self) -> None:
        for provider in ('OpenAI', 'OpenRouter', 'LM Studio'):
            profile = default_profile(provider)
            for model in ('gpt-5.5', 'gpt-5.6-luna', 'gpt-6-astra', 'openai/gpt-5.6-luna'):
                with self.subTest(provider=provider, model=model):
                    args = openai_chat_completion_args(profile, model)
                    self.assertNotIn('temperature', args)
                    self.assertNotIn('top_p', args)
                    token_key = 'max_completion_tokens' if provider == 'OpenAI' else 'max_tokens'
                    self.assertEqual(args, {token_key: profile.max_tokens})
            for model in ('gpt-5.4', 'gpt-4o', 'deepseek-v4-flash'):
                with self.subTest(provider=provider, model=model):
                    args = openai_chat_completion_args(profile, model)
                    self.assertEqual(args['temperature'], profile.temperature)
                    self.assertEqual(args['top_p'], profile.top_p)

    def test_reasoning_control_maps_provider_specific_disable_requests(self):
        self.profile.thinking_level = 'None'
        automatic = openai_chat_completion_args(
            self.profile,
            'gpt-5.5',
        )
        self.assertNotIn('reasoning_effort', automatic)
        self.assertNotIn('extra_body', automatic)

        self.profile.thinking_level = 'Disabled'
        native_openai = openai_chat_completion_args(
            self.profile,
            'gpt-5.5',
        )
        self.assertEqual(native_openai['reasoning_effort'], 'none')

        self.profile.base_url = (
            'https://api.deepseek.com/v1/chat/completions'
        )
        deepseek = openai_chat_completion_args(
            self.profile,
            'deepseek-v4-flash-vision-exp',
        )
        self.assertEqual(
            deepseek['extra_body'],
            {'thinking': {'type': 'disabled'}},
        )
        self.assertNotIn('reasoning_effort', deepseek)

        self.profile.base_url = 'https://openrouter.ai/api/v1'
        openrouter = openai_chat_completion_args(
            self.profile,
            'deepseek/deepseek-v4-flash-vision-exp',
        )
        self.assertEqual(
            openrouter['extra_body'],
            {'reasoning': {'effort': 'none'}},
        )

        self.profile.thinking_level = 'low'
        explicit = openai_chat_completion_args(
            self.profile,
            'deepseek/deepseek-v4-flash-vision-exp',
        )
        self.assertEqual(explicit['reasoning_effort'], 'low')
        self.assertNotIn('extra_body', explicit)

    def test_json_response_format_preserves_strict_and_compatible_shapes(self):
        schema = {'type': 'object', 'properties': {}}
        self.profile.json_schema_response_format = True

        strict = openai_json_response_format(
            self.profile,
            'demo_response',
            schema,
        )
        self.profile.json_schema_response_format = False
        compatible = openai_json_response_format(
            self.profile,
            'demo_response',
            schema,
        )

        self.assertEqual(strict, {
            'type': 'json_schema',
            'json_schema': {
                'name': 'demo_response',
                'strict': True,
                'schema': schema,
            },
        })
        self.assertIs(strict['json_schema']['schema'], schema)
        self.assertEqual(compatible, {'type': 'json_object'})

    def test_request_returns_normalized_content_and_usage(self):
        usage = SimpleNamespace(total_tokens=3)
        completion = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content='hello'),
                finish_reason='stop',
            )],
            usage=usage,
        )
        completions = SimpleNamespace(create=mock.Mock(return_value=completion))
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        api_args = {'model': 'gpt-5.5', 'messages': []}

        with (
            mock.patch.object(
                self.requester, '_openai_module', return_value=FakeOpenAI
            ),
            mock.patch.object(
                self.requester, '_initialize_client', return_value=client
            ),
            mock.patch.object(self.requester, '_respect_delay') as delay,
        ):
            result = self.requester.request_chat_completion(
                self.profile, api_args
            )

        delay.assert_called_once_with()
        completions.create.assert_called_once_with(**api_args)
        self.assertEqual(result.content, 'hello')
        self.assertIs(result.usage, usage)
        self.assertEqual(result.finish_reason, 'stop')

    def test_request_raises_actionable_output_limit(self):
        self.profile.max_tokens = 1234
        self.profile.thinking_level = 'low'
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=1234,
            total_tokens=1244,
        )
        completion = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"partial":'),
                finish_reason='length',
            )],
            usage=usage,
        )
        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=mock.Mock(return_value=completion)
            )
        ))

        with (
            mock.patch.object(
                self.requester, '_openai_module', return_value=FakeOpenAI
            ),
            mock.patch.object(
                self.requester, '_initialize_client', return_value=client
            ),
            mock.patch.object(self.requester, '_respect_delay'),
        ):
            with self.assertRaisesRegex(
                LLMOutputLimitError,
                r'Max Tokens: 1234.*Thinking Level: low',
            ):
                self.requester.request_chat_completion(
                    self.profile,
                    {'model': 'demo-model', 'messages': []},
                )

    def test_request_normalizes_authentication_and_status_errors(self):
        def client_for(error: Exception):
            completions = SimpleNamespace(
                create=mock.Mock(side_effect=error)
            )
            return SimpleNamespace(
                chat=SimpleNamespace(completions=completions)
            )

        with (
            mock.patch.object(
                self.requester, '_openai_module', return_value=FakeOpenAI
            ),
            mock.patch.object(self.requester, '_respect_delay'),
            mock.patch.object(
                self.requester,
                '_initialize_client',
                return_value=client_for(FakeAuthError('bad key')),
            ),
        ):
            with self.assertRaisesRegex(LLMAuthenticationError, 'Authentication failed'):
                self.requester.request_chat_completion(self.profile, {})

        provider_error = FakeStatusError()
        with (
            mock.patch.object(
                self.requester, '_openai_module', return_value=FakeOpenAI
            ),
            mock.patch.object(self.requester, '_respect_delay'),
            mock.patch.object(
                self.requester,
                '_initialize_client',
                return_value=client_for(provider_error),
            ),
        ):
            with self.assertRaisesRegex(
                LLMChatRequestError, 'provider says no'
            ) as caught:
                self.requester.request_chat_completion(self.profile, {})
        self.assertIs(caught.exception.provider_error, provider_error)

    def test_client_is_reused_for_the_same_profile_and_proxy(self):
        client = object()
        openai = SimpleNamespace(OpenAI=mock.Mock(return_value=client))
        http_client = object()
        with (
            mock.patch.object(
                self.requester, '_openai_module', return_value=openai
            ),
            mock.patch.object(
                self.requester, '_http_client', return_value=http_client
            ),
        ):
            first = self.requester._initialize_client(self.profile)
            second = self.requester._initialize_client(self.profile)

        self.assertIs(first, client)
        self.assertIs(second, client)
        openai.OpenAI.assert_called_once_with(
            api_key='sk-demo',
            base_url='https://api.openai.com/v1',
            http_client=http_client,
        )

    def test_full_chat_endpoint_is_requested_once(self):
        requested_urls = []

        def respond(request: httpx.Request) -> httpx.Response:
            requested_urls.append(str(request.url))
            return httpx.Response(200, json={
                'id': 'chatcmpl-test',
                'object': 'chat.completion',
                'created': 0,
                'model': 'test-model',
                'choices': [{
                    'index': 0,
                    'message': {'role': 'assistant', 'content': 'ok'},
                    'finish_reason': 'stop',
                }],
            })

        self.profile.base_url = (
            'https://openrouter.ai/api/v1/chat/completions/'
        )
        with (
            httpx.Client(transport=httpx.MockTransport(respond)) as http_client,
            mock.patch.object(
                self.requester, '_http_client', return_value=http_client
            ),
            mock.patch.object(self.requester, '_respect_delay'),
        ):
            result = self.requester.request_chat_completion(
                self.profile,
                {'model': 'test-model', 'messages': []},
            )

        self.assertEqual(result.content, 'ok')
        self.assertEqual(
            requested_urls,
            ['https://openrouter.ai/api/v1/chat/completions'],
        )

    def test_wait_remains_stop_aware(self):
        stop_event = threading.Event()
        stop_event.set()
        self.requester.set_stop_event(stop_event)

        with self.assertRaises(LLMRequestStopped):
            self.requester._wait(5)


if __name__ == '__main__':
    unittest.main()
