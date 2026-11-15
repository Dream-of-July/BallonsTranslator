import copy
import json
import threading
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import httpx
import numpy as np

from ballontranslator.modules.exceptions import (
    LLMModelRequiredError,
    LLMApiKeyRequiredError,
    LLMAuthenticationError,
    LLMBaseURLRequiredError,
    LLMOutputLimitError,
    LLMRequestStopped,
    LLMUserActionRequiredError,
)
from ballontranslator.modules.context.errors import ContextLengthError
from ballontranslator.modules.context.translation_context import PageSummary
from ballontranslator.modules.context.token_usage import format_token_usage
from ballontranslator.modules.llm_chat import LLMChatRequestError, LLMChatResult
from ballontranslator.modules.translators.llm_translation_contract import (
    TranslationPromptSpec,
    InvalidNumTranslations,
    translation_system_prompt,
)
from ballontranslator.modules.translators.trans_llm import LLMTranslator
from ballontranslator.utils import shared
from ballontranslator.utils.config import LLMTranslateContext, pcfg
from ballontranslator.utils.llm_profiles import default_codex_profile, default_profile
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.textblock import TextBlock


class FakeStatusError(Exception):
    def __init__(self, message='provider says no', status_code=400, code=''):
        self.status_code = status_code
        self.code = code
        self.response = SimpleNamespace(
            json=lambda: {'error': {'message': message, 'code': code}},
            text='raw',
            status_code=status_code,
        )
        super().__init__('status')


class LLMTranslatorTest(unittest.TestCase):
    def setUp(self):
        self.translator = LLMTranslator('日本語', '简体中文')

    def test_native_translation_thinking_override_preserves_profile_and_arguments(self) -> None:
        messages = [{'role': 'user', 'content': 'source'}]
        for provider in ('DeepSeek', 'OpenAI', 'OpenRouter', 'LM Studio'):
            for level in ('Auto', 'None', ' auto ', 'Disabled', 'low', 'high'):
                for native in (False, True):
                    with self.subTest(provider=provider, level=level, native=native):
                        profile = default_profile(provider)
                        profile.thinking_level = level
                        before = copy.deepcopy(profile)
                        expected_profile = (
                            replace(profile, thinking_level='Disabled')
                            if native and level in ('Auto', 'None', ' auto ')
                            else profile
                        )
                        expected = json.dumps(self.translator._api_args(
                            expected_profile, messages, summary_enabled=True))
                        with mock.patch.object(shared, 'ENGINE_TRANSLATION_AUTO_THINKING_DISABLED', native), \
                                mock.patch.object(self.translator, 'request_chat_completion',
                                    return_value=LLMChatResult(content='target')) as request:
                            self.assertEqual(self.translator._request_translation(
                                profile, messages, summary_enabled=True), 'target')
                        request.assert_called_once()
                        self.assertEqual(json.dumps(request.call_args.args[1]), expected)
                        self.assertEqual(request.call_args.args[0], expected_profile)
                        self.assertEqual(profile, before)

    def test_native_translation_keeps_profiles_without_disabled_support(self) -> None:
        for profile in (default_profile('DeepSeek'), default_codex_profile()):
            with self.subTest(backend=profile.backend):
                profile.model, profile.model_options = 'offline-model', ['offline-model']
                profile.thinking_level = 'Auto'
                profile.thinking_level_options = ['Auto', 'high']
                before = copy.deepcopy(profile)
                messages = [{'role': 'user', 'content': 'source'}]
                expected = json.dumps(self.translator._api_args(profile, messages))
                with mock.patch.object(shared, 'ENGINE_TRANSLATION_AUTO_THINKING_DISABLED', True), \
                        mock.patch.object(self.translator, 'request_chat_completion',
                            return_value=LLMChatResult(content='target')) as request:
                    self.translator._request_translation(profile, messages)
                self.assertIs(request.call_args.args[0], profile)
                self.assertEqual(json.dumps(request.call_args.args[1]), expected)
                self.assertEqual(profile, before)

    def test_native_translation_exhaustion_retries_once_without_reapplying_override(self) -> None:
        profile = default_profile('DeepSeek')
        before = copy.deepcopy(profile)
        original = LLMOutputLimitError(profile.id, profile.name, profile.max_tokens, 'Disabled')
        second = LLMOutputLimitError(profile.id, profile.name, profile.max_tokens, 'Disabled')
        for outcome in (LLMChatResult(content='target'), second):
            with self.subTest(succeeds=isinstance(outcome, LLMChatResult)), \
                    mock.patch.object(shared, 'ENGINE_TRANSLATION_AUTO_THINKING_DISABLED', True), \
                    mock.patch.object(self.translator, 'request_chat_completion',
                        side_effect=[original, outcome]) as request:
                if isinstance(outcome, LLMChatResult):
                    self.assertEqual(self.translator._request_translation(profile, []), 'target')
                else:
                    with self.assertRaises(LLMOutputLimitError) as caught:
                        self.translator._request_translation(profile, [])
                    self.assertIs(caught.exception, original)
                self.assertEqual(request.call_count, 2)
                first, retry = request.call_args_list
                self.assertIs(first.args[0], retry.args[0])
                self.assertEqual(first, retry)
                self.assertEqual(first.args[1]['extra_body'], {'thinking': {'type': 'disabled'}})
                self.assertEqual(profile, before)

    def test_native_memory_compaction_keeps_auto_thinking(self) -> None:
        profile = default_profile('DeepSeek')
        before = copy.deepcopy(profile)
        with mock.patch.object(shared, 'ENGINE_TRANSLATION_AUTO_THINKING_DISABLED', True), \
                mock.patch.object(self.translator, 'request_chat_completion',
                    return_value=LLMChatResult(content='memory')) as request:
            result = self.translator._compact_summary_batch(
                previous=None, summaries=(PageSummary('001.png', 'summary'),),
                profile=profile, model='offline-model', target_language='Chinese')
        request.assert_called_once()
        self.assertIs(request.call_args.args[0], profile)
        self.assertNotIn('extra_body', request.call_args.args[1])
        self.assertNotIn('reasoning_effort', request.call_args.args[1])
        self.assertEqual(result.text, 'memory')
        self.assertEqual(profile, before)

    def test_exhausted_translation_retries_once_without_persisting_thinking(self) -> None:
        profile = default_profile('DeepSeek')
        profile.api_key = 'offline-test-key'
        profile.model, profile.model_options = 'deepseek-chat', ['deepseek-chat']
        profile.thinking_level = 'Auto'
        before = copy.deepcopy(profile)
        requests = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content))
            exhausted = len(requests) == 1
            return httpx.Response(200, json={'choices': [{
                'index': 0, 'finish_reason': 'length' if exhausted else 'stop',
                'message': {'role': 'assistant', 'content': '' if exhausted else '{"1":"target"}'},
            }]})

        with httpx.Client(transport=httpx.MockTransport(respond)) as client, \
                mock.patch.object(self.translator, '_http_client', return_value=client), \
                mock.patch.object(self.translator, '_respect_delay'), \
                mock.patch.object(self.translator, '_wait') as wait:
            for _ in range(2):
                self.assertEqual(self.translator._translate(['source'], profile=profile,
                    prompt_spec=self._prompt_spec(profile)), ['target'])
        self.assertEqual(len(requests), 3)
        self.assertNotIn('thinking', requests[0])
        self.assertEqual(requests[1]['thinking'], {'type': 'disabled'})
        self.assertNotIn('thinking', requests[2])
        self.assertEqual(profile, before)
        wait.assert_not_called()

    def test_second_exhaustion_raises_original_and_other_setup_errors_do_not_retry(self) -> None:
        profile = default_profile('DeepSeek')
        original = LLMOutputLimitError(profile.id, profile.name, profile.max_tokens, 'Auto')
        second = LLMOutputLimitError(profile.id, profile.name, profile.max_tokens, 'Disabled')
        with mock.patch.object(self.translator, 'request_chat_completion',
                side_effect=[original, second]) as request:
            with self.assertRaises(LLMOutputLimitError) as caught:
                self.translator._translate(['source'], profile=profile,
                    prompt_spec=self._prompt_spec(profile))
        self.assertIs(caught.exception, original)
        self.assertEqual(request.call_count, 2)
        first, retry = request.call_args_list
        self.assertEqual(first.kwargs, retry.kwargs)
        self.assertEqual(retry.args[1]['extra_body']['thinking'], {'type': 'disabled'})
        for error_type in (LLMApiKeyRequiredError, LLMModelRequiredError,
                           LLMBaseURLRequiredError, LLMAuthenticationError):
            error = error_type(profile.id, profile.name)
            with self.subTest(error=error_type), \
                    mock.patch.object(self.translator, 'request_chat_completion', side_effect=error) as request:
                with self.assertRaises(error_type):
                    self.translator._translate(['source'], profile=profile,
                        prompt_spec=self._prompt_spec(profile))
                self.assertEqual(request.call_count, 1)

    def test_stop_during_exhausted_retry_is_honoured(self) -> None:
        profile = default_profile('DeepSeek')
        profile.api_key = 'offline-test-key'
        stop = threading.Event()
        self.translator.set_stop_event(stop)
        requests = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if len(requests) == 2:
                stop.set()
            return httpx.Response(200, json={'choices': [{
                'index': 0, 'finish_reason': 'length',
                'message': {'role': 'assistant', 'content': ''},
            }]})

        with httpx.Client(transport=httpx.MockTransport(respond)) as client, \
                mock.patch.object(self.translator, '_http_client', return_value=client), \
                mock.patch.object(self.translator, '_respect_delay'):
            with self.assertRaises(LLMRequestStopped):
                self.translator._translate(['source'], profile=profile,
                    prompt_spec=self._prompt_spec(profile))
        self.assertEqual(len(requests), 2)

    def test_exhausted_retry_keeps_deadline_and_unsupported_backend_arguments(self) -> None:
        profile = default_codex_profile()
        profile.model = 'offline-model'
        profile.thinking_level_options = ['Auto', 'high']
        error = LLMOutputLimitError(profile.id, profile.name, profile.max_tokens, 'Auto')
        received = []

        def request(request_profile, args, *, codex_turn=None):
            self.assertIs(request_profile, profile)
            received.append((args, self.translator._request_deadline.value))
            if len(received) == 1:
                raise error
            return LLMChatResult(content='{"translations":[{"id":1,"translation":"target"}]}')

        with mock.patch.object(self.translator, 'request_chat_completion', side_effect=request):
            self.assertEqual(self.translator._translate(['source'], profile=profile,
                prompt_spec=self._prompt_spec(profile)), ['target'])
        self.assertEqual(received[0], received[1])

        profile = default_profile('DeepSeek')
        received.clear()
        def expire(request_profile, args, *, codex_turn=None):
            received.append(self.translator._request_deadline.value)
            if len(received) == 1:
                time.sleep(.02)
                raise error
            self.assertLess(self.translator._request_deadline.value, time.monotonic())
            raise TimeoutError('Request deadline expired')

        with mock.patch.object(self.translator, '_request_timeout', return_value=.01), \
                mock.patch.object(self.translator, 'request_chat_completion', side_effect=expire):
            with self.assertRaises(LLMOutputLimitError) as caught:
                self.translator._translate(['source'], profile=profile,
                    prompt_spec=self._prompt_spec(profile))
        self.assertIs(caught.exception, error)
        self.assertEqual(received[0], received[1])

    def test_older_params_get_default_request_timeout(self) -> None:
        from ballontranslator.modules.lazy_registry import _scan_file

        specs = _scan_file('ballontranslator/modules/translators/trans_llm.py', 'translator')
        spec = next(spec for spec in specs if spec.key == 'LLMTranslator')
        self.assertEqual(spec.params['request timeout']['value'], 120.0)
        self.assertEqual(spec.params['retry timeout']['description'], 'Delay between retries in seconds.')
        params = copy.deepcopy(self.translator.params)
        params.pop('request timeout')
        params['retry timeout']['value'] = 0.01
        with mock.patch.object(LLMTranslator, 'params', copy.deepcopy(LLMTranslator.params)):
            translator = LLMTranslator('日本語', '简体中文', **params)
            self.assertEqual(translator.get_param_value('retry timeout'), 0.01)
            self.assertEqual(translator.get_param_value('request timeout'), 120.0)
            self.assertEqual(translator._request_timeout(), 120.0)

    def test_transport_timeout_connection_and_server_retries_are_finite(self) -> None:
        from openai import APIConnectionError, APITimeoutError
        profile = default_profile('LM Studio')
        profile.model, profile.model_options = 'offline-model', ['offline-model']
        for failure in (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError, 503):
            with self.subTest(failure=failure), \
                    mock.patch.object(self.translator, 'params', copy.deepcopy(self.translator.params)):
                self.translator.set_param_value('retry attempts', 3)
                self.translator.set_param_value('retry timeout', 0.01)
                requests, clients = [], []

                def respond(request: httpx.Request) -> httpx.Response:
                    requests.append(request)
                    if failure == 503:
                        return httpx.Response(503, json={'error': {'message': 'Unavailable'}})
                    raise failure('Offline transport failure', request=request)

                def client(proxy: str) -> httpx.Client:
                    transport = httpx.Client(transport=httpx.MockTransport(respond))
                    clients.append(transport)
                    return transport

                with mock.patch.object(self.translator, '_http_client', side_effect=client), \
                        mock.patch.object(self.translator, '_respect_delay'), \
                        mock.patch.object(self.translator, '_wait') as wait:
                    expected = (LLMChatRequestError if failure == 503 else
                                APITimeoutError if issubclass(failure, httpx.TimeoutException) else
                                APIConnectionError)
                    with self.assertRaises(expected):
                        self.translator._translate(['source'], profile=profile,
                                                   prompt_spec=self._prompt_spec(profile))
                self.assertEqual(len(requests), 3)
                self.assertEqual(wait.call_count, 2)
                self.assertEqual(wait.call_args_list, [mock.call(0.01)] * 2)
                for transport in clients:
                    transport.close()
                self.translator.client = None
                self.translator.client_cache_key = None

    def test_nonfinite_retry_count_falls_back_to_finite_attempts(self) -> None:
        profile = default_profile('LM Studio')
        with mock.patch.object(self.translator, 'params', copy.deepcopy(self.translator.params)):
            self.translator.set_param_value('retry attempts', float('inf'), convert_dtype=False)
            with mock.patch.object(self.translator, '_request_translation', side_effect=TimeoutError('Timed out')) as request, \
                    mock.patch.object(self.translator, '_wait'):
                with self.assertRaises(TimeoutError):
                    self.translator._translate(['source'], profile=profile,
                                               prompt_spec=self._prompt_spec(profile))
            self.assertEqual(request.call_count, 5)

    def test_gateway_cache_diagnostics_reach_logs_without_changing_requests(self) -> None:
        profile = default_profile('OpenAI')
        profile.base_url = 'https://gateway.example/v1'
        profile.api_key = 'test-key'
        profile.model, profile.model_options = 'gpt-6-astra', ['gpt-6-astra']
        responses = [
            {'type': 'cache_miss', 'reason': 'input_changed',
             'comparison_reusable_tokens': 2048, 'cache_missed_tokens': 1024,
             'input': 'private-source', 'access_token': 'private-token'},
            None,
            {'type': 'private\nAuthorization: secret', 'reason': {'input': 'private-source'},
             'comparison_reusable_tokens': True, 'cache_missed_tokens': -1},
        ]
        received = []

        def respond(request: httpx.Request) -> httpx.Response:
            diagnostics = responses[len(received)]
            received.append(json.loads(request.content))
            body = {'choices': [{'index': 0, 'finish_reason': 'stop',
                                'message': {'role': 'assistant', 'content': 'translated'}}],
                    'usage': {'prompt_tokens': 2500, 'completion_tokens': 10, 'total_tokens': 2510}}
            if diagnostics is not None:
                body['prompt_cache_diagnostics'] = diagnostics
            return httpx.Response(200, json=body)

        with httpx.Client(transport=httpx.MockTransport(respond)) as client, \
                mock.patch.object(self.translator, '_http_client', return_value=client), \
                mock.patch.object(self.translator, '_respect_delay'), \
                mock.patch.object(self.translator.logger, 'debug') as debug:
            for attempt in range(1, 4):
                self.assertEqual(self.translator._request_translation(
                    profile, [{'role': 'system', 'content': 'rules'},
                              {'role': 'user', 'content': 'source'}],
                    usage_page_key='001.png', usage_attempt=attempt,
                ), 'translated')
        self.assertEqual(len(received), 3)
        self.assertTrue(all(body == received[0] for body in received))
        self.assertNotIn('comparison_response_id', json.dumps(received))
        lines = [call.args[0] for call in debug.call_args_list if call.args[0].startswith('LLM token usage:')]
        self.assertEqual(len(lines), 3)
        self.assertIn('page=001.png, attempt=1', lines[0])
        self.assertIn('"type":"cache_miss","reason":"input_changed"', lines[0])
        self.assertIn('"comparison_reusable_tokens":2048,"cache_missed_tokens":1024', lines[0])
        self.assertIn('prompt_cache_diagnostics=not_reported', lines[1])
        self.assertIn('prompt_cache_diagnostics=unrecognized', lines[2])
        self.assertNotIn('private', '\n'.join(lines))
        self.assertNotIn('secret', '\n'.join(lines))

    def test_gpt_page_contract_and_explicit_cache_reach_api_and_gateway(self) -> None:
        cases = (
            ('https://api.openai.com/v1', 'gpt-5.5', False, True),
            ('https://api.openai.com/v1', 'gpt-6-astra', True, True),
            ('https://gateway.example/v1', 'openai/gpt-5.6-luna', True, True),
            ('https://gateway.example/v1', 'gpt-5.10-snapshot', True, False),
        )
        for base_url, model, explicit, strict in cases:
            with self.subTest(base_url=base_url, model=model):
                profile = default_profile('OpenAI')
                profile.base_url, profile.model = base_url, model
                profile.model_options, profile.api_key = [model], 'test-key'
                profile.json_schema_response_format = strict
                project = ProjImgTrans()
                project.pages = {str(page): [TextBlock(text=[f'source-{page}-{i}'])
                                            for i in range(count)]
                                 for page, count in enumerate((1, 3, 2, 1))}
                project._pagename2idx = {key: index for index, key in enumerate(project.pages)}
                project._image_info = {page: {'finish_code': 0} for page in project.pages}
                project.read_img = mock.Mock(return_value=np.zeros((16, 16, 3), dtype=np.uint8))
                received = []

                def respond(request: httpx.Request) -> httpx.Response:
                    body = json.loads(request.content)
                    received.append(body)
                    page = str(len(received) - 1)
                    content = {'translations': [{'id': i + 1, 'translation': f'target-{page}-{i}'}
                                                for i in range(len(project.pages[page]))]}
                    if page == '3':
                        content['translations'].append(content['translations'][0])
                    return httpx.Response(200, json={'choices': [{
                        'index': 0, 'finish_reason': 'stop',
                        'message': {'role': 'assistant', 'content': json.dumps(content)},
                    }]})

                settings = {'llm_profiles': [profile], 'translator_llm_id': profile.id,
                            'llm_translate_context': LLMTranslateContext.HISTORY,
                            'llm_prior_context_token_budget': 4096, 'llm_translate_vision': True,
                            'llm_translate_summary_memory': False, 'llm_glossary_path': ''}
                with mock.patch.object(LLMTranslator, 'params', copy.deepcopy(LLMTranslator.params)), \
                        mock.patch.dict(pcfg.module.__dict__, settings), \
                        mock.patch('websockets.asyncio.client.connect', side_effect=OSError('Offline HTTP contract test')), \
                        httpx.Client(transport=httpx.MockTransport(respond)) as client:
                    translator = LLMTranslator('日本語', 'English')
                    translator.set_param_value('retry attempts', 1)
                    with mock.patch.object(translator, '_http_client', return_value=client), \
                            mock.patch.object(translator, '_respect_delay'):
                        for page in ('0', '1', '2'):
                            translator.translate_textblk_lst(project.pages[page], project=project,
                                                             page_key=page, full_page=True)
                            project.mark_translation_finished(page, 'English')
                            self.assertEqual(project.pages[page][0].translation, f'target-{page}-0')
                        window = translator._history_window
                        with self.assertRaises(InvalidNumTranslations):
                            translator.translate_textblk_lst(project.pages['3'], project=project,
                                                             page_key='3', full_page=True)
                        self.assertIs(translator._history_window, window)
                        self.assertEqual(project.pages['3'][0].translation, '')
                self.assertEqual(len(received), 4)
                for body in received:
                    self.assertEqual(body['response_format'], received[0]['response_format'])
                    if strict:
                        self.assertEqual(body['response_format']['json_schema']['schema']['required'], ['translations'])
                    else:
                        self.assertEqual(body['response_format'], {'type': 'json_object'})
                    self.assertEqual(body['model'], model)
                    self.assertNotIn('extra_body', body)
                    self.assertEqual(body['messages'][-1]['content'][-1]['type'], 'image_url')
                    if explicit:
                        self.assertEqual(body['prompt_cache_options'], {'mode': 'explicit', 'ttl': '30m'})
                        self.assertIn('prompt_cache_breakpoint', body['messages'][0]['content'][0])
                        self.assertNotIn('prompt_cache_breakpoint', body['messages'][-1]['content'][0])
                    else:
                        self.assertNotIn('prompt_cache_options', body)
                        self.assertIsInstance(body['messages'][0]['content'], str)
                history_content = received[2]['messages'][-2]['content']
                self.assertEqual(received[2]['messages'][-2]['role'], 'assistant')
                history = json.loads(history_content)
                self.assertEqual(history['translations'], [
                    {'id': i + 1, 'translation': f'target-1-{i}'} for i in range(3)
                ])
                # Ignore only cache metadata, never normalize message representation.
                for previous, current in zip(received[1:], received[2:]):
                    prefix = copy.deepcopy(previous['messages'][:-1])
                    following = copy.deepcopy(current['messages'][:len(prefix)])
                    for message in prefix + following:
                        if isinstance(message['content'], list):
                            for part in message['content']:
                                part.pop('prompt_cache_breakpoint', None)
                    self.assertEqual(prefix, following)

    def _prompt_spec(
        self,
        profile,
        *,
        summary_enabled: bool = False,
    ) -> TranslationPromptSpec:
        target_language = self.translator._translated_lang(
            self.translator.lang_target
        )
        return TranslationPromptSpec(
            source_language=self.translator._translated_lang(
                self.translator.lang_source
            ),
            target_language=target_language,
            system_prompt=translation_system_prompt(
                profile.prompt,
                target_language,
                history_enabled=False,
                summary_enabled=summary_enabled,
            ),
            summary_enabled=summary_enabled,
        )

    def test_text_disabled_profile_is_not_translator_usable(self):
        old_profiles = pcfg.module.llm_profiles
        old_translator_llm_id = pcfg.module.translator_llm_id
        profile = default_profile('OpenAI')
        profile.support_text = False
        try:
            pcfg.module.llm_profiles = [profile]
            pcfg.module.translator_llm_id = profile.id

            with self.assertRaisesRegex(RuntimeError, 'text translation'):
                _ = self.translator.profile
        finally:
            pcfg.module.llm_profiles = old_profiles
            pcfg.module.translator_llm_id = old_translator_llm_id

    def test_text_enabled_profile_requires_model(self) -> None:
        old_profiles = pcfg.module.llm_profiles
        old_translator_llm_id = pcfg.module.translator_llm_id
        try:
            for profile in (default_profile('OpenAI'), default_codex_profile()):
                with self.subTest(backend=profile.backend):
                    profile.model = ''
                    pcfg.module.llm_profiles = [profile]
                    pcfg.module.translator_llm_id = profile.id

                    with self.assertRaises(LLMModelRequiredError):
                        _ = self.translator.profile
                    with self.assertRaises(LLMModelRequiredError):
                        self.translator._api_args(profile, [{'role': 'user', 'content': 'x'}])

                    profile.model = 'stale-model'
                    profile.model_options = []
                    with self.assertRaises(LLMModelRequiredError):
                        _ = self.translator.profile
                    with self.assertRaises(LLMModelRequiredError):
                        self.translator._api_args(profile, [{'role': 'user', 'content': 'x'}])
        finally:
            pcfg.module.llm_profiles = old_profiles
            pcfg.module.translator_llm_id = old_translator_llm_id

    def test_run_diagnostics_only_report_applicable_output_limit(self) -> None:
        for profile in (default_profile('OpenAI'), default_codex_profile()):
            with self.subTest(backend=profile.backend):
                profile.model = 'selected-model'
                profile.model_options = ['selected-model']
                profile.max_tokens = 1234
                with mock.patch.object(pcfg.module, 'llm_profiles', [profile]), \
                        mock.patch.object(pcfg.module, 'translator_llm_id', profile.id):
                    description = self.translator.translation_run_description()
                if profile.backend == 'codex':
                    self.assertNotIn('max_output_tokens=', description)
                else:
                    self.assertIn('max_output_tokens=1234', description)
                self.assertIn(f'model={profile.model!r}', description)
                self.assertIn('thinking_setting=', description)

    def test_legacy_provider_keeps_page_id_schema_and_message_content(self):
        profile = default_profile('LM Studio')
        messages = [{'role': 'system', 'content': 'stable prefix'}]

        one_item = self.translator._api_args(
            profile,
            messages,
            expected_translations=1,
        )
        three_items = self.translator._api_args(
            profile,
            messages,
            expected_translations=3,
        )

        self.assertIs(one_item['messages'], messages)
        self.assertIs(three_items['messages'], messages)
        self.assertEqual(
            one_item['response_format']['json_schema']['schema']['required'],
            ['1'],
        )
        self.assertEqual(
            three_items['response_format']['json_schema']['schema']['required'],
            ['1', '2', '3'],
        )

    def test_translate_passes_input_count_to_structured_request(self):
        profile = default_profile('LM Studio')
        with mock.patch.object(
            self.translator,
            '_request_translation',
            return_value='{"1":"甲","2":"乙","3":"丙"}',
        ) as request:
            result = self.translator._translate(
                ['a', 'b', 'c'],
                profile=profile,
                prompt_spec=self._prompt_spec(profile),
            )

        self.assertEqual(result, ['甲', '乙', '丙'])
        self.assertEqual(request.call_args.kwargs['expected_translations'], 3)

    def test_context_status_error_is_typed_and_preserves_provider_message(self):
        provider_message = (
            "This model's maximum context length is 4096 tokens, but the "
            'request used 5000 tokens.'
        )
        provider_error = FakeStatusError(provider_message)
        profile = default_profile('OpenAI')

        with mock.patch.object(
            self.translator,
            'request_chat_completion',
            side_effect=LLMChatRequestError(provider_error),
        ), self.assertRaisesRegex(
            ContextLengthError, 'maximum context length'
        ) as caught:
            self.translator._request_translation(
                profile, [{'role': 'user', 'content': 'x'}]
            )
        self.assertIs(caught.exception.__cause__, provider_error)

    def test_context_error_code_is_recognized_but_unrelated_errors_are_not(self):
        profile = default_profile('OpenAI')
        coded_error = FakeStatusError(
            'input rejected',
            code='context_length_exceeded',
        )
        with mock.patch.object(
            self.translator,
            'request_chat_completion',
            side_effect=LLMChatRequestError(coded_error),
        ), self.assertRaises(ContextLengthError):
            self.translator._request_translation(
                profile, [{'role': 'user', 'content': 'x'}]
            )

        unrelated_errors = (
            FakeStatusError('max_tokens must be less than 8192'),
            FakeStatusError('maximum context length exceeded', status_code=404),
            FakeStatusError('maximum context length exceeded', status_code=500),
            FakeStatusError('rate limit exceeded', status_code=429),
        )
        for error in unrelated_errors:
            with self.subTest(error=error.response.json()['error']['message']):
                with mock.patch.object(
                    self.translator,
                    'request_chat_completion',
                    side_effect=LLMChatRequestError(error),
                ), self.assertRaises(LLMChatRequestError) as caught:
                    self.translator._request_translation(
                        profile,
                        [{'role': 'user', 'content': 'x'}],
                    )
                self.assertNotIsInstance(caught.exception, ContextLengthError)

    def test_user_action_required_bypasses_ordinary_retries(self):
        profile = default_profile('OpenAI')
        self.translator.set_param_value('retry attempts', 5)
        self.translator.set_param_value('retry timeout', 0)

        with mock.patch.object(
            self.translator,
            '_request_translation',
            side_effect=LLMUserActionRequiredError('update the profile'),
        ) as request:
            with self.assertRaisesRegex(
                LLMUserActionRequiredError,
                'update the profile',
            ):
                self.translator._translate(
                    ['source'],
                    profile=profile,
                    prompt_spec=self._prompt_spec(profile),
                    page_key='003-0.png',
                )

        self.assertEqual(request.call_count, 1)

    def test_token_usage_supports_openai_and_deepseek_cache_fields(self):
        openai_usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details=SimpleNamespace(cached_tokens=80),
        )
        deepseek_usage = {
            'prompt_tokens': 100,
            'completion_tokens': 20,
            'completion_tokens_details': {'reasoning_tokens': 18},
            'total_tokens': 120,
            'prompt_cache_hit_tokens': 70,
            'prompt_cache_miss_tokens': 30,
        }
        self.assertEqual(
            format_token_usage(openai_usage),
            'prompt=100, completion=20, total=120, cache_hit=80',
        )
        self.assertEqual(
            format_token_usage(deepseek_usage),
            'prompt=100, completion=20, reasoning=18, total=120, '
            'cache_hit=70, cache_miss=30',
        )

        class IncompleteUsage:
            total_tokens = 3

            @property
            def prompt_tokens(self):
                raise RuntimeError('not available')

        self.assertEqual(
            format_token_usage(IncompleteUsage()),
            'total=3',
        )
        self.assertEqual(format_token_usage(None), '')


if __name__ == '__main__':
    unittest.main()
