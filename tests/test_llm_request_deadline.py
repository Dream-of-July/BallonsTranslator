import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from ballontranslator.modules.exceptions import LLMOutputLimitError, LLMRequestStopped
from ballontranslator.modules.llm_chat import LLMChatResult
from ballontranslator.modules.translators.trans_llm import LLMTranslator
from ballontranslator.utils.llm_profiles import default_codex_profile, default_profile


class LLMRequestDeadlineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.translator = LLMTranslator('日本語', '简体中文')
        self.profile = default_profile('DeepSeek')

    def test_old_signature_standin_accepts_openai_and_codex_requests(self) -> None:
        for profile in (self.profile, default_codex_profile()):
            profile.model, profile.model_options = 'offline-model', ['offline-model']
            turn = object() if profile.backend == 'codex' else None
            calls = []

            def respond(profile, api_args, *, codex_turn=None):
                calls.append((profile, api_args, codex_turn))
                return LLMChatResult(content='target')

            with self.subTest(backend=profile.backend), mock.patch.object(
                self.translator, 'request_chat_completion', side_effect=respond
            ):
                self.assertEqual(self.translator._request_translation(
                    profile, [], codex_turn=turn), 'target')
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0][2], turn)
            self.assertIsNone(self.translator._request_deadline.value)

    def test_deadline_is_cleared_after_success_failure_and_stop(self) -> None:
        for outcome in ('success', 'failure', 'stop'):
            stop = threading.Event()
            self.translator.set_stop_event(stop)

            def respond(profile, api_args, *, codex_turn=None):
                self.assertGreater(self.translator._request_deadline.value, time.monotonic())
                if outcome == 'failure':
                    raise TypeError('Programming error')
                if outcome == 'stop':
                    stop.set()
                    raise LLMRequestStopped()
                return LLMChatResult(content='target')

            with self.subTest(outcome=outcome), mock.patch.object(
                self.translator, 'request_chat_completion', side_effect=respond
            ):
                if outcome == 'success':
                    self.assertEqual(self.translator._request_translation(self.profile, []), 'target')
                else:
                    error = TypeError if outcome == 'failure' else LLMRequestStopped
                    with self.assertRaises(error):
                        self.translator._request_translation(self.profile, [])
            self.assertIsNone(self.translator._request_deadline.value)

        self.translator.set_stop_event(None)
        with mock.patch.object(self.translator, '_api_args', side_effect=ValueError('Invalid arguments')):
            with self.assertRaises(ValueError):
                self.translator._request_translation(self.profile, [])
        self.assertIsNone(self.translator._request_deadline.value)

    def test_deadline_is_isolated_between_instances_and_threads(self) -> None:
        other = LLMTranslator('日本語', '简体中文')
        entered, release = threading.Event(), threading.Event()
        deadlines, errors = [], []

        def respond(profile, api_args, *, codex_turn=None):
            deadlines.append(self.translator._request_deadline.value)
            entered.set()
            if not release.wait(2):
                raise AssertionError('Request was not released')
            self.assertEqual(self.translator._request_deadline.value, deadlines[0])
            return LLMChatResult(content='target')

        def run() -> None:
            try:
                self.assertEqual(self.translator._request_translation(self.profile, []), 'target')
                self.assertIsNone(self.translator._request_deadline.value)
            except BaseException as error:
                errors.append(error)

        with mock.patch.object(self.translator, 'request_chat_completion', side_effect=respond):
            worker = threading.Thread(target=run)
            worker.start()
            try:
                self.assertTrue(entered.wait(2))
                self.assertIsNone(getattr(self.translator._request_deadline, 'value', None))
                self.assertIsNone(getattr(other._request_deadline, 'value', None))

                def other_respond(profile, api_args, *, codex_turn=None):
                    self.assertGreater(other._request_deadline.value, deadlines[0])
                    return LLMChatResult(content='other')

                with mock.patch.object(other, 'request_chat_completion', side_effect=other_respond):
                    self.assertEqual(other._request_translation(self.profile, []), 'other')
                self.assertIsNone(other._request_deadline.value)
            finally:
                release.set()
                worker.join(2)
            self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])

    def test_real_requester_bounds_first_attempt_and_recovery_with_one_deadline(self) -> None:
        for exhausted in (False, True):
            release, finished = threading.Event(), threading.Event()
            calls, budgets = [], []
            bounded = self.translator._bounded_request

            def create(**api_args):
                calls.append(api_args)
                if exhausted and len(calls) == 1:
                    time.sleep(.08)
                    return SimpleNamespace(choices=[SimpleNamespace(
                        message=SimpleNamespace(content=''), finish_reason='length')])
                try:
                    release.wait(2)
                    return SimpleNamespace(choices=[])
                finally:
                    finished.set()

            def record(request, close, stop_event, deadline):
                budgets.append((deadline, deadline - time.monotonic()))
                return bounded(request, close, stop_event, deadline)

            client = SimpleNamespace(
                base_url='https://offline.test/v1', close=release.set,
                chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            sdk = SimpleNamespace(AuthenticationError=type('AuthError', (Exception,), {}),
                                  APIStatusError=type('StatusError', (Exception,), {}))
            with self.subTest(exhausted=exhausted), \
                    mock.patch.object(self.translator, '_request_timeout', return_value=.2), \
                    mock.patch.object(self.translator, '_openai_module', return_value=sdk), \
                    mock.patch.object(self.translator, '_initialize_client', return_value=client), \
                    mock.patch.object(self.translator, '_respect_delay'), \
                    mock.patch.object(self.translator, '_bounded_request', side_effect=record):
                start = time.monotonic()
                try:
                    with self.assertRaises(LLMOutputLimitError if exhausted else TimeoutError) as caught:
                        self.translator._request_translation(self.profile, [])
                    self.assertLess(time.monotonic() - start, .7)
                    self.assertTrue(finished.wait(1))
                    self.assertIsNone(self.translator._request_deadline.value)
                    self.assertEqual(len(calls), 2 if exhausted else 1)
                    if exhausted:
                        self.assertIsInstance(caught.exception.__cause__, TimeoutError)
                        self.assertEqual(budgets[0][0], budgets[1][0])
                        self.assertLess(budgets[1][1], budgets[0][1] - .05)
                        self.assertEqual(calls[1]['extra_body']['thinking'], {'type': 'disabled'})
                finally:
                    release.set()

    def test_waiting_request_stop_clears_deadline(self) -> None:
        stop, entered, release = threading.Event(), threading.Event(), threading.Event()
        self.translator.set_stop_event(stop)
        errors = []

        def create(**api_args):
            entered.set()
            release.wait(2)
            return SimpleNamespace(choices=[])

        def run() -> None:
            try:
                self.translator._request_translation(self.profile, [])
            except BaseException as error:
                errors.append(error)
            finally:
                errors.append(getattr(self.translator._request_deadline, 'value', None))

        client = SimpleNamespace(base_url='https://offline.test/v1', close=release.set,
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        sdk = SimpleNamespace(AuthenticationError=type('AuthError', (Exception,), {}),
                              APIStatusError=type('StatusError', (Exception,), {}))
        with mock.patch.object(self.translator, '_openai_module', return_value=sdk), \
                mock.patch.object(self.translator, '_initialize_client', return_value=client), \
                mock.patch.object(self.translator, '_respect_delay'):
            worker = threading.Thread(target=run)
            worker.start()
            try:
                self.assertTrue(entered.wait(2))
                stop.set()
                worker.join(1)
                self.assertFalse(worker.is_alive())
            finally:
                release.set()
                stop.set()
                worker.join(2)
        self.assertEqual(len(errors), 2)
        self.assertIsInstance(errors[0], LLMRequestStopped)
        self.assertIsNone(errors[1])
