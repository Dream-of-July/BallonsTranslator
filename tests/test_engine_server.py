"""Offline engine contract checks using a real hidden MainWindow."""
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('BALLOONTRANS_APP_ROOT', str(Path(__file__).resolve().parents[1]))


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ballontranslator.utils import shared
        from ballontranslator.utils.logger import logger
        # Bootstrap sets LDPI from the offscreen screen (72 on macOS); do not
        # leave later layout tests using the engine's point/pixel conversion.
        cls.enterClassContext(patch.multiple(shared,
            HEADLESS=shared.HEADLESS, LDPI=shared.LDPI, ENGINE_IMAGES_LOCAL=shared.ENGINE_IMAGES_LOCAL,
            ENGINE_TRANSLATION_AUTO_THINKING_DISABLED=shared.ENGINE_TRANSLATION_AUTO_THINKING_DISABLED))
        cls.enterClassContext(patch.object(logger, 'disabled', logger.disabled))
        cls.root = tempfile.TemporaryDirectory()
        cls.enterClassContext(patch.dict(os.environ,
            BALLOONTRANS_ENGINE_CONFIG=str(Path(cls.root.name) / 'config.json')))
        from ballontranslator.engine.bootstrap import bootstrap
        from qtpy.QtWidgets import QApplication
        # Qt retains font DPI even after an earlier QApplication is destroyed.
        # closingDown distinguishes that case from the first app in a process.
        cls.bootstrap_created_first_application = (
            QApplication.instance() is None and not QApplication.closingDown())
        cls.app, cls.window = bootstrap()
        from ballontranslator.engine.server import Server
        from ballontranslator.engine.transport import Writer
        cls.sink = io.StringIO()
        cls.server = Server(cls.app, cls.window, Writer(cls.sink))

    @classmethod
    def tearDownClass(cls):
        cls.server.dispose()
        cls.root.cleanup()

    def setUp(self):
        self.sink.seek(0)
        self.sink.truncate()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        from PIL import Image
        Image.new('RGB', (480, 320), 'white').save(self.directory / '001.png')
        block = dict(xyxy=[20, 20, 210, 110], lines=[[[20,20],[210,20],[210,110],[20,110]]],
                     _bounding_rect=[20,20,190,90], translation='Hello', text=['source'],
                     fontformat=dict(font_family='Arial', font_size=32, frgb=[0,0,0], stroke_width=0))
        other = copy.deepcopy(block)
        other.update(_bounding_rect=[240,180,190,90], xyxy=[240,180,430,270],
                     rich_text='<p style="font-size:18pt">Untouched</p>', translation='Untouched')
        data = dict(pages={'001.png': [block, other]}, current_img='001.png', image_info={})
        self.project_file = self.directory / ('imgtrans_' + self.directory.name + '.json')
        self.project_file.write_text(json.dumps(data))
        self.call('open_project', dict(path=str(self.directory)))

    def messages(self):
        return [json.loads(line) for line in self.sink.getvalue().splitlines()]

    def call(self, method, params=None):
        request_id = len(self.messages()) + 1
        self.server.dispatch(dict(id=request_id, method=method, params=params or {}))
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            self.app.processEvents()
            for message in self.messages():
                if message.get('id') == request_id:
                    return message
            time.sleep(.005)
        self.fail('response timed out: ' + method)

    def test_hello_page_fonts(self):
        hello = self.call('hello')['result']
        self.assertEqual(hello['protocol'], 1)
        self.assertEqual(hello['capabilities'], ['keep_original', 'vision_probe'])
        self.assertIn(hello['device'], ('cpu', 'mps', 'cuda', None))
        page = self.call('get_page', {'name': '001.png'})['result']
        self.assertEqual((page['width'], page['height']), (480, 320))
        self.assertEqual(len(page['blocks']), 2)
        self.assertEqual(page['blocks'][0]['source_text'], 'source')
        self.assertTrue(self.call('list_fonts')['result']['fonts'])

    def test_translation_auto_override_keeps_get_settings_unchanged(self) -> None:
        from ballontranslator.modules.llm_chat import LLMChatResult
        from ballontranslator.modules.translators.trans_llm import LLMTranslator
        from ballontranslator.utils import config, shared
        from ballontranslator.utils.llm_profiles import default_profile

        self.assertTrue(shared.ENGINE_TRANSLATION_AUTO_THINKING_DISABLED)
        profile = default_profile('DeepSeek')
        before_profile = copy.deepcopy(profile)
        translator = LLMTranslator('日本語', '简体中文')
        with patch.object(config.pcfg.module, 'llm_profiles', [profile]):
            before_settings = self.call('get_settings')['result']
            with patch.object(translator, 'request_chat_completion',
                    return_value=LLMChatResult(content='target')) as request:
                self.assertEqual(translator._request_translation(profile, []), 'target')
            self.assertEqual(request.call_args.args[1]['extra_body'], {'thinking': {'type': 'disabled'}})
            self.assertEqual(self.call('get_settings')['result'], before_settings)
        self.assertEqual(profile, before_profile)
        self.assertEqual(profile.thinking_level, 'Auto')

    def quality_page(self, translation: str) -> None:
        from PIL import Image, ImageDraw
        import numpy as np
        from ballontranslator.engine.manga_quality import draft_path, prepare_background
        from ballontranslator.utils.io_utils import imwrite

        original = Image.new('RGB', (480, 320), 'white')
        ImageDraw.Draw(original).rectangle((75, 45, 85, 85), fill='black')
        original.save(self.directory / '001.png')
        data = json.loads(self.project_file.read_text())
        block = data['pages']['001.png'][0]
        block.update(text=['原文'], translation=translation,
                     manga_area=[[20,20],[210,20],[210,110],[20,110]])
        self.project_file.write_text(json.dumps(data))
        self.call('open_project', dict(path=str(self.directory)))
        mask = np.zeros((320, 480), dtype=np.uint8)
        mask[44:87, 74:87] = 255
        draft = prepare_background(np.array(original), mask, self.server.project.pages['001.png'], None)
        path = draft_path(str(self.directory), '001.png')
        path.parent.mkdir(exist_ok=True)
        imwrite(str(path), draft)

    def test_keep_original_withheld_persists_and_false_refits(self):
        from PIL import Image
        import numpy as np

        self.quality_page('')
        withheld = self.call('update_blocks', dict(page='001.png', edits=[]))['result']
        self.assertEqual(withheld['blocks'][0]['review_reason'], 'untranslated')
        protected = self.call('update_blocks', dict(page='001.png',
            edits=[dict(index=0, keep_original=True)]))['result']
        self.assertEqual(protected['blocks'][0]['review_reason'], 'protected')
        self.assertNotIn('keep_original', protected['blocks'][0])
        self.assertTrue(np.array_equal(np.array(Image.open(protected['result']))[20:110,20:210],
                                       np.array(Image.open(self.directory/'001.png'))[20:110,20:210]))
        saved = json.loads(self.project_file.read_text())['pages']['001.png'][0]
        self.assertTrue(saved['manga_keep_original'])
        self.call('open_project', dict(path=str(self.directory)))
        self.assertTrue(self.server.project.pages['001.png'][0].manga_keep_original)
        other_edit = self.call('update_blocks', dict(page='001.png',
            edits=[dict(index=1, translation='Other edit')]))['result']
        self.assertEqual(other_edit['blocks'][0]['review_reason'], 'protected')
        # Continue skips a finished page and must not replace its saved blocks.
        from ballontranslator.utils.config import pcfg
        self.server.project.set_page_progress('001.png', pcfg.module.finish_code)
        self.call('set_settings', dict(translator=dict(kind='google')))
        with patch.object(self.window, 'on_run_imgtrans') as run:
            self.assertTrue(self.call('run', dict(mode='continue'))['result']['started'])
            run.assert_not_called()
        self.server.settle()
        self.assertTrue(self.server.project.pages['001.png'][0].manga_keep_original)
        released = self.call('update_blocks', dict(page='001.png',
            edits=[dict(index=0, keep_original=False, translation='中文')]))['result']
        self.assertIsNone(released['blocks'][0]['review_reason'])
        self.assertFalse(self.server.project.pages['001.png'][0].manga_keep_original)
        withheld_again = self.call('update_blocks', dict(page='001.png',
            edits=[dict(index=0, keep_original=False, translation='')]))['result']
        self.assertEqual(withheld_again['blocks'][0]['review_reason'], 'untranslated')

    def test_keep_original_restores_previously_translated_pixels(self):
        from PIL import Image
        import numpy as np

        self.quality_page('中文')
        translated = self.call('update_blocks', dict(page='001.png',
            edits=[dict(index=0, translation='中文')], autofit=True))['result']
        self.assertIsNone(translated['blocks'][0]['review_reason'])
        original = np.array(Image.open(self.directory/'001.png'))[20:110,20:210]
        self.assertFalse(np.array_equal(np.array(Image.open(translated['result']))[20:110,20:210], original))
        protected = self.call('update_blocks', dict(page='001.png',
            edits=[dict(index=0, keep_original=True)]))['result']
        self.assertEqual(protected['blocks'][0]['review_reason'], 'protected')
        self.assertTrue(np.array_equal(np.array(Image.open(protected['result']))[20:110,20:210], original))

    def test_keep_original_rejects_non_bool_and_legacy_page(self):
        for value in (True, False, 1, 0, 'true', None, []):
            response = self.call('update_blocks', dict(page='001.png',
                edits=[dict(index=0, keep_original=value)]))
            self.assertEqual(response['error']['code'], 'bad_request')
            if type(value) is bool:
                self.assertIn('manga-quality', response['error']['message'])
        self.assertFalse(self.server.project.pages['001.png'][0].manga_keep_original)
        self.quality_page('中文')
        for value in (1, 0, 'true', None, []):
            response = self.call('update_blocks', dict(page='001.png',
                edits=[dict(index=0, keep_original=value)]))
            self.assertEqual(response['error']['code'], 'bad_request')

    def test_edit_delete_real_pixels_and_persistence(self):
        from PIL import Image, ImageChops
        before_other = copy.deepcopy(self.server.project.pages['001.png'][1].__dict__)
        first = self.call('update_blocks', dict(page='001.png', edits=[]))['result']
        image_path = Path(first['result'])
        before = Image.open(image_path).copy()
        stamp = image_path.stat().st_mtime_ns
        edited = self.call('update_blocks', dict(page='001.png', edits=[dict(index=0,
            translation='XXXXX', rect=[30,30,190,90], font_family='Arial', font_size=48)]))
        self.assertIn('result', edited, edited)
        page = edited['result']
        self.assertGreater(page['revision'], first['revision'])
        self.assertGreater(image_path.stat().st_mtime_ns, stamp)
        after = Image.open(image_path).copy()
        self.assertIsNotNone(ImageChops.difference(before, after).crop((0,0,230,160)).getbbox())
        saved = json.loads(self.project_file.read_text())['pages']['001.png']
        self.assertEqual(saved[0]['translation'], 'XXXXX')
        self.assertEqual(saved[0]['rich_text'], '')
        self.assertEqual(saved[0]['fontformat']['font_size'], 48)
        self.assertEqual(saved[0]['_bounding_rect'][:2], [30,30])
        other = self.server.project.pages['001.png'][1]
        self.assertEqual(other.translation, before_other['translation'])
        self.assertEqual(other.rich_text, before_other['rich_text'])
        self.assertEqual(other.fontformat.__dict__, before_other['fontformat'].__dict__)
        deleted = self.call('delete_blocks', dict(page='001.png', indexes=[0]))['result']
        self.assertEqual(len(deleted['blocks']), 1)
        self.assertGreater(deleted['revision'], page['revision'])
        self.assertIsNotNone(ImageChops.difference(after, Image.open(image_path)).getbbox())
        self.assertEqual(len(json.loads(self.project_file.read_text())['pages']['001.png']), 1)
        self.assertEqual(self.call('export', dict(destination=str(self.directory/'export'), what='images'))['result']['files'],
                         [str(self.directory/'export'/'001.png')])

    def test_other_page_and_autofit_scope(self):
        from PIL import Image
        Image.new('RGB', (480,320), 'white').save(self.directory/'002.png')
        data = json.loads(self.project_file.read_text())
        data['pages']['002.png'] = copy.deepcopy(data['pages']['001.png'])
        self.project_file.write_text(json.dumps(data))
        self.call('open_project', dict(path=str(self.directory)))
        before = self.call('get_page', dict(name='002.png'))['result']['blocks']
        untouched = copy.deepcopy(self.server.project.pages['001.png'][1].__dict__)
        # Patch just the layout algorithm so the test needs no balloon model or
        # language tokenization download; scene construction/render/save are real.
        with patch.object(self.window.st_manager, 'layout_textblk') as layout:
            result = self.call('update_blocks', dict(page='001.png',
                edits=[dict(index=0, translation='Edited')], autofit=True))
            self.assertIn('result', result)
            self.assertEqual(layout.call_count, 1)
        self.assertFalse(self.window.st_manager.auto_textlayout_flag)
        self.assertEqual(self.call('get_page', dict(name='002.png'))['result']['blocks'], before)
        self.assertEqual(self.server.project.pages['001.png'][1].rich_text, untouched['rich_text'])
        self.call('update_blocks', dict(page='002.png', edits=[dict(index=0, translation='Second')]))
        self.assertEqual(self.call('get_page', dict(name='001.png'))['result']['blocks'][0]['translation'], 'Edited')

    def test_text_exports_and_async_profile(self):
        import threading
        from ballontranslator.modules.llm_chat import LLMConnectionTestResult
        from ballontranslator.utils import config
        settings = self.call('get_settings')['result']
        profile = self.call('create_profile', dict(template=settings['provider_templates'][0]))['result']
        entered, release = threading.Event(), threading.Event()
        main_thread = threading.get_ident()
        def connection(profile, *, proxy):
            self.assertNotEqual(threading.get_ident(), main_thread)
            self.assertEqual(proxy, 'http://127.0.0.1:8765')
            entered.set()
            release.wait(5)
            return LLMConnectionTestResult(True, 'ok', 'ok', [])
        config.pcfg.module.translator_params['LLMTranslator']['proxy'] = {'value': 'http://127.0.0.1:8765'}
        with patch('ballontranslator.modules.llm_chat.check_profile_connection', side_effect=connection):
            self.server.dispatch(dict(id=9001, method='test_profile', params=dict(id=profile['id'])))
            self.assertTrue(entered.wait(2))
            try:
                self.assertIn('result', self.call('hello'))
            finally:
                release.set()
            deadline = time.monotonic()+3
            while self.server.tests and time.monotonic()<deadline:
                self.app.processEvents()
                time.sleep(.005)
            self.assertFalse(self.server.tests)
        config.pcfg.module.translator_params['LLMTranslator']['proxy'] = {'value': ''}
        for what, expected in (('source_txt', 'source'), ('translation_txt', 'Hello')):
            result = self.call('export', dict(destination=str(self.directory/'export'), what=what))['result']
            self.assertIn(expected, Path(result['files'][0]).read_text())

    def vision_profile(self):
        from ballontranslator.utils import config, llm_profiles

        profile = llm_profiles.profile_from_template('DeepSeek')
        self.enterContext(patch.object(config.pcfg.module, 'llm_profiles', [profile]))
        self.enterContext(patch.object(llm_profiles, 'resolve_api_key', return_value=''))
        return profile

    def assert_saved_vision(self, profile, vision):
        from ballontranslator.utils import config, shared, llm_profiles

        # Bootstrap redirects this path to a fresh temporary config, never user data.
        saved = config.ProgramConfig.load(shared.CONFIG_PATH)
        loaded = llm_profiles.profile_by_id(saved.module.llm_profiles, profile.id)
        self.assertIs(loaded.vision_probe_result, vision == 'yes')
        self.assertIs(loaded.support_vision, vision == 'yes')
        self.assertEqual(self.server.public_profile(loaded)['vision'], vision)

    def test_vision_probe_correct_digits_persist_and_use_one_image_request(self):
        import base64
        import threading
        import cv2
        import numpy as np
        from unittest.mock import Mock
        from ballontranslator.utils import shared

        profile = self.vision_profile()
        profile.vision_model = 'offline-vision'
        before_thinking = profile.thinking_level
        before_local = shared.ENGINE_IMAGES_LOCAL
        main_thread = threading.get_ident()

        def respond(candidate, path, *, proxy, timeout, payload):
            self.assertNotEqual(threading.get_ident(), main_thread)
            self.assertEqual(path, 'chat/completions')
            self.assertEqual(timeout, 15.0)
            self.assertEqual(payload['model'], 'offline-vision')
            self.assertEqual(payload['thinking'], {'type': 'disabled'})
            self.assertNotIn('extra_body', payload)
            self.assertEqual(candidate.thinking_level, 'Disabled')
            self.assertNotIn('0042', json.dumps(payload['messages'][0]))
            text, image = payload['messages'][1]['content']
            self.assertNotIn('0042', text['text'])
            self.assertEqual(image['type'], 'image_url')
            encoded = image['image_url']['url'].split(',', 1)[1]
            pixels = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(pixels.shape, (96, 256, 3))
            self.assertLess(int(pixels.min()), 30)
            return Mock(json=lambda: {'choices': [{'message': {'content': '0042'}}]})

        with patch('ballontranslator.engine.vision_probe.secrets.randbelow', return_value=42), \
                patch('ballontranslator.engine.vision_probe._profile_connection_request', side_effect=respond) as request:
            self.assertEqual(self.call('probe_vision', dict(profile_id=profile.id))['result'], {'vision': 'yes'})
            request.assert_called_once()
        self.assertTrue(profile.support_vision)
        self.assertEqual(self.server.public_profile(profile)['vision'], 'yes')
        self.assert_saved_vision(profile, 'yes')
        self.assertEqual(profile.thinking_level, before_thinking)
        self.assertEqual(shared.ENGINE_IMAGES_LOCAL, before_local)

    def test_vision_probe_image_rejection_persists_no(self):
        from ballontranslator.modules.llm_chat import LLMConnectionError

        profile = self.vision_profile()
        with patch('ballontranslator.engine.vision_probe._profile_connection_request',
                   side_effect=LLMConnectionError('invalid_response', 'HTTP 400: Image unsupported.')) as request:
            self.assertEqual(self.call('probe_vision', dict(profile_id=profile.id))['result'], {'vision': 'no'})
            request.assert_called_once()
        self.assertFalse(profile.support_vision)
        self.assert_saved_vision(profile, 'no')

    def test_vision_probe_wrong_or_empty_answer_and_model_fallback(self):
        from unittest.mock import Mock

        profile = self.vision_profile()
        profile.vision_model = ''
        profile.thinking_level_options = ['Auto']
        for content in ('9999', '', None):
            with self.subTest(content=content), \
                    patch('ballontranslator.engine.vision_probe.secrets.randbelow', return_value=42), \
                    patch('ballontranslator.engine.vision_probe._profile_connection_request',
                          return_value=Mock(json=lambda: {'choices': [{'message': {'content': content}}]})) as request:
                self.assertEqual(self.call('probe_vision', dict(profile_id=profile.id))['result'], {'vision': 'no'})
                payload = request.call_args.kwargs['payload']
                self.assertEqual(payload['model'], profile.model)
                self.assertNotIn('thinking', payload)
        self.assert_saved_vision(profile, 'no')

    def test_vision_probe_connection_errors_preserve_stored_state(self):
        from ballontranslator.modules.llm_chat import LLMConnectionError
        from ballontranslator.modules.exceptions import LLMApiKeyRequiredError

        profile = self.vision_profile()
        failures = [(kind, LLMConnectionError(kind, 'unsafe-provider-detail'))
                    for kind in ('auth', 'timeout', 'network', 'server', 'rate_limited')]
        failures.append(('auth', LLMApiKeyRequiredError(profile.id)))
        for measured in (None, True, False):
            profile.vision_probe_result = measured
            profile.support_vision = measured is not False
            for kind, failure in failures:
                before = copy.deepcopy(profile)
                with self.subTest(measured=measured, kind=kind), \
                        patch('ballontranslator.engine.vision_probe._profile_connection_request', side_effect=failure), \
                        patch.object(self.server, 'persist') as persist:
                    result = self.call('probe_vision', dict(profile_id=profile.id))['result']
                    self.assertEqual(result, dict(ok=False, message='Connection failed.', kind=kind, models=[]))
                    persist.assert_not_called()
                    self.assertEqual(profile, before)
        self.assertNotIn('unsafe-provider-detail', self.sink.getvalue())

    def test_vision_template_and_legacy_profiles_are_unknown(self):
        from ballontranslator.utils import llm_profiles
        from ballontranslator.utils.logger import logger

        profile = self.vision_profile()
        self.assertTrue(profile.support_vision)
        self.assertEqual(self.server.public_profile(profile)['vision'], 'unknown')
        legacy = profile.to_dict()
        legacy.pop('vision_probe_result')
        with patch.object(logger, 'warning') as warning:
            loaded = llm_profiles.profile_by_id(llm_profiles.load_profiles([legacy]), profile.id)
            warning.assert_not_called()
        self.assertIsNone(loaded.vision_probe_result)
        self.assertEqual(self.server.public_profile(loaded)['vision'], 'unknown')

    def test_vision_update_model_or_endpoint_resets_measurement(self):
        from ballontranslator.utils import config, shared, llm_profiles

        profile = self.vision_profile()
        for key, value in (('model', 'different-model'), ('base_url', 'https://offline.test/v1')):
            profile.vision_probe_result = True
            with self.subTest(key=key):
                unchanged = self.call('update_profile', dict(id=profile.id, **{key: getattr(profile, key)}))
                self.assertEqual(unchanged['result']['vision'], 'yes')
                changed = self.call('update_profile', dict(id=profile.id, **{key: value}))
                self.assertEqual(changed['result']['vision'], 'unknown')
                saved = config.ProgramConfig.load(shared.CONFIG_PATH)
                self.assertIsNone(llm_profiles.profile_by_id(saved.module.llm_profiles, profile.id).vision_probe_result)

    def test_vision_probe_pending_profile_edit_discards_stale_measurement(self):
        import threading
        from unittest.mock import Mock

        profile = self.vision_profile()
        entered, release = threading.Event(), threading.Event()

        def respond(*args, **kwargs):
            entered.set()
            release.wait(5)
            return Mock(json=lambda: {'choices': [{'message': {'content': '0042'}}]})

        with patch('ballontranslator.engine.vision_probe.secrets.randbelow', return_value=42), \
                patch('ballontranslator.engine.vision_probe._profile_connection_request', side_effect=respond):
            self.server.dispatch(dict(id=9002, method='probe_vision', params=dict(profile_id=profile.id)))
            try:
                self.assertTrue(entered.wait(2))
                result = self.call('update_profile', dict(id=profile.id, model='replacement-model'))
                self.assertEqual(result['result']['vision'], 'unknown')
            finally:
                release.set()
            deadline = time.monotonic() + 3
            while self.server.tests and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(.005)
            self.assertFalse(self.server.tests)
        message = next(m for m in self.messages() if m.get('id') == 9002)
        self.assertEqual(message['error']['code'], 'busy')
        self.assertIsNone(profile.vision_probe_result)

    def test_vision_probe_rejects_invalid_parameters(self):
        profile = self.vision_profile()
        for params, code in ((dict(id=profile.id), 'bad_request'),
                             (dict(profile_id=42), 'bad_request'),
                             (dict(profile_id='missing-profile'), 'not_found')):
            with self.subTest(params=params):
                self.assertEqual(self.call('probe_vision', params)['error']['code'], code)

    def test_vision_probe_save_failure_restores_previous_state(self):
        from unittest.mock import Mock

        profile = self.vision_profile()
        before = copy.deepcopy(profile)
        with patch('ballontranslator.engine.vision_probe._profile_connection_request',
                   return_value=Mock(json=lambda: {'choices': []})), \
                patch.object(self.server, 'persist', side_effect=OSError('unsafe-save-detail')):
            result = self.call('probe_vision', dict(profile_id=profile.id))
        self.assertEqual(result['error']['code'], 'internal')
        self.assertEqual(profile, before)
        self.assertNotIn('unsafe-save-detail', self.sink.getvalue())

    def test_bad_requests_and_busy(self):
        self.assertEqual(self.call('unknown')['error']['code'], 'bad_request')
        for edits in ([dict(index=-1)], [dict(index=0, font_size=-1)], [dict(index=0, rect=[0,0,-1,20])]):
            self.assertEqual(self.call('update_blocks', dict(page='001.png', edits=edits))['error']['code'], 'bad_request')
        self.server.active = True
        try:
            self.assertEqual(self.call('run')['error']['code'], 'busy')
            self.assertEqual(self.call('update_blocks', dict(page='001.png', edits=[]))['error']['code'], 'busy')
        finally:
            self.server.active = False

    def test_settings_profiles_preflight_and_worker(self):
        from ballontranslator.utils import config
        from ballontranslator.modules.llm_chat import LLMConnectionTestResult
        template = self.call('get_settings')['result']['provider_templates'][0]
        profile = self.call('create_profile', dict(template=template))['result']
        pid = profile['id']
        # Synthetic sentinel, never a real credential.
        sentinel = 'unit-test-synthetic-secret'
        self.call('update_profile', dict(id=pid, api_key=sentinel, model='offline-test'))
        self.assertNotIn(sentinel, self.sink.getvalue())
        self.assertNotIn(sentinel, Path(config.shared.CONFIG_PATH).read_text())
        result = self.call('set_settings', dict(translator=dict(kind='llm', profile_id=pid), output=dict(format='.png', quality=95)))
        self.assertEqual(result['result']['translator']['profile_id'], pid)
        with patch('ballontranslator.modules.llm_chat.check_profile_connection',
                   return_value=LLMConnectionTestResult(True, 'ok', 'ok', ['offline-test'])) as check:
            result = self.call('test_profile', dict(id=pid))['result']
            self.assertTrue(result['ok'])
            check.assert_called_once()
        self.call('update_profile', dict(id=pid, api_key='', model=''))
        error = self.call('run')['error']
        self.assertEqual(error['code'], 'translator_not_ready')
        self.assertEqual(error['data']['missing'], 'model')
        self.call('update_profile', dict(id=pid, model='offline-test', base_url=''))
        self.assertEqual(self.call('run')['error']['data']['missing'], 'base_url')
        self.call('update_profile', dict(id=pid, base_url='https://example.invalid/v1'))
        self.server.profile(pid).require_api_key = True
        self.assertEqual(self.call('run')['error']['data']['missing'], 'api_key')
        self.call('set_settings', dict(translator=dict(kind='google')))
        self.assertIn('result', self.call('delete_profile', dict(id=pid)))
        self.assertTrue(Path(config.shared.CONFIG_PATH).is_file())

    def test_orchestration_order_and_cancel(self):
        manager = self.server.manager
        self.call('set_settings', dict(translator=dict(kind='google')))
        with patch.object(manager, '_prepare_modules_then'):
            self.assertTrue(self.call('run')['result']['started'])
            self.assertEqual(self.call('run')['error']['code'], 'busy')
            thread = manager.imgtrans_thread
            thread.num_pages = 1
            thread.parallel_trans = False
            thread.detect_counter = 1
            thread.ocr_counter = thread.translate_counter = thread.inpaint_counter = 0
            thread.update_detect_progress.emit(1)
            self.app.processEvents()
            # Use the real image writer; only inference is stubbed.
            from qtpy.QtGui import QImage, QColor
            image = QImage(480,320,QImage.Format.Format_RGB32)
            image.fill(QColor('white'))
            Path(self.server.project.result_dir()).mkdir(exist_ok=True)
            self.window.imsave_thread.saveImg(self.server.project.get_result_path('001.png'), image, '001.png')
            self.window.imsave_thread.wait_for_saves(None)
            manager.imgtrans_pipeline_finished.emit()
            deadline = time.monotonic()+3
            while self.server.active and time.monotonic()<deadline:
                self.app.processEvents()
                time.sleep(.01)
            events = [m['event'] for m in self.messages() if 'event' in m]
            self.assertLess(events.index('stage_progress'), events.index('page_done'))
            self.assertLess(events.index('page_done'), events.index('run_finished'))
            self.call('run')
            self.call('cancel')
            manager.imgtrans_pipeline_finished.emit()
            self.server.settle()
            self.assertTrue([m for m in self.messages() if m.get('event')=='run_finished'][-1]['cancelled'])

    def test_structured_page_failure(self):
        self.server.active = True
        self.server.pages = ['001.png']
        self.server.capture_page_error(ValueError('private detail'), '', '', '001.png')
        self.app.processEvents()
        self.server.pipeline_finished()
        self.server.settle()
        self.assertEqual([m for m in self.messages() if m.get('event')=='run_finished'][-1]['failed_pages'], ['001.png'])
        self.assertNotIn('private detail', self.sink.getvalue())

    def test_watchdog_stops_stalled_run_and_allows_next_run(self):
        manager = self.server.manager
        self.call('set_settings', dict(translator=dict(kind='google')))
        with patch.dict(os.environ, BALLOONTRANS_ENGINE_STALL_SECONDS='1'), \
                patch.object(manager, '_prepare_modules_then'), \
                patch.object(manager, 'stopImgtransPipeline', wraps=manager.stopImgtransPipeline) as stop:
            self.assertTrue(self.call('run')['result']['started'])
            deadline = time.monotonic() + 3
            while self.server.active and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(.01)
            self.assertFalse(self.server.active)
            stop.assert_called_once()
            events = [m for m in self.messages() if 'event' in m]
            errors = [m for m in events if m['event'] == 'error']
            self.assertEqual(errors, [dict(event='error',
                message='The run stopped because a stage stopped responding.')])
            finished = [m for m in events if m['event'] == 'run_finished']
            self.assertEqual(len(finished), 1)
            self.assertEqual(finished[0]['failed_pages'], ['001.png'])
            self.assertFalse(finished[0]['cancelled'])
            self.assertLess(events.index(errors[0]), events.index(finished[0]))
            self.assertTrue(self.call('run')['result']['started'])
            self.call('cancel')
            self.server.settle()
            self.assertFalse(self.server.active)

    def test_watchdog_exempts_silent_model_preparation_and_packages(self):
        manager = self.server.manager
        self.call('set_settings', dict(translator=dict(kind='google')))
        for module, worker in (('detector', manager.textdetect_thread),
                               ('packages', manager.package_install_thread)):
            with self.subTest(module=module), \
                    patch.dict(os.environ, BALLOONTRANS_ENGINE_STALL_SECONDS='1'), \
                    patch.object(manager, '_prepare_modules_then'), \
                    patch.object(worker, 'isRunning', return_value=True):
                self.assertTrue(self.call('run')['result']['started'])
                self.server.prepare_progress(module, {'event': 'loading_model'})
                deadline = time.monotonic() + 1.2
                while time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(.01)
                self.assertTrue(self.server.active)
                self.assertFalse(any(m.get('event') == 'error' for m in self.messages()))
            self.server.check_stall()
            self.assertTrue(self.server.active)
            self.call('cancel')
            self.server.settle()
            self.assertFalse(self.server.active)

    def test_watchdog_tracks_progress_and_silent_prepare_queue(self):
        manager = self.server.manager
        self.call('set_settings', dict(translator=dict(kind='google')))
        with patch.dict(os.environ, BALLOONTRANS_ENGINE_STALL_SECONDS='1'), \
                patch.object(manager, '_prepare_modules_then'):
            self.call('run')
            self.server.last_progress = time.monotonic() - 2
            with patch.object(manager, '_pending_prepare_success', object()):
                self.server.check_stall()
                self.assertFalse(self.server.stalled)
            self.server.stage_progress('translate', 1)
            stamp = self.server.last_progress
            self.server.stage_progress('translate', 1)
            self.assertEqual(self.server.last_progress, stamp)
            self.server.check_stall()
            self.assertFalse(self.server.stalled)
            self.call('cancel')
            self.server.settle()

    def test_dpi_ink_measurement(self):
        from ballontranslator.utils import shared
        from ballontranslator.utils.textblock import TextBlock
        from ballontranslator.ui.text_engine.item import TextBlkItem
        from qtpy.QtWidgets import QGraphicsScene
        from qtpy.QtGui import QImage, QPainter, QColor
        import numpy as np
        dpi = shared.LDPI
        def ink(ldpi, html=''):
            shared.LDPI = ldpi
            blk = TextBlock(xyxy=[0,0,300,150], _bounding_rect=[0,0,300,150],
                            translation='H', rich_text=html,
                            fontformat=dict(font_family='Arial',font_size=48,frgb=[0,0,0],stroke_width=0))
            item = TextBlkItem(blk, 0)
            scene = QGraphicsScene()
            scene.addItem(item)
            image = QImage(300,150,QImage.Format.Format_RGB32)
            image.fill(QColor('white'))
            painter = QPainter(image)
            scene.render(painter)
            painter.end()
            ptr = image.bits()
            ptr.setsize(image.sizeInBytes())
            data = np.frombuffer(ptr, np.uint8).reshape(150,300,4)
            rows = np.where((data[:,:,:3] < 100).any(axis=2).any(axis=1))[0]
            result = (int(rows[-1]-rows[0]+1), item.toHtml())
            scene.clear()
            return result
        try:
            height, html = ink(dpi)
            other_height, _ = ink(dpi*1.5)
            rich_height, _ = ink(dpi, html)
            print('ENGINE_DPI_MEASUREMENT', dict(screen_dpi=dpi, font_px=48, ink_height=height,
                  mismatched_ldpi=dpi*1.5, mismatched_ink_height=other_height, html_ink_height=rich_height))
            self.assertTrue(28 <= height <= 45)
            self.assertLess(other_height, height)
            self.assertEqual(rich_height, height)
            self.assertEqual(dpi, self.app.primaryScreen().logicalDotsPerInch())
            if sys.platform == 'darwin' and self.bootstrap_created_first_application:
                self.assertEqual(round(dpi), 72)
        finally:
            shared.LDPI = dpi


class TransportTests(unittest.TestCase):
    def test_engine_translation_flag_restored(self) -> None:
        from ballontranslator.utils import shared

        self.assertFalse(shared.ENGINE_TRANSLATION_AUTO_THINKING_DISABLED)

    def test_real_pipes_print_isolation_framing_and_eof(self):
        code = """
from ballontranslator.engine.transport import Writer, isolate_stdout, decode, ProtocolError
import sys, os
writer = Writer(isolate_stdout())
print('stray print')
os.write(1, b'native print\\n')
for line in sys.stdin:
    try:
        request=decode(line)
        writer.send({'id':request.get('id'), 'result':True})
    except ProtocolError as e:
        writer.send({'id':None,'error':{'code':e.code}})
"""
        process = subprocess.run([sys.executable, '-c', code], input='{"id":1}\nnot json\n',
                                 text=True, capture_output=True, timeout=10)
        self.assertEqual(process.returncode, 0)
        rows = [json.loads(s) for s in process.stdout.splitlines()]
        self.assertEqual(rows[0]['id'], 1)
        self.assertEqual(rows[1]['error']['code'], 'bad_request')
        self.assertIn('stray print', process.stderr)
        self.assertIn('native print', process.stderr)

    def test_end_to_end_engine_eof(self):
        with tempfile.TemporaryDirectory() as temp:
            env = dict(os.environ, BALLOONTRANS_ENGINE_CONFIG=str(Path(temp)/'config.json'))
            process = subprocess.run([sys.executable, '-m', 'ballontranslator.engine'],
                input='{"id":1,"method":"hello"}\n{"id":2,"method":"invalid"}\n{\n',
                text=True, capture_output=True, env=env, timeout=60)
        self.assertEqual(process.returncode, 0, process.stderr[-1000:])
        messages = [json.loads(line) for line in process.stdout.splitlines()]
        self.assertEqual(next(m for m in messages if m.get('id')==1)['result']['protocol'], 1)
        self.assertEqual(next(m for m in messages if m.get('id')==2)['error']['code'], 'bad_request')
        self.assertTrue(any(m.get('error',{}).get('code')=='bad_request' and m['id'] is None for m in messages))


if __name__ == '__main__':
    unittest.main()
