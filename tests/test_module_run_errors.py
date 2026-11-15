import unittest
import threading
import time
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import Mock, patch

import numpy as np
from qtpy.QtTest import QSignalSpy
from qtpy.QtWidgets import QApplication

from ballontranslator.modules.exceptions import ModuleRunError
from ballontranslator.modules.ocr.base import OCRBase, postprocess_ocr_text
from ballontranslator.modules.exceptions import (
    CodexSignInRequiredError,
    LLMApiKeyRequiredError,
    LLMAuthenticationError,
    LLMBaseURLRequiredError,
    LLMMemoryCompactionError,
    LLMModelRequiredError,
    LLMOutputLimitError,
    LLMRequestStopped,
    LLMUserActionRequiredError,
)
from ballontranslator.modules.textdetector.base import TextDetectorBase
from ballontranslator.ui import module_manager
from ballontranslator.utils.config import ModuleConfig, OCRTextPostprocess, RunStatus, pcfg
from ballontranslator.utils.registry import ModuleSpec
from ballontranslator.utils.textblock import TextBlock
from ballontranslator.utils.text_processing import capitalize_sentences
from ballontranslator.utils import shared
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.py_package_manager import MissingModuleRequirements
from ballontranslator.modules.translators.trans_google import TransGoogle


class SuccessfulOCR(OCRBase):
    def _ocr_blk_list(self, _img, blk_list, *args, **kwargs):
        blk_list[0].text = ['new text']


class FailingOCR(OCRBase):
    def _ocr_blk_list(self, _img, blk_list, *args, **kwargs):
        blk_list[0].text = ['partial text']
        raise ValueError('ocr backend failed')


class MissingKeyOCR(OCRBase):
    def _ocr_blk_list(self, _img, blk_list, *args, **kwargs):
        blk_list[0].text = ['partial text']
        raise LLMApiKeyRequiredError('profile-1', 'Profile 1')


class MissingModelOCR(OCRBase):
    def _ocr_blk_list(self, _img, blk_list, *args, **kwargs):
        blk_list[0].text = ['partial text']
        raise LLMModelRequiredError('profile-1', 'Profile 1', vision=True)


class FailingDetector(TextDetectorBase):
    def _detect(self, _img, _proj):
        raise ValueError('detector backend failed')

    def setup_detector(self):
        pass


class ReloadableOCR(OCRBase):
    _load_model_keys = {'model'}
    params = {}

    def __init__(self):
        super().__init__()
        self.name = 'reloadable_ocr'
        self.model = None
        self.load_count = 0

    def _load_model(self):
        self.load_count += 1
        self.model = object()


class FakeOCRRegistry:
    def get_spec(self, module_name):
        return ModuleSpec(
            key=module_name,
            import_path='tests.fake_ocr',
            class_name='ReloadableOCR',
            dependencies=[],
        )

    def resolve_module(self, _module_name):
        return ReloadableOCR


class FakeManager:
    def __init__(self, thread):
        self.thread = thread

    def _thread_for_module_key(self, _module_key):
        return self.thread


class ModuleRunErrorTest(unittest.TestCase):

    def test_ocr_postprocesses_new_text_only_after_success(self):
        ocr = SuccessfulOCR()
        block = TextBlock(xyxy=[0, 0, 10, 10], text=['old text'])
        substitutions = [{
            'keyword': 'new',
            'sub': 'final',
            'use_reg': False,
            'case_sens': True,
        }]

        with patch.object(pcfg, 'ocr_sublist', substitutions):
            with patch.object(
                pcfg.module,
                'ocr_text_postprocess',
                OCRTextPostprocess.UPPERCASE,
            ):
                ocr.run_ocr(
                    np.zeros((12, 12, 3), dtype=np.uint8),
                    [block],
                )

        self.assertEqual(block.text, 'FINAL TEXT')

    def test_ocr_restores_existing_text_after_failure(self):
        ocr = FailingOCR()
        ocr.name = 'failing_ocr'
        block = TextBlock(xyxy=[0, 0, 10, 10], text=['old text'])

        with self.assertRaises(ModuleRunError) as caught:
            ocr.run_ocr(np.zeros((12, 12, 3), dtype=np.uint8), [block])

        self.assertEqual(block.text, ['old text'])
        self.assertEqual(caught.exception.module_key, 'ocr')
        self.assertEqual(caught.exception.module_name, 'failing_ocr')
        self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_ocr_preserves_llm_key_error_after_restoring_text(self):
        ocr = MissingKeyOCR()
        ocr.name = 'LLMOCR'
        block = TextBlock(xyxy=[0, 0, 10, 10], text=['old text'])

        with self.assertRaises(LLMApiKeyRequiredError):
            ocr.run_ocr(np.zeros((12, 12, 3), dtype=np.uint8), [block])

        self.assertEqual(block.text, ['old text'])

    def test_ocr_preserves_llm_model_error_after_restoring_text(self):
        ocr = MissingModelOCR()
        ocr.name = 'LLMOCR'
        block = TextBlock(xyxy=[0, 0, 10, 10], text=['old text'])

        with self.assertRaises(LLMModelRequiredError):
            ocr.run_ocr(np.zeros((12, 12, 3), dtype=np.uint8), [block])

        self.assertEqual(block.text, ['old text'])

    def test_text_detector_wraps_runtime_failure(self):
        detector = FailingDetector()
        detector.name = 'failing_detector'

        with self.assertRaises(ModuleRunError) as caught:
            detector.detect(np.zeros((12, 12, 3), dtype=np.uint8))

        self.assertEqual(caught.exception.module_key, 'textdetector')
        self.assertEqual(caught.exception.module_name, 'failing_detector')
        self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_unloaded_same_module_is_not_ready(self):
        thread = module_manager.ModuleThread('ocr', FakeOCRRegistry())
        thread.module = ReloadableOCR()
        manager = FakeManager(thread)

        ready = module_manager.ModuleManager._module_ready(
            manager, 'ocr', 'reloadable_ocr'
        )

        self.assertFalse(ready)

    def test_same_module_is_reloaded_through_prepare_thread(self):
        thread = module_manager.ModuleThread('ocr', FakeOCRRegistry())
        module = ReloadableOCR()
        thread.module = module

        thread._set_module('reloadable_ocr')

        self.assertTrue(thread.last_set_success)
        self.assertIs(thread.module, module)
        self.assertTrue(module.all_model_loaded())
        self.assertEqual(module.load_count, 1)


class PipelineFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.project = ProjImgTrans()
        self.project.pages = {'page.png': [TextBlock(text=['source'], translation='old')]}
        self.project._image_info = {'page.png': {'finish_code': 0}}
        self.translate = module_manager.TranslateThread()
        self.thread = module_manager.ImgtransThread(None, None, self.translate, None)
        self.manager = module_manager.ModuleManager(self.project)
        self.manager.progress_msgbox = Mock()
        self.thread.pipeline_stopped.connect(self.manager.on_imgtrans_thread_stopped)
        self.stopped = QSignalSpy(self.thread.pipeline_stopped)
        self.finished = QSignalSpy(self.manager.imgtrans_pipeline_finished)

    def tearDown(self) -> None:
        self.thread.requestStop()
        if self.translate.pipeline_stop_event is not None:
            self.translate.pipeline_stop_event.set()
        self.assertTrue(self.thread.wait(3000))
        self.assertTrue(self.translate.wait(3000))
        self.app.processEvents()

    def _wait_for_shutdown(self) -> None:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            self.app.processEvents()
            if not self.thread.isRunning() and not self.translate.isRunning() and len(self.stopped):
                break
            time.sleep(0.005)
        self.assertTrue(self.thread.wait(100))
        self.assertTrue(self.translate.wait(100))
        self.app.processEvents()
        self.assertTrue(self.thread.isStopRequested())
        self.assertEqual(len(self.stopped), 1)
        self.assertEqual(len(self.finished), 1)
        self.manager.progress_msgbox.hide.assert_called_once()
        self.thread.on_module_thread_stopped()
        self.assertEqual(len(self.stopped), 1)

    def test_unexpected_exception_stops_run_and_next_run_can_start(self) -> None:
        self.thread.job = Mock(side_effect=ValueError('unexpected job failure'))
        with patch.object(module_manager, 'create_error_dialog') as dialog:
            self.thread.start()
            self._wait_for_shutdown()
        dialog.assert_called_once()
        self.assertIsNone(self.thread.job)

        self.manager.translate_thread = self.translate
        self.manager.terminateRunningThread = Mock()
        self.manager._thread_for_module_key = Mock(return_value=self.translate)
        self.manager._module_ready = Mock(return_value=True)
        self.manager._startImgtransPipeline = Mock(side_effect=lambda _pages: self.thread.runImgtransPipeline(self.project))
        with patch.object(pcfg.module, 'enable_detect', False), \
                patch.object(pcfg.module, 'enable_ocr', False), \
                patch.object(pcfg.module, 'enable_translate', True), \
                patch.object(pcfg.module, 'enable_inpaint', False), \
                patch.object(self.thread, '_imgtrans_pipeline') as next_job:
            self.manager.runImgtransPipeline()
            self.assertTrue(self.thread.wait(3000))
            self.app.processEvents()
        next_job.assert_called_once()
        self.assertFalse(self.thread.isStopRequested())
        self.assertIsNone(self.manager._pending_prepare_success)
        self.assertEqual(len(self.stopped), 1)

    def test_unexpected_exception_stops_waiting_parallel_translation(self) -> None:
        waiting = threading.Event()
        wait_for_stop = self.thread.stop_event.wait

        def wait(timeout: float) -> bool:
            waiting.set()
            return wait_for_stop(timeout)

        self.translate.translator = SimpleNamespace(delay=lambda: 0)
        self.translate.num_process_pages = 1
        with patch.object(self.thread.stop_event, 'wait', side_effect=wait), \
                patch.object(module_manager, 'create_error_dialog') as dialog:
            self.translate.runTranslatePipeline(self.project, self.thread.stop_event)
            self.assertTrue(waiting.wait(2))
            self.thread.job = Mock(side_effect=ValueError('unexpected job failure'))
            self.thread.start()
            self._wait_for_shutdown()
        dialog.assert_called_once()
        self.assertEqual(self.translate.finished_counter, 0)

    def test_stopped_emits_when_parallel_translation_finishes_last(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def delay() -> float:
            entered.set()
            release.wait(3)
            return 0

        self.translate.translator = SimpleNamespace(delay=delay)
        self.translate.num_process_pages = 1
        self.translate.runTranslatePipeline(self.project, self.thread.stop_event)
        try:
            self.assertTrue(entered.wait(2))
            self.thread.job = Mock(side_effect=ValueError('unexpected job failure'))
            with patch.object(module_manager, 'create_error_dialog') as dialog:
                self.thread.start()
                self.assertTrue(self.thread.wait(2000))
                self.assertTrue(self.translate.isRunning())
                release.set()
                self._wait_for_shutdown()
            dialog.assert_called_once()
            self.assertEqual(self.translate.finished_counter, 0)
        finally:
            release.set()

    def test_stopped_emits_if_translator_finishes_during_imgtrans_unwind(self) -> None:
        entered = threading.Event()
        release_translation = threading.Event()
        unwinding = threading.Event()
        release_imgtrans = threading.Event()

        def delay() -> float:
            entered.set()
            release_translation.wait(3)
            return 0

        emit_if_ready = self.thread._emit_pipeline_stopped_if_ready

        def hold_unwind(imgtrans_running: bool) -> None:
            emit_if_ready(imgtrans_running)
            if not imgtrans_running:
                release_translation.set()
                self.translate.wait(2000)
                unwinding.set()
                release_imgtrans.wait(3)

        self.translate.translator = SimpleNamespace(delay=delay)
        self.translate.num_process_pages = 1
        self.translate.runTranslatePipeline(self.project, self.thread.stop_event)
        try:
            self.assertTrue(entered.wait(2))
            self.thread.job = Mock(side_effect=ValueError('unexpected job failure'))
            with patch.object(module_manager, 'create_error_dialog'), \
                    patch.object(self.thread, '_emit_pipeline_stopped_if_ready', side_effect=hold_unwind):
                self.thread.start()
                self.assertTrue(unwinding.wait(2))
                self.app.processEvents()
                self.assertEqual(len(self.stopped), 0)
                release_imgtrans.set()
                self._wait_for_shutdown()
        finally:
            release_translation.set()
            release_imgtrans.set()

    def test_google_failure_does_not_mark_page_finished_and_shows_error(self) -> None:
        translator = TransGoogle('English', '简体中文')
        self.addCleanup(translator.internal_google_translator.requests_session.close)
        self.translate.translator = self.translate.module = translator
        block = self.project.pages['page.png'][0]
        with patch.object(translator.internal_google_translator.requests_session, 'request',
                          side_effect=ConnectionError('network unavailable')), \
                patch.object(module_manager, '_mark_translation_finished') as mark, \
                patch.object(module_manager, 'create_error_dialog') as dialog:
            self.assertFalse(self.thread._translate_full_page(self.project, 'page.png', [block]))
            self.assertFalse(self.translate._translate_page(self.project, 'page.png'))
        mark.assert_not_called()
        self.assertEqual(dialog.call_count, 2)
        self.assertEqual(block.translation, 'old')

    def test_declined_dependencies_notify_once_and_leave_no_pending_preparation(self) -> None:
        missing = [MissingModuleRequirements('translator', 'google', ['torch'])]
        self.manager.terminateRunningThread = Mock()
        self.manager._thread_for_module_key = Mock(return_value=self.translate)
        self.manager._module_ready = Mock(return_value=False)
        self.manager._missing_module_requirements_for_modules = Mock(return_value=missing)
        self.manager._startImgtransPipeline = Mock()
        self.manager.package_install_thread = Mock()
        self.manager.package_install_thread.isRunning.return_value = False

        for headless, install_accepted in ((False, False), (False, True), (True, False)):
            with self.subTest(headless=headless, install_accepted=install_accepted), \
                    patch.object(shared, 'HEADLESS', headless), \
                    patch.object(pcfg.package_manager, 'auto_install_missing_packages', False), \
                    patch.object(pcfg.module, 'enable_detect', False), \
                    patch.object(pcfg.module, 'enable_ocr', False), \
                    patch.object(pcfg.module, 'enable_translate', True), \
                    patch.object(pcfg.module, 'enable_inpaint', False), \
                    patch.object(self.manager, '_ask_install_missing_package_batch', return_value=install_accepted), \
                    patch.object(self.manager, '_confirm_torch_install_device', return_value=(False, None, None)), \
                    patch.object(module_manager, 'create_info_dialog') as info, \
                    patch.object(module_manager.LOGGER, 'error'):
                self.manager.runImgtransPipeline()
                if headless:
                    info.assert_not_called()
                else:
                    info.assert_called_once_with('Run cancelled: required packages were not installed.')
                self.assertEqual(self.manager._pending_prepare_queue, [])
                self.assertIsNone(self.manager._pending_prepare_success)
                self.assertIsNone(self.manager._pending_prepare_failure)
                self.assertEqual(self.manager._pending_batch_package_queue, [])
                self.assertIsNone(self.manager._pending_batch_package_success)
                self.assertIsNone(self.manager._pending_batch_package_failure)
        self.manager._startImgtransPipeline.assert_not_called()
        self.manager.package_install_thread.installPackages.assert_not_called()
        self.assertEqual(len(self.finished), 1)


class PageFailureIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @contextmanager
    def _pipeline(
        self,
        parallel: bool,
        error: Exception = None,
        *,
        quality_hook: str = None,
        user_stop: bool = False,
    ) -> Iterator[SimpleNamespace]:
        project = ProjImgTrans()
        keys = ['page-1.png', 'page-2.png', 'page-3.png']
        project.pages = {key: [TextBlock(text=['source'])] for key in keys}
        # A failed replacement must invalidate even previously finished pages.
        project._image_info = {key: {'finish_code': RunStatus.FIN_TRANSLATE} for key in keys}
        project.read_img = Mock(return_value=np.zeros((12, 12, 3), dtype=np.uint8))
        project.save_mask = Mock()
        project.save_inpainted = Mock()
        masks = [np.full((12, 12), i, dtype=np.uint8) for i in range(3)]
        detected = [[TextBlock(xyxy=[0, 0, 10, 10], text=['detected'])] for _ in keys]
        detector = SimpleNamespace(detect=Mock(side_effect=list(zip(masks, detected))))

        def ocr_result(_img: np.ndarray, blocks: list[TextBlock]) -> list[TextBlock]:
            blocks[0].text = ['plain OCR']
            return blocks

        ocr = SimpleNamespace(run_ocr=Mock(side_effect=ocr_result))
        translate = module_manager.TranslateThread()
        thread = module_manager.ImgtransThread(
            SimpleNamespace(textdetector=detector),
            SimpleNamespace(ocr=ocr),
            translate,
            SimpleNamespace(inpainter=SimpleNamespace(inpaint=Mock(return_value=project.read_img.return_value))),
        )

        def translate_page(blocks: list[TextBlock], *, page_key: str, **_kwargs) -> None:
            if page_key == keys[1] and error is not None and quality_hook is None:
                raise error
            blocks[0].translation = 'translated'

        translator = SimpleNamespace(
            low_vram_mode=False,
            lang_target='English',
            is_computational_intensive=lambda: not parallel,
            delay=lambda: 0,
            translate_textblk_lst=Mock(side_effect=translate_page),
            on_page_translation_finished=Mock(),
        )
        translate.translator = translate.module = translator
        manager = module_manager.ModuleManager(project)
        manager.imgtrans_thread = thread
        manager.translate_thread = translate
        manager.last_finished_index = -1
        manager.progress_msgbox = Mock()
        thread.update_detect_progress.connect(manager.on_update_detect_progress)
        thread.update_ocr_progress.connect(manager.on_update_ocr_progress)
        thread.update_translate_progress.connect(manager.on_update_translate_progress)
        thread.update_inpaint_progress.connect(manager.on_update_inpaint_progress)
        translate.progress_changed.connect(manager.on_update_translate_progress)
        thread.pipeline_stopped.connect(manager.on_imgtrans_thread_stopped)
        finished = QSignalSpy(manager.imgtrans_pipeline_finished)
        stopped = QSignalSpy(thread.pipeline_stopped)

        def page_error(*_args) -> None:
            if user_stop:
                thread.requestStop()

        def enhance(_img, mask, blocks, _detector, _stopped):
            if detector.detect.call_count == 2 and quality_hook == 'enhance_detection':
                blocks[0].text = ['partial enhancement']
                blocks.append(TextBlock(text=['extra detection']))
                raise error
            return mask, blocks

        def retry(_img, blocks, _ocr) -> None:
            blocks[0].text = ['refined OCR']
            if ocr.run_ocr.call_count == 2 and quality_hook == 'retry_uncertain_ocr':
                raise error

        with ExitStack() as stack:
            for name, value in (
                ('enable_detect', True), ('enable_ocr', True),
                ('enable_translate', True), ('enable_inpaint', True),
                ('keep_exist_textlines', False), ('translate_source', '日本語'),
            ):
                stack.enter_context(patch.object(pcfg.module, name, value))
            stack.enter_context(patch.object(pcfg, 'restore_ocr_empty', False))
            dialog = stack.enter_context(patch.object(module_manager, '_create_page_error_dialog', side_effect=page_error))
            # Keep the real routing and de-duplication boundary; suppress only
            # presentation of setup dialogs in these worker-only tests.
            setup_dialog = stack.enter_context(patch.object(module_manager, '_show_llm_key_required_dialog'))
            unexpected = stack.enter_context(patch.object(module_manager, 'create_error_dialog'))
            if quality_hook:
                thread.native_quality = True
                stack.enter_context(patch('ballontranslator.engine.manga_quality.enhance_detection', side_effect=enhance))
                stack.enter_context(patch('ballontranslator.engine.manga_quality.retry_uncertain_ocr', side_effect=retry))
                stack.enter_context(patch('ballontranslator.engine.manga_quality.prepare_background', return_value=project.read_img.return_value))
            try:
                thread.runImgtransPipeline(project)
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    self.app.processEvents()
                    if not thread.isRunning() and not translate.isRunning() and len(finished):
                        break
                    time.sleep(0.005)
                self.assertTrue(thread.wait(100))
                self.assertTrue(translate.wait(100))
                self.app.processEvents()
                unexpected.assert_not_called()
                yield SimpleNamespace(
                    project=project, thread=thread, translate=translate,
                    translator=translator, dialog=dialog, setup_dialog=setup_dialog,
                    finished=finished, stopped=stopped, keys=keys,
                    masks=masks, detected=detected,
                )
            finally:
                thread.requestStop()
                self.assertTrue(thread.wait(3000))
                self.assertTrue(translate.wait(3000))
                self.app.processEvents()

    def _assert_page_failure_continues(self, parallel: bool, error: Exception) -> None:
        with self._pipeline(parallel, error) as run:
            self.assertFalse(run.thread.isStopRequested())
            if parallel:
                self.assertIs(run.translate.pipeline_stop_event, run.thread.stop_event)
                self.assertEqual(run.translate.finished_counter, 3)
            else:
                self.assertEqual(run.thread.translate_counter, 3)
            self.assertGreaterEqual(len(run.finished), 1)
            self.assertEqual(len(run.stopped), 0)
            for key in (run.keys[0], run.keys[2]):
                self.assertEqual(run.project.pages[key][0].translation, 'translated')
                self.assertTrue(run.project._image_info[key]['finish_code'] & RunStatus.FIN_TRANSLATE)
            self.assertFalse(run.project._image_info[run.keys[1]]['finish_code'] & RunStatus.FIN_TRANSLATE)
            self.assertEqual(run.translator.on_page_translation_finished.call_count, 2)
            run.dialog.assert_called_once_with(
                error, 'Translation Failed.',
                f'TranslationFailed:{type(error).__name__}:profile-1', run.keys[1], 'Page',
            )

    def test_output_limit_continues_parallel_pipeline(self) -> None:
        self._assert_page_failure_continues(True, LLMOutputLimitError('profile-1', 'Profile 1', 8192, 'low'))

    def test_output_limit_continues_sequential_pipeline(self) -> None:
        self._assert_page_failure_continues(False, LLMOutputLimitError('profile-1', 'Profile 1', 8192, 'low'))

    def test_memory_compaction_failure_is_page_level(self) -> None:
        for parallel in (False, True):
            with self.subTest(parallel=parallel):
                self._assert_page_failure_continues(parallel, LLMMemoryCompactionError('profile-1', 'Profile 1', 3, 'empty response'))

    def test_run_level_errors_still_stop_both_modes(self) -> None:
        for parallel in (False, True):
            for error_type in (LLMApiKeyRequiredError, LLMAuthenticationError):
                with self.subTest(parallel=parallel, error=error_type.__name__), \
                        self._pipeline(parallel, error_type('profile-1', 'Profile 1')) as run:
                    self.assertTrue(run.thread.isStopRequested())
                    self.assertEqual(len(run.stopped), 1)
                    self.assertEqual(len(run.finished), 1)
                    self.assertFalse(run.project._image_info[run.keys[1]]['finish_code'] & RunStatus.FIN_TRANSLATE)
                    self.assertEqual(run.project.pages[run.keys[2]][0].translation, '')
                    self.assertEqual(run.translator.translate_textblk_lst.call_count, 2)
                    if error_type is LLMApiKeyRequiredError:
                        run.setup_dialog.assert_called_once()
                    else:
                        run.dialog.assert_called_once()

    def test_unknown_action_error_defaults_to_run_level(self) -> None:
        class UnknownActionError(LLMUserActionRequiredError):
            pass

        for error_type in (
            LLMUserActionRequiredError, UnknownActionError, LLMApiKeyRequiredError,
            LLMAuthenticationError, LLMModelRequiredError, LLMBaseURLRequiredError,
            CodexSignInRequiredError,
        ):
            self.assertTrue(error_type.stops_pipeline)
        for parallel in (False, True):
            with self.subTest(parallel=parallel), self._pipeline(parallel, UnknownActionError('unavailable')) as run:
                self.assertTrue(run.thread.isStopRequested())
                self.assertEqual(len(run.stopped), 1)
                self.assertEqual(run.translator.translate_textblk_lst.call_count, 2)

    def test_quality_hook_errors_preserve_plain_results_and_continue(self) -> None:
        for parallel in (False, True):
            for hook, error_type in (
                ('enhance_detection', 'MangaDetectionEnhancementFailed'),
                ('retry_uncertain_ocr', 'MangaOCRRefinementFailed'),
            ):
                error = ValueError('quality hook failed')
                with self.subTest(parallel=parallel, hook=hook), \
                        self._pipeline(parallel, error, quality_hook=hook) as run:
                    self.assertFalse(run.thread.isStopRequested())
                    self.assertEqual(run.thread.detect_counter, 3)
                    self.assertEqual(run.thread.ocr_counter, 3)
                    self.assertEqual(run.thread.inpaint_counter, 3)
                    self.assertGreaterEqual(len(run.finished), 1)
                    self.assertEqual(len(run.project.pages[run.keys[1]]), 1)
                    expected = 'plain OCR' if hook == 'retry_uncertain_ocr' else 'refined OCR'
                    self.assertEqual(run.project.pages[run.keys[1]][0].text, [expected])
                    self.assertEqual(run.project.pages[run.keys[2]][0].text, ['refined OCR'])
                    self.assertEqual(run.project.pages[run.keys[2]][0].translation, 'translated')
                    run.dialog.assert_called_once()
                    self.assertIs(run.dialog.call_args.args[0], error)
                    self.assertEqual(run.dialog.call_args.args[2:], (error_type, run.keys[1], 'Page'))
                    if hook == 'enhance_detection':
                        self.assertEqual(run.detected[1][0].text, ['plain OCR'])
                        saved = [call.args for call in run.project.save_mask.call_args_list]
                        self.assertEqual(len(saved), 1)
                        self.assertEqual(saved[0][0], run.keys[1])
                        np.testing.assert_array_equal(saved[0][1], run.masks[1])

    def test_user_stop_during_page_failure_is_honoured(self) -> None:
        for parallel in (False, True):
            with self.subTest(parallel=parallel), self._pipeline(
                parallel, LLMOutputLimitError('profile-1', 'Profile 1', 8192, 'low'), user_stop=True,
            ) as run:
                self.assertTrue(run.thread.isStopRequested())
                self.assertEqual(len(run.stopped), 1)
                self.assertEqual(run.translator.translate_textblk_lst.call_count, 2)
                self.assertFalse(run.project._image_info[run.keys[1]]['finish_code'] & RunStatus.FIN_TRANSLATE)

    def test_quality_hook_stop_is_honoured(self) -> None:
        for hook in ('enhance_detection', 'retry_uncertain_ocr'):
            for explicit_stop in (False, True):
                error = LLMRequestStopped() if explicit_stop else ValueError('quality hook failed')
                with self.subTest(hook=hook, explicit_stop=explicit_stop), self._pipeline(
                    False, error, quality_hook=hook, user_stop=not explicit_stop,
                ) as run:
                    self.assertTrue(run.thread.isStopRequested())
                    self.assertEqual(len(run.stopped), 1)
                    self.assertEqual(run.translator.translate_textblk_lst.call_count, 1)
                    self.assertEqual(run.project.pages[run.keys[2]][0].text, ['source'])


if __name__ == '__main__':
    unittest.main()
