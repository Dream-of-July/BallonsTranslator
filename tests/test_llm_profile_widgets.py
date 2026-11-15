import gc
import os
import threading
import time
import unittest
import weakref
from typing import Callable
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QCoreApplication, QEvent, Qt
from qtpy.QtGui import QContextMenuEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QMenu, QVBoxLayout, QWidget

from ballontranslator.ui.llm_profile_widgets import LLMProfilesWidget, ProfileCardWidget
from ballontranslator.ui.misc import parse_stylesheet
from ballontranslator.utils.config import pcfg
from ballontranslator.modules.llm_chat import LLMConnectionTestResult
from ballontranslator.utils.llm_profiles import (
    default_codex_profile, LLMProfile, default_profile, profile_from_template, profile_to_dict,
    provider_template_names,
)


class LLMProfileModelSelectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_model_text_is_selectable_and_vision_add_updates_text_choices(
        self,
    ) -> None:
        profile = LLMProfile(
            id='test',
            name='Test',
            model='text-model',
            model_options=['text-model', 'shared-model'],
            support_vision=True,
            vision_model='vision-model',
            vision_model_options=['vision-model'],
        )
        card = ProfileCardWidget(profile)
        self.addCleanup(card.deleteLater)

        for combo in (
            card.model_combo,
            card.vision_model_combo,
            card.image_model_combo,
        ):
            self.assertTrue(combo.isEditable())
            self.assertTrue(combo.lineEdit().isReadOnly())
            combo.lineEdit().selectAll()
            self.assertEqual(combo.lineEdit().selectedText(), combo.currentText())

        card.startVisionModelEdit()
        card.vision_model_combo.lineEdit().setText('new-vision-model')
        card.finishVisionModelEdit()

        self.assertEqual(profile.vision_model, 'new-vision-model')
        self.assertEqual(profile.model, 'text-model')
        self.assertIn('new-vision-model', profile.model_options)
        self.assertGreaterEqual(card.model_combo.findText('new-vision-model'), 0)

        card.startVisionModelEdit()
        card.vision_model_combo.lineEdit().setText('shared-model')
        card.finishVisionModelEdit()
        self.assertEqual(profile.model_options.count('shared-model'), 1)

    def image_card(self) -> ProfileCardWidget:
        card = ProfileCardWidget(LLMProfile(
            id='image-test', name='Image test', support_image=True, support_vision=True,
            image_base_url='https://api.example/v1/images/edits',
            image_model='gpt-image-2', image_model_options=['gpt-image-2', 'gpt-image-1'],
            vision_model='gpt-6-sol', vision_model_options=['gpt-6-sol', 'other-vision'],
        ))
        self.addCleanup(card.deleteLater)
        return card

    def test_image_pair_selection_does_not_pollute_saved_options(self) -> None:
        card = self.image_card()
        pair = 'gpt-6-sol → gpt-image-2'
        self.assertGreaterEqual(card.image_model_combo.findText(pair), 0)
        self.assertEqual(card.image_model_combo.findText('other-vision → gpt-image-2'), -1)
        card.image_model_combo.setCurrentText(pair)
        self.assertEqual(card.profile.image_model, pair)
        card.toggleImageSupport()
        card.toggleImageSupport()
        card.syncFromProfile()
        saved = profile_to_dict(card.profile)
        self.assertEqual(saved['image_model'], pair)
        self.assertEqual(saved['image_model_options'], ['gpt-image-2', 'gpt-image-1'])
        self.assertTrue(card.image_model_combo.lineEdit().isReadOnly())

    def test_custom_gateway_base_url_offers_and_saves_image_pairs(self) -> None:
        profile = LLMProfile(id='custom-gateway', name='Custom gateway', support_image=True, support_vision=True)
        profile.image_base_url = 'https://gateway.example/v1'
        profile.vision_model_options = ['gpt-test-reasoning']
        profile.image_model = 'gpt-image-test'
        profile.image_model_options = ['gpt-image-test']
        card = ProfileCardWidget(profile)
        self.addCleanup(card.deleteLater)
        pair = 'gpt-test-reasoning → gpt-image-test'
        self.assertGreaterEqual(card.image_model_combo.findText(pair), 0)
        card.image_model_combo.setCurrentText(pair)
        saved = profile_to_dict(profile)
        self.assertEqual(saved['image_model'], pair)
        self.assertEqual(saved['image_model_options'], ['gpt-image-test'])
        self.assertEqual(saved['image_base_url'], 'https://gateway.example/v1')

    def test_vision_add_delete_refreshes_pairs_and_preserves_unavailable_selection(self) -> None:
        card = self.image_card()
        image_combo = card.image_model_combo
        card.startVisionModelEdit()
        card.vision_model_combo.lineEdit().setText('gpt-6-luna')
        card.finishVisionModelEdit()
        pair = 'gpt-6-luna → gpt-image-2'
        self.assertGreaterEqual(image_combo.findText(pair), 0)
        self.assertGreaterEqual(image_combo.findText('gpt-6-luna → gpt-image-1'), 0)
        image_combo.setCurrentText(pair)
        card.deleteCurrentVisionModel()
        self.assertNotIn('gpt-6-luna', card.profile.vision_model_options)
        self.assertEqual(image_combo.findText('gpt-6-luna → gpt-image-1'), -1)
        self.assertEqual(image_combo.currentText(), pair)
        self.assertEqual(card.profile.image_model, pair)
        self.assertEqual(card.profile.image_model_options, ['gpt-image-2', 'gpt-image-1'])
        self.assertIs(card.image_model_combo, image_combo)

    def test_image_pair_delete_removes_underlying_image_and_all_combinations(self) -> None:
        card = self.image_card()
        card.profile.vision_model_options.append('gpt-6-luna')
        card.syncFromProfile()
        card.image_model_combo.setCurrentText('gpt-6-sol → gpt-image-2')
        card.deleteCurrentImageModel()
        choices = [card.image_model_combo.itemText(index) for index in range(card.image_model_combo.count())]
        self.assertEqual(card.profile.image_model_options, ['gpt-image-1'])
        self.assertEqual(card.profile.image_model, 'gpt-image-1')
        self.assertFalse(any('gpt-image-2' in choice for choice in choices))
        self.assertIn('gpt-6-sol → gpt-image-1', choices)
        self.assertIn('gpt-6-luna → gpt-image-1', choices)

    def test_image_add_derives_pairs_and_preserves_draft_during_sync(self) -> None:
        card = self.image_card()
        card.startImageModelEdit()
        card.image_model_combo.lineEdit().setText('gpt-image-3')
        card.profile.vision_model_options.append('gpt-6-luna')
        card.syncFromProfile()
        self.assertEqual(card.image_model_combo.lineEdit().text(), 'gpt-image-3')
        self.assertFalse(card.image_model_combo.lineEdit().isReadOnly())
        self.assertEqual(card.profile.image_model, 'gpt-image-2')
        card.finishImageModelEdit()
        self.assertEqual(card.profile.image_model, 'gpt-image-3')
        self.assertEqual(card.profile.image_model_options, ['gpt-image-2', 'gpt-image-1', 'gpt-image-3'])
        self.assertGreaterEqual(card.image_model_combo.findText('gpt-6-luna → gpt-image-3'), 0)

    def test_image_add_rejects_pairs_and_restores_previous_value(self) -> None:
        card = self.image_card()
        for value in ('gpt-6-sol → gpt-image-2', 'gpt-6-sol -> gpt-image-3', 'other → gpt-image-2'):
            with self.subTest(value=value), patch('ballontranslator.ui.llm_profile_widgets.QMessageBox.warning') as warning:
                card.startImageModelEdit()
                card.image_model_combo.lineEdit().setText(value)
                self.assertFalse(card.finishImageModelEdit())
                warning.assert_called_once()
                self.assertEqual(card.profile.image_model, 'gpt-image-2')
                self.assertEqual(card.image_model_combo.currentText(), 'gpt-image-2')
                self.assertEqual(card.profile.image_model_options, ['gpt-image-2', 'gpt-image-1'])
                self.assertTrue(card.image_model_combo.lineEdit().isReadOnly())

    def test_delete_aborts_when_pending_image_add_is_rejected(self) -> None:
        card = self.image_card()
        card.startImageModelEdit()
        card.image_model_combo.lineEdit().setText('other-model → gpt-image-2')
        with patch('ballontranslator.ui.llm_profile_widgets.QMessageBox.warning') as warning:
            card.deleteCurrentImageModel()
        warning.assert_called_once()
        self.assertEqual(card.profile.image_model, 'gpt-image-2')
        self.assertEqual(card.profile.image_model_options, ['gpt-image-2', 'gpt-image-1'])

    def test_clicking_delete_during_image_add_cannot_delete_saved_model(self) -> None:
        card = self.image_card()
        card.show()
        card.activateWindow()
        self.app.processEvents()
        card.setActionButtonsVisible(True)
        card.startImageModelEdit()
        card.image_model_combo.lineEdit().setText('other-model -> gpt-image-2')
        self.assertFalse(card.remove_image_model_btn.isEnabled())
        with patch('ballontranslator.ui.llm_profile_widgets.QMessageBox.warning') as warning:
            QTest.mouseClick(card.remove_image_model_btn, Qt.MouseButton.LeftButton)
            self.app.processEvents()
            self.assertEqual(card.profile.image_model_options, ['gpt-image-2', 'gpt-image-1'])
            QTest.mouseClick(card.model_combo.lineEdit(), Qt.MouseButton.LeftButton)
            self.app.processEvents()
        warning.assert_called_once()
        self.assertEqual(card.image_model_combo.currentText(), 'gpt-image-2')
        self.assertEqual(card.profile.image_model_options, ['gpt-image-2', 'gpt-image-1'])
        self.assertTrue(card.remove_image_model_btn.isEnabled())
        card.close()

    def test_image_endpoint_edits_refresh_choices_and_native_services_offer_no_pairs(self) -> None:
        card = self.image_card()
        editor = card.details.param_widgets['image_base_url']
        image_combo = card.image_model_combo
        for endpoint in (
            'https://generativelanguage.googleapis.com/v1beta/openai/',
            'https://openrouter.ai/api/v1',
            'https://api.example/v1/images/edits',
            'https://api.example/v1',
        ):
            with self.subTest(endpoint=endpoint):
                editor.setText(endpoint)
                editor.textEdited.emit(endpoint)
                editor.editingFinished.emit()
                self.assertEqual(card.profile.image_base_url, endpoint)
                self.assertEqual(image_combo.findText('gpt-6-sol → gpt-image-2') >= 0,
                                 endpoint.startswith('https://api.example/'))
                self.assertEqual(card.profile.image_model_options, ['gpt-image-2', 'gpt-image-1'])
                self.assertIs(card.image_model_combo, image_combo)


class LLMProfileTitleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_title_preserves_background_above_and_below_card_border(self) -> None:
        for theme in ('eva-light', 'eva-dark'):
            with self.subTest(theme=theme):
                panel = QWidget()
                self.addCleanup(panel.deleteLater)
                panel.setObjectName('ConfigContentScrollContent')
                panel.setStyleSheet(parse_stylesheet(theme))
                layout = QVBoxLayout(panel)
                card = ProfileCardWidget(LLMProfile(name='Example', title_url='https://example.com'))
                layout.addWidget(card)
                panel.show()
                self.app.processEvents()
                title_rect = card.title_label.geometry().translated(card.pos())
                rendered = panel.grab().toImage()
                # Compare empty title padding with its surroundings, on both
                # sides of the border. Neither background may become a patch.
                title_x = title_rect.right() - 1
                beside_x = title_rect.right() + 5
                above = title_rect.top()
                below = title_rect.bottom()
                self.assertNotEqual(
                    rendered.pixelColor(beside_x, above),
                    rendered.pixelColor(beside_x, below),
                )
                for y in (above, below):
                    self.assertEqual(
                        rendered.pixelColor(title_x, y),
                        rendered.pixelColor(beside_x, y),
                    )
                panel.close()

    def test_link_click_and_single_field_edit_roundtrip(self) -> None:
        profile = LLMProfile(name='Example', title_url='https://example.com')
        url = profile.title_url
        card = ProfileCardWidget(profile)
        self.addCleanup(card.deleteLater)
        card.show()
        self.app.processEvents()

        with patch('ballontranslator.ui.llm_profile_widgets.QDesktopServices.openUrl') as open_url:
            QTest.mouseClick(card.title_label, Qt.MouseButton.LeftButton)
            self.assertEqual(open_url.call_args[0][0].toString(), url)
            self.assertFalse(card.name_edit.isVisible())

        with patch.object(QMenu, 'exec', lambda menu, *args: menu.actions()[0]):
            position = card.title_label.rect().center()
            self.app.sendEvent(card.title_label, QContextMenuEvent(
                QContextMenuEvent.Reason.Mouse, position,
                card.title_label.mapToGlobal(position),
            ))
        self.assertTrue(card.name_edit.isVisible())
        self.assertEqual(card.name_edit.text(), f'[Example]({url})')
        card.name_edit.setText(f'[A & <B>]({url})')
        QTest.keyClick(card.name_edit, Qt.Key.Key_Return)
        self.assertEqual(profile.name, 'A & <B>')
        self.assertEqual(profile.title_url, url)
        self.assertIn('A &amp; &lt;B&gt;', card.title_label.text())
        card.startNameEdit()
        self.assertEqual(card.name_edit.text(), f'[A & <B>]({url})')
        card.name_edit.setText('Plain <title>')
        QTest.mouseClick(card.model_combo.lineEdit(), Qt.MouseButton.LeftButton)
        self.assertEqual(profile.name, 'Plain <title>')
        self.assertEqual(profile.title_url, '')
        self.assertEqual(card.title_label.textFormat(), Qt.TextFormat.PlainText)
        with patch('ballontranslator.ui.llm_profile_widgets.QDesktopServices.openUrl') as open_url:
            QTest.mouseClick(card.title_label, Qt.MouseButton.LeftButton)
            open_url.assert_not_called()
        QTest.mouseDClick(card.title_label, Qt.MouseButton.LeftButton)
        self.assertTrue(card.name_edit.isVisible())
        card.close()

    def test_malformed_or_non_web_links_remain_plain_text(self) -> None:
        card = ProfileCardWidget(LLMProfile(name='Original'))
        self.addCleanup(card.deleteLater)
        for text in ('[Broken](https://)', '[Local](file:///tmp/file)', '[Broken](https://[bad)', '<b>Plain</b>'):
            with self.subTest(text=text):
                card.startNameEdit()
                card.name_edit.setText(text)
                card.on_name_edit_finished()
                self.assertEqual(card.profile.name, text)
                self.assertEqual(card.profile.title_url, '')
                self.assertEqual(card.title_label.textFormat(), Qt.TextFormat.PlainText)


class APIProfilesPanelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_repeated_profile_copies_have_unique_ids_and_independent_settings(self) -> None:
        source = default_profile('Gemini')
        source.api_key = 'saved-test-key'
        original = profile_to_dict(source)
        with patch.object(pcfg.module, 'llm_profiles', [source]):
            panel = LLMProfilesWidget()
            self.addCleanup(panel.deleteLater)
            panel.copyProfile(source.id)
            panel.copyProfile(source.id)
            self.app.processEvents()
            first, second = pcfg.module.llm_profiles[1:]
            self.assertEqual(len({source.id, first.id, second.id}), 3)
            for copied in (first, second):
                self.assertIn(copied.id, panel.rows)
                self.assertEqual(profile_to_dict(copied), {
                    **original, 'id': copied.id, 'name': source.name + ' Copy', 'built_in': False,
                })
            first.model_options.append('copy-only-model')
            self.assertNotIn('copy-only-model', source.model_options)
            self.assertNotIn('copy-only-model', second.model_options)
            self.assertEqual(profile_to_dict(source), original)

    def test_codex_is_excluded_from_profile_editing_copy_and_delete(self) -> None:
        codex = default_codex_profile()
        api = default_profile('OpenAI')
        with patch.object(pcfg.module, 'llm_profiles', [codex, api]):
            panel = LLMProfilesWidget()
            self.addCleanup(panel.deleteLater)
            self.assertNotIn(codex.id, panel.rows)
            self.assertIn('base_url', panel.rows[api.id].details.param_widgets)
            panel.rows[api.id].toggleVisionSupport()
            self.assertFalse(api.support_vision)
            previous_clipboard = QApplication.clipboard().text()
            panel.copyProfileAsJson(codex.id)
            self.assertEqual(QApplication.clipboard().text(), previous_clipboard)
            panel.copyProfile(codex.id)
            panel.deleteProfile(codex.id)
            self.assertEqual(pcfg.module.llm_profiles, [codex, api])

    def test_new_menu_creates_prefilled_profiles_from_provider_templates(self) -> None:
        existing = default_profile('DeepSeek')
        with patch.object(pcfg.module, 'llm_profiles', [existing]):
            panel = LLMProfilesWidget()
            self.addCleanup(panel.deleteLater)
            panel.show()
            actions = {action.text(): action for action in panel.new_btn.menu().actions()}
            for name in provider_template_names():
                self.assertIn(name, actions)

            actions['Moonshot'].trigger()
            self.app.processEvents()
            moonshot = pcfg.module.llm_profiles[-1]
            template = profile_from_template('Moonshot')
            self.assertEqual(
                (moonshot.name, moonshot.base_url, moonshot.model, moonshot.built_in, moonshot.api_key),
                ('Moonshot', template.base_url, template.model, False, ''),
            )
            self.assertTrue(moonshot.id.startswith('custom-'))
            row = panel.rows[moonshot.id]
            self.assertTrue(row.details.isVisible())
            self.assertTrue(row.api_key_widget.editor.hasFocus())

            actions['DeepSeek'].trigger()
            duplicate = pcfg.module.llm_profiles[-1]
            self.assertEqual(duplicate.name, 'DeepSeek 2')
            self.assertNotEqual(duplicate.id, existing.id)
            self.assertEqual(profile_to_dict(existing), profile_to_dict(default_profile('DeepSeek')))

            actions['Zhipu'].trigger()
            self.assertEqual(pcfg.module.llm_profiles[-1].model, '')

            custom = next(action for text, action in actions.items() if text.startswith('Custom'))
            custom.trigger()
            blank = pcfg.module.llm_profiles[-1]
            self.assertEqual((blank.base_url, blank.model, blank.built_in), ('', '', False))
            self.assertEqual(len(panel.rows), 5)
            panel.hide()


class ProfileConnectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.release = threading.Event()
        self.started = 0
        self.destroyed = 0
        self.requests: list[tuple[LLMProfile, str]] = []
        self.result = LLMConnectionTestResult(True, 'Connection successful.', 'ok', ['m1', 'm2', 'm3'])
        check_patch = patch(
            'ballontranslator.modules.llm_chat.check_profile_connection', self.check_connection,
        )
        check_patch.start()
        self.addCleanup(check_patch.stop)
        # A failed assertion must not leave the worker blocked while the patch is removed.
        self.addCleanup(self.finish_requests)

    def check_connection(self, profile: LLMProfile, *, proxy: str = '', timeout: float = 15.0):
        self.requests.append((profile, proxy))
        self.release.wait(10)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    def finish_requests(self) -> None:
        self.release.set()
        self.wait_until(self.all_workers_destroyed)

    def all_workers_destroyed(self) -> bool:
        return self.destroyed == self.started

    def on_worker_destroyed(self, *_) -> None:
        self.destroyed += 1

    def wait_until(self, predicate: Callable[[], bool]) -> None:
        deadline = time.monotonic() + 10
        while not predicate() and time.monotonic() < deadline:
            self.app.processEvents()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            time.sleep(0.005)
        self.assertTrue(predicate())

    def start(self, card: ProfileCardWidget) -> None:
        card.test_connection_btn.click()
        self.started += 1
        card._connection_worker.destroyed.connect(self.on_worker_destroyed)

    def card(self, profile: LLMProfile) -> ProfileCardWidget:
        card = ProfileCardWidget(profile)
        self.addCleanup(card.deleteLater)
        return card

    def test_success_uses_unsaved_fields_and_offers_models_without_saving_them(self) -> None:
        profile = LLMProfile(
            id='t', name='T', base_url='https://saved.example/v1',
            model='m1', model_options=['m1', 'user-added'],
        )
        card = self.card(profile)
        card.api_key_widget.setText('typed-key')
        card.details.param_widgets['base_url'].setText('https://typed.example/v1')
        proxy_params = {'LLMTranslator': {'proxy': {'value': 'http://127.0.0.1:7890'}}}
        with patch.object(pcfg.module, 'translator_params', proxy_params):
            self.start(card)
        self.assertFalse(card.test_connection_btn.isEnabled())
        self.assertTrue(card.connection_status_label.text())
        card.test_connection_btn.click()
        self.wait_until(lambda: bool(self.requests))
        self.release.set()
        self.wait_until(card.test_connection_btn.isEnabled)

        self.assertEqual(len(self.requests), 1)
        tested, proxy = self.requests[0]
        self.assertEqual(
            (tested.api_key, tested.base_url, tested.model, proxy),
            ('typed-key', 'https://typed.example/v1', 'm1', 'http://127.0.0.1:7890'),
        )
        self.assertEqual((profile.api_key, profile.base_url), ('', 'https://saved.example/v1'))
        self.assertIn('3', card.connection_status_label.text())
        self.assertNotIn('typed-key', card.connection_status_label.text())
        combo = card.model_combo
        self.assertEqual(
            [combo.itemText(index) for index in range(combo.count())],
            ['m1', 'user-added', 'm2', 'm3'],
        )
        self.assertEqual(combo.currentText(), 'm1')
        self.assertEqual(profile.model_options, ['m1', 'user-added'])

        card.syncFromProfile()
        self.assertEqual(combo.count(), 4)
        combo.setCurrentText('m3')
        self.assertEqual(profile.model, 'm3')
        self.assertEqual(profile.model_options, ['m1', 'user-added', 'm3'])

    def test_failure_and_unexpected_errors_show_a_safe_inline_message(self) -> None:
        card = self.card(LLMProfile(id='t', name='T', api_key='secret-key', model='m1', model_options=['m1']))
        self.release.set()
        self.result = LLMConnectionTestResult(
            False, 'HTTP 401: Authentication failed. Check the API key.', 'auth', [],
        )
        self.start(card)
        self.wait_until(card.test_connection_btn.isEnabled)
        self.assertEqual(card.connection_status_label.text(), self.result.message)
        self.assertEqual(card.model_combo.count(), 1)

        self.result = RuntimeError('proxy rejected secret-key')
        self.start(card)
        self.wait_until(card.test_connection_btn.isEnabled)
        self.assertTrue(card.connection_status_label.text())
        self.assertNotIn('secret-key', card.connection_status_label.text())

        card.api_key_widget.editor.textEdited.emit('other')
        self.assertEqual(card.connection_status_label.text(), '')

    def test_fetched_models_do_not_select_a_model_for_an_empty_profile(self) -> None:
        profile = profile_from_template('Zhipu')
        card = self.card(profile)
        card.api_key_widget.setText('typed-key')
        self.release.set()
        self.result = LLMConnectionTestResult(True, 'Connection successful.', 'ok', ['glm-a', 'glm-b', 'glm-a'])
        self.start(card)
        self.wait_until(card.test_connection_btn.isEnabled)
        self.assertEqual(card.model_combo.count(), 2)
        self.assertEqual((card.model_combo.currentText(), profile.model, profile.model_options), ('', '', []))

    def test_deleting_the_card_or_panel_during_a_request_is_safe(self) -> None:
        first, second = default_profile('OpenAI'), default_profile('DeepSeek')
        first.api_key = second.api_key = 'key'
        with patch.object(pcfg.module, 'llm_profiles', [first, second]), \
                patch.object(pcfg.module, 'translator_llm_id', first.id), \
                patch.object(pcfg.module, 'ocr_llm_id', first.id), \
                patch.object(pcfg.module, 'inpaint_llm_id', first.id):
            panel = LLMProfilesWidget()
            self.start(panel.rows[second.id])
            self.start(panel.rows[first.id])
            self.wait_until(lambda: len(self.requests) == 2)
            card_ref = weakref.ref(panel.rows[second.id])

            panel.deleteProfile(second.id)
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            panel.deleteLater()
            panel_ref = weakref.ref(panel)
            del panel
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            gc.collect()
            self.assertIsNone(card_ref())
            self.assertIsNone(panel_ref())

            self.assertEqual(self.destroyed, 0)
            self.release.set()
            self.wait_until(self.all_workers_destroyed)
            self.assertEqual(pcfg.module.llm_profiles, [first])


if __name__ == '__main__':
    unittest.main()
