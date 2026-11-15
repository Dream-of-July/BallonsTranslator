import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtWidgets import QApplication, QLineEdit

from ballontranslator.modules import TRANSLATORS
from ballontranslator.modules.base import merge_config_module_params
from ballontranslator.ui.module_parse_widgets import ParamWidget
from ballontranslator.utils.config import ModuleConfig

SECRET_PARAMS = (
    ('DeepL', 'api_key'),
    ('Yandex', 'api_key'),
    ('Baidu', 'token'),
    ('Caiyun', 'token'),
)


class TranslatorSecretParamsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_plain_saved_keys_load_into_masked_fields_and_save_back_plain(self) -> None:
        for module_name, key in SECRET_PARAMS:
            with self.subTest(translator=module_name):
                # The pre-change config format: a plain string per key.
                saved = {module_name: {key: 'saved-key', 'delay': 0.0}}
                merge_config_module_params(saved, [module_name], TRANSLATORS.module_dict.get)
                params = saved[module_name]
                self.assertEqual(params[key]['value'], 'saved-key')

                widget = ParamWidget(params, module_type='translator', module_key=module_name)
                self.addCleanup(widget.deleteLater)
                editor = widget.param_widgets[key].editor
                self.assertEqual(editor.echoMode(), QLineEdit.EchoMode.Password)
                self.assertEqual(editor.text(), 'saved-key')
                self.assertNotIn('obfuscated', editor.toolTip())
                edits = []
                widget.paramwidget_edited.connect(lambda name, content: edits.append((name, content['content'])))
                editor.setText('new-key')
                self.assertEqual(edits[-1], (key, 'new-key'))

                config = ModuleConfig()
                config.translator_params = saved
                self.assertEqual(config.get_params('translator', for_saving=True)[module_name][key], 'saved-key')

    def test_translators_send_the_plain_key(self) -> None:
        from ballontranslator.modules.translators.trans_baidu import BaiduTranslator
        from ballontranslator.modules.translators.trans_caiyun import CaiyunTranslator

        for translator_cls, saved, reply in (
            (CaiyunTranslator, {'token': 'saved-key', 'delay': 0.0}, '{"target": ["x"]}'),
            (BaiduTranslator, {'token': 'saved-key', 'appId': 'app', 'delay': 0.0},
             '{"trans_result": [{"dst": "x"}]}'),
        ):
            with self.subTest(translator=translator_cls.__name__):
                name = next(key for key, value in TRANSLATORS.module_dict.items() if value is translator_cls)
                config = {name: saved}
                merge_config_module_params(config, [name], lambda _key: translator_cls)
                translator = translator_cls('日本語', '简体中文', **config[name])
                self.assertEqual(translator.get_param_value('token'), 'saved-key')
                request = Mock(return_value=Mock(text=reply))
                with patch('requests.request', request):
                    self.assertEqual(translator._translate(['a']), ['x'])
                sent = request.call_args.kwargs
                if translator_cls is CaiyunTranslator:
                    self.assertEqual(sent['headers']['x-authorization'], 'token saved-key')
                else:
                    self.assertEqual(sent['data']['appid'], 'app')
                with patch('requests.request', request):
                    translator.updateParam('token', '')
                    with self.assertRaises(Exception) as raised:
                        translator._translate(['a'])
                self.assertEqual(type(raised.exception).__name__, 'MissingTranslatorParams')


if __name__ == '__main__':
    unittest.main()
