import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QCoreApplication, QEvent, QObject, Qt
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import QApplication, QPlainTextEdit, QWidget

from ballontranslator.ui import appearance
from ballontranslator.ui.custom_widget import ScrollBar
from ballontranslator.ui.mainwindow import MainWindow
from ballontranslator.ui.mainwindowbars import TitleBar
from ballontranslator.ui.menu_style import install_app_style_filters
from ballontranslator.ui.misc import parse_stylesheet
from ballontranslator.utils import shared
from ballontranslator.utils.config import (
    APPEARANCE_DARK,
    APPEARANCE_LIGHT,
    APPEARANCE_SYSTEM,
    ProgramConfig,
    pcfg,
)


def load_config(payload: dict) -> ProgramConfig:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'config.json')
        with open(path, 'w', encoding='utf8') as f:
            json.dump(payload, f)
        return ProgramConfig.load(path)


class AppearanceConfigTests(unittest.TestCase):

    def test_old_config_with_dark_mode_stays_dark(self) -> None:
        config = load_config({'darkmode': True})
        self.assertEqual(config.appearance, APPEARANCE_DARK)
        self.assertTrue(config.darkmode)

    def test_old_config_without_a_choice_follows_the_system(self) -> None:
        for payload in ({}, {'darkmode': False}):
            with self.subTest(payload=payload):
                self.assertEqual(load_config(payload).appearance, APPEARANCE_SYSTEM)

    def test_saved_choice_wins_over_the_legacy_flag(self) -> None:
        config = load_config({'darkmode': True, 'appearance': APPEARANCE_LIGHT})
        self.assertEqual(config.appearance, APPEARANCE_LIGHT)

    def test_invalid_choice_is_discarded_with_a_warning(self) -> None:
        for value in ('sepia', 3, None):
            with self.subTest(value=value), self.assertLogs('BallonTranslator', 'WARNING'):
                config = load_config({'darkmode': True, 'appearance': value})
                self.assertEqual(config.appearance, APPEARANCE_SYSTEM)
                self.assertTrue(config.darkmode)


class AppearanceStateTestCase(unittest.TestCase):
    """Restore the shared config fields these tests rewrite."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.addCleanup(setattr, pcfg, 'darkmode', pcfg.darkmode)
        self.addCleanup(setattr, pcfg, 'appearance', pcfg.appearance)

    def system_scheme(self, dark: bool):
        """Patch the style hints so no real OS appearance is involved."""
        scheme = Qt.ColorScheme.Dark if dark else Qt.ColorScheme.Light
        hints = SimpleNamespace(colorScheme=lambda: scheme, setColorScheme=Mock())
        return patch.object(QGuiApplication, 'styleHints', return_value=hints)


@unittest.skipUnless(hasattr(Qt, 'ColorScheme'), 'needs Qt >= 6.5')
class AppearanceResolutionTests(AppearanceStateTestCase):

    def test_macos_resolves_the_three_states(self) -> None:
        cases = (
            (APPEARANCE_SYSTEM, True, True),
            (APPEARANCE_SYSTEM, False, False),
            (APPEARANCE_LIGHT, True, False),
            (APPEARANCE_DARK, False, True),
        )
        for choice, system_dark, expected in cases:
            with self.subTest(choice=choice, system_dark=system_dark), \
                    patch.object(shared, 'ON_MACOS', True), \
                    self.system_scheme(system_dark):
                pcfg.appearance = choice
                appearance.sync_darkmode()
                self.assertEqual(pcfg.darkmode, expected)
                self.assertEqual(
                    appearance.current_theme(),
                    'mac-dark' if expected else 'mac-light',
                )

    def test_other_platforms_keep_the_dark_mode_flag(self) -> None:
        with patch.object(shared, 'ON_MACOS', False), self.system_scheme(True) as hints:
            pcfg.appearance = APPEARANCE_SYSTEM
            for darkmode in (False, True):
                pcfg.darkmode = darkmode
                appearance.apply_appearance_override()
                self.assertFalse(appearance.sync_darkmode())
                self.assertEqual(pcfg.darkmode, darkmode)
                self.assertEqual(
                    appearance.current_theme(),
                    'eva-dark' if darkmode else 'eva-light',
                )
            hints.return_value.setColorScheme.assert_not_called()
            self.assertEqual(appearance.accent_rgb(), appearance.DEFAULT_ACCENT_RGB)

    def test_forced_choice_overrides_the_native_scheme(self) -> None:
        with patch.object(shared, 'ON_MACOS', True), self.system_scheme(False) as hints:
            for choice, scheme in (
                (APPEARANCE_DARK, Qt.ColorScheme.Dark),
                (APPEARANCE_LIGHT, Qt.ColorScheme.Light),
                (APPEARANCE_SYSTEM, Qt.ColorScheme.Unknown),
            ):
                pcfg.appearance = choice
                appearance.apply_appearance_override()
                hints.return_value.setColorScheme.assert_called_with(scheme)

    def test_missing_color_scheme_support_falls_back_to_light(self) -> None:
        with patch.object(shared, 'ON_MACOS', True), \
                patch.object(QGuiApplication, 'styleHints', return_value=object()):
            pcfg.appearance = APPEARANCE_SYSTEM
            appearance.apply_appearance_override()
            self.assertFalse(appearance.resolve_darkmode())

    def test_color_scheme_change_switches_the_applied_theme(self) -> None:
        host = QWidget()
        self.addCleanup(host.deleteLater)
        host.resetStyleSheet = lambda: MainWindow.resetStyleSheet(host)
        pcfg.appearance = APPEARANCE_SYSTEM
        with patch.object(shared, 'ON_MACOS', True):
            with self.system_scheme(False):
                appearance.sync_darkmode()
                MainWindow.resetStyleSheet(host)
            light = host.styleSheet()
            self.assertEqual(light, parse_stylesheet('mac-light'))

            with self.system_scheme(True):
                MainWindow.on_color_scheme_changed(host)
            self.assertTrue(pcfg.darkmode)
            self.assertEqual(host.styleSheet(), parse_stylesheet('mac-dark'))
            self.assertNotEqual(host.styleSheet(), light)

            # A forced choice ignores later system changes.
            pcfg.appearance = APPEARANCE_DARK
            with self.system_scheme(False):
                MainWindow.on_color_scheme_changed(host)
            self.assertEqual(host.styleSheet(), parse_stylesheet('mac-dark'))

    def test_live_switch_rewrites_construction_time_icons(self) -> None:
        from ballontranslator.ui.misc import themed_icon_path

        def icon_source() -> str:
            with open(themed_icon_path('menu_check.svg'), encoding='utf8') as f:
                return f.read()

        with patch.object(shared, 'ON_MACOS', True):
            pcfg.darkmode = False
            path = themed_icon_path('menu_check.svg')
            light = icon_source()
            pcfg.darkmode = True
            self.assertEqual(themed_icon_path('menu_check.svg'), path)
            self.assertNotEqual(icon_source(), light)

    def test_view_menu_offers_exclusive_appearance_choices_on_macos(self) -> None:
        pcfg.appearance = APPEARANCE_SYSTEM
        with patch.object(shared, 'ON_MACOS', True):
            host = QWidget()
            self.addCleanup(host.deleteLater)
            title_bar = TitleBar(host)
        chosen = []
        title_bar.appearance_changed.connect(chosen.append)
        actions = title_bar.appearanceMenu.actions()
        self.assertEqual(
            [action.data() for action in actions],
            [APPEARANCE_SYSTEM, APPEARANCE_LIGHT, APPEARANCE_DARK],
        )
        self.assertEqual([a.isChecked() for a in actions], [True, False, False])
        self.assertNotIn(title_bar.darkModeAction, title_bar.viewMenu.actions())
        actions[2].trigger()
        self.assertEqual(chosen, [APPEARANCE_DARK])
        self.assertEqual([a.isChecked() for a in actions], [False, False, True])

    def test_other_platforms_keep_the_dark_mode_action(self) -> None:
        with patch.object(shared, 'ON_MACOS', False):
            host = QWidget()
            self.addCleanup(host.deleteLater)
            title_bar = TitleBar(host)
        self.assertIn(title_bar.darkModeAction, title_bar.viewMenu.actions())
        self.assertFalse(hasattr(title_bar, 'appearanceMenu'))


class NativeControlTests(AppearanceStateTestCase):

    def test_macos_keeps_native_scrollbars(self) -> None:
        editor = QPlainTextEdit()
        self.addCleanup(editor.deleteLater)
        with patch.object(shared, 'ON_MACOS', True):
            self.assertIsNone(ScrollBar(Qt.Orientation.Vertical, editor))
        self.assertNotEqual(
            editor.verticalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )

    def test_other_platforms_install_the_custom_scrollbar(self) -> None:
        editor = QPlainTextEdit()
        self.addCleanup(editor.deleteLater)
        with patch.object(shared, 'ON_MACOS', False):
            bar = ScrollBar(Qt.Orientation.Vertical, editor)
        self.assertIsInstance(bar, ScrollBar)
        self.assertIs(bar.parent(), editor)
        self.assertEqual(
            editor.verticalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )

    def test_menu_style_filter_is_skipped_on_macos_only(self) -> None:
        for on_macos in (True, False):
            with self.subTest(on_macos=on_macos), patch.object(shared, 'ON_MACOS', on_macos):
                host = QObject()
                install_app_style_filters(host)
                self.assertEqual(
                    getattr(host, '_menu_style_filter', None) is None, on_macos
                )
                self.assertIsNotNone(host._dropdown_style_filter)
                host.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


class MacThemeConsumerTests(AppearanceStateTestCase):

    def test_module_menu_colors_come_from_the_current_theme(self) -> None:
        from ballontranslator.ui import module_tool_button

        with open(shared.THEME_PATH, encoding='utf8') as f:
            themes = json.load(f)
        pcfg.darkmode = True
        for on_macos, theme in ((True, 'mac-dark'), (False, 'eva-dark')):
            with self.subTest(theme=theme), patch.object(shared, 'ON_MACOS', on_macos):
                self.assertEqual(
                    module_tool_button._theme_menu_background_hex(),
                    themes[theme]['@emptyContentBackgroundColor'],
                )

    def test_text_checker_label_leaves_its_checked_look_to_the_mac_theme(self) -> None:
        from ballontranslator.ui.custom_widget import TextCheckerLabel

        with patch.object(shared, 'ON_MACOS', True):
            label = TextCheckerLabel('Source', True)
            self.addCleanup(label.deleteLater)
            self.assertEqual(label.styleSheet(), '')
            self.assertTrue(label.property('checked'))
            label.setCheckState(False)
            self.assertFalse(label.property('checked'))
        with patch.object(shared, 'ON_MACOS', False):
            label = TextCheckerLabel('Source', True)
            self.addCleanup(label.deleteLater)
            self.assertIn('rgb(30, 147, 229)', label.styleSheet())
            label.setCheckState(False)
            self.assertEqual(label.styleSheet(), '')

    def test_new_text_formats_default_to_a_platform_font(self) -> None:
        import sys
        from ballontranslator.utils.fontformat import FontFormat

        expected = 'PingFang SC' if sys.platform == 'darwin' else 'Microsoft YaHei UI'
        self.assertEqual(FontFormat.__dataclass_fields__['font_family'].default, expected)
        # A saved family is never replaced by the platform default.
        self.assertEqual(FontFormat(font_family='Custom Family').font_family, 'Custom Family')


class ThemeFileTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        with open(shared.THEME_PATH, encoding='utf8') as f:
            cls.themes = json.load(f)

    def test_every_theme_defines_the_same_tokens(self) -> None:
        self.assertEqual(
            set(self.themes), {'eva-light', 'eva-dark', 'mac-light', 'mac-dark'}
        )
        expected = set(self.themes['eva-light'])
        for name, tokens in self.themes.items():
            with self.subTest(theme=name):
                self.assertEqual(set(tokens), expected)

    def test_no_token_name_is_a_prefix_of_another(self) -> None:
        # Tokens are substituted with plain str.replace in arbitrary order.
        names = sorted(set(self.themes['eva-light']) | {'@macAccentRgb'})
        for name in names:
            for other in names:
                if name != other:
                    self.assertFalse(other.startswith(name), (name, other))

    def test_platform_blocks_only_reach_their_own_themes(self) -> None:
        for theme in ('eva-light', 'eva-dark'):
            with self.subTest(theme=theme):
                stylesheet = parse_stylesheet(theme)
                self.assertIn('QScrollBar:vertical {', stylesheet)
                self.assertNotIn('@macos-begin', stylesheet)
                self.assertNotIn('macAccentRgb', stylesheet)
                self.assertNotIn('checkbox_checked_mac.svg', stylesheet)
        for theme in ('mac-light', 'mac-dark'):
            with self.subTest(theme=theme):
                stylesheet = parse_stylesheet(theme)
                self.assertNotIn('QScrollBar:vertical {', stylesheet)
                self.assertNotIn('macAccentRgb', stylesheet)
                self.assertIn('checkbox_checked_mac.svg', stylesheet)
                for token in self.themes[theme]:
                    self.assertNotIn(token, stylesheet)

    def test_system_accent_replaces_the_established_blue_in_mac_themes(self) -> None:
        with patch('ballontranslator.ui.misc.accent_rgb', return_value=(10, 20, 30)):
            mac = parse_stylesheet('mac-light')
            eva = parse_stylesheet('eva-light')
        self.assertIn('rgb(10, 20, 30)', mac)
        self.assertNotIn('30, 147, 229', mac)
        self.assertIn('rgb(30, 147, 229)', eva)
        self.assertNotIn('10, 20, 30', eva)


if __name__ == '__main__':
    unittest.main()
