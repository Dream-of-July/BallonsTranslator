import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QCoreApplication, QEvent, Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QMessageBox, QWidget

from ballontranslator.ui import mainwindow as mainwindow_module
from ballontranslator.ui.configpanel import ConfigPanel
from ballontranslator.ui.mainwindow import MainWindow
from ballontranslator.ui.mainwindowbars import LeftBar, TitleBar
from ballontranslator.utils import shared
from ballontranslator.utils.config import ProgramConfig, RunStatus, pcfg
from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.proj_imgtrans import ProjImgTrans

if shared.FLAG_QT6:
    from qtpy.QtGui import QAction
else:
    from qtpy.QtWidgets import QAction


class ShellHost(QWidget):
    """Main-window stand-in that owns the real bars and menu-bar builder."""

    setupMacMenuBar = MainWindow.setupMacMenuBar
    _active_window = MainWindow._active_window
    on_window_minimize = MainWindow.on_window_minimize
    on_window_zoom = MainWindow.on_window_zoom
    on_window_close = MainWindow.on_window_close
    on_about = MainWindow.on_about

    def __init__(self) -> None:
        super().__init__()
        self.settings_requests = 0
        self.close_requests = 0
        self.leftBar = LeftBar(self)
        self.titleBar = TitleBar(self)

    def setupConfigUI(self) -> None:
        self.settings_requests += 1

    def on_closebtn_clicked(self) -> None:
        self.close_requests += 1


class MacShellTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def make_host(self) -> ShellHost:
        host = ShellHost()
        self.addCleanup(
            QCoreApplication.sendPostedEvents,
            None,
            QEvent.Type.DeferredDelete,
        )
        self.addCleanup(host.deleteLater)
        return host

    def test_macos_menu_bar_reuses_bar_menus_and_app_menu_roles(self) -> None:
        with patch.object(shared, 'ON_MACOS', True):
            host = self.make_host()
            host.setupMacMenuBar()
        menus = {
            action.text(): action.menu()
            for action in host.macMenuBar.actions()
        }
        self.assertEqual(
            list(menus),
            ['File', 'Edit', 'View', 'Go', 'Tools', 'Window', 'Help'],
        )
        self.assertIs(menus['File'], host.leftBar.fileMenu)
        self.assertIn(host.leftBar.recentMenu.menuAction(), menus['File'].actions())
        self.assertIs(menus['View'], host.titleBar.viewMenu)
        self.assertIn(host.titleBar.patreonSponsorAction, menus['Help'].actions())

        roles = {
            action.menuRole(): action
            for menu in menus.values()
            for action in menu.actions()
        }
        menu_role = QAction.MenuRole
        for role in (menu_role.AboutRole, menu_role.PreferencesRole, menu_role.QuitRole):
            self.assertIn(role, roles)
        self.assertEqual(
            roles[menu_role.PreferencesRole].shortcut(),
            QKeySequence('Ctrl+,'),
        )
        roles[menu_role.PreferencesRole].trigger()
        self.assertEqual(host.settings_requests, 1)
        roles[menu_role.QuitRole].trigger()
        self.assertEqual(host.close_requests, 1)

        window_shortcuts = {
            action.shortcut().toString() for action in menus['Window'].actions()
        }
        self.assertTrue({'Ctrl+M', 'Ctrl+W'} <= window_shortcuts)

        tool_texts = [action.text() for action in menus['Tools'].actions()]
        self.assertNotIn('Photoshop Bridge', tool_texts)
        run_action = host.leftBar.runAction
        self.assertIs(menus['Tools'].actions()[0], run_action)
        self.assertEqual(
            [shortcut.toString() for shortcut in run_action.shortcuts()],
            ['Ctrl+R', 'F5'],
        )
        run_requests = []
        host.leftBar.run_imgtrans_clicked.connect(
            lambda: run_requests.append(True)
        )
        run_action.trigger()
        self.assertEqual(run_requests, [True])
        # Hidden bar buttons no longer carry the shortcuts; the window does.
        self.assertIn(run_action, host.actions())

    def test_macos_window_title_follows_title_content(self) -> None:
        with patch.object(shared, 'ON_MACOS', True):
            host = self.make_host()
            host.titleBar.setTitleContent('proj', 'page.png', 'unsaved')
        self.assertEqual(host.windowTitle(), 'proj - page.png - unsaved')

    def test_other_platforms_keep_custom_title_bar_and_run_shortcut(self) -> None:
        with patch.object(shared, 'ON_MACOS', False):
            host = self.make_host()
            host.titleBar.setTitleContent('proj', 'page.png', 'saved')
        title_bar = host.titleBar
        self.assertEqual(title_bar.titleLabel.text(), 'proj - page.png - saved')
        self.assertNotEqual(host.windowTitle(), title_bar.titleLabel.text())
        close_requests = []
        title_bar.closebtn_clicked.connect(lambda: close_requests.append(True))
        title_bar.closeBtn.click()
        self.assertEqual(close_requests, [True])
        self.assertIn(
            'Photoshop Bridge',
            [action.text() for action in title_bar.toolsToolBtn.menu().actions()],
        )
        self.assertIsNone(host.leftBar.runAction)
        self.assertEqual(
            host.leftBar.runImgtransBtn.shortcut(), QKeySequence('F5')
        )
        self.assertEqual(host.leftBar.runImgtransBtn.toolTip(), 'Run (F5)')
        native_search = QKeySequence('Ctrl+G').toString(
            QKeySequence.SequenceFormat.NativeText
        )
        self.assertEqual(
            host.leftBar.globalSearchChecker.toolTip(),
            f'Global Search ({native_search})',
        )

    def test_capitalize_shortcut_does_not_take_quit_on_macos(self) -> None:
        capitalize = QKeySequence(mainwindow_module.CAPITALIZE_SHORTCUT)
        if shared.ON_MACOS:
            self.assertNotEqual(capitalize, QKeySequence('Ctrl+Q'))
        else:
            self.assertEqual(capitalize, QKeySequence('Ctrl+Q'))

    def test_run_without_project_prompts_instead_of_opening_dialog(self) -> None:
        owner = SimpleNamespace(
            imgtrans_proj=ProjImgTrans(),
            _prompt_open_folder_for_run=Mock(),
        )
        with patch.object(mainwindow_module, 'RunPipelineDialog') as dialog_cls:
            MainWindow.run_imgtrans(owner)
        dialog_cls.assert_not_called()
        owner._prompt_open_folder_for_run.assert_called_once_with()

    def test_open_folder_prompt_reuses_the_open_folder_action(self) -> None:
        host = QWidget()
        self.addCleanup(host.deleteLater)
        open_action = QAction('Open Folder ...', host)
        open_requests = []
        open_action.triggered.connect(lambda: open_requests.append(True))
        host.leftBar = SimpleNamespace(openFolderAction=open_action)

        def accept(box: QMessageBox) -> int:
            box.buttons()[0].click()
            return 0

        def cancel(box: QMessageBox) -> int:
            box.reject()
            return 0

        with patch.object(QMessageBox, 'exec', cancel):
            MainWindow._prompt_open_folder_for_run(host)
        self.assertEqual(open_requests, [])
        with patch.object(QMessageBox, 'exec', accept):
            MainWindow._prompt_open_folder_for_run(host)
        self.assertEqual(open_requests, [True])

    def run_owner(self, project: ProjImgTrans, calls: list) -> SimpleNamespace:
        return SimpleNamespace(
            backup_blkstyles=[],
            _render_only=False,
            _render_global_format=None,
            textPanel=SimpleNamespace(
                formatpanel=SimpleNamespace(global_format=FontFormat())
            ),
            bottomBar=SimpleNamespace(
                textblockChecker=SimpleNamespace(isChecked=lambda: False)
            ),
            imgtrans_proj=project,
            st_manager=SimpleNamespace(updateTextBlkList=lambda: None),
            module_manager=SimpleNamespace(
                runImgtransPipeline=lambda *args, **kwargs: calls.append(
                    (args, kwargs)
                )
            ),
            tr=lambda text: text,
        )

    def test_run_with_nothing_left_tells_the_user(self) -> None:
        project = ProjImgTrans()
        project.pages = {'001.png': [], '002.png': []}
        project._image_info = {
            page: {'finish_code': RunStatus.FIN_ALL} for page in project.pages
        }
        calls = []
        owner = self.run_owner(project, calls)

        with patch.object(mainwindow_module, 'create_info_dialog') as notify:
            MainWindow.on_run_imgtrans(owner, continue_mode=True)
        notify.assert_called_once()
        self.assertTrue(notify.call_args.args[0].strip())
        self.assertEqual(calls, [])

        with patch.object(mainwindow_module, 'create_info_dialog') as notify:
            MainWindow.on_run_imgtrans(owner, pages_to_process=['missing.png'])
        notify.assert_called_once()
        self.assertEqual(calls, [])
        self.assertEqual(
            {info['finish_code'] for info in project._image_info.values()},
            {RunStatus.FIN_ALL},
        )

    def test_window_geometry_config_tolerates_old_and_invalid_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'config.json')
            for saved, expected in (
                ({}, ''),
                ({'window_geometry': 7}, ''),
                ({'window_geometry': 'AAAA'}, 'AAAA'),
            ):
                with self.subTest(saved=saved):
                    with open(path, 'w', encoding='utf8') as config_file:
                        json.dump(saved, config_file)
                    self.assertEqual(
                        ProgramConfig.load(path).window_geometry,
                        expected,
                    )

    def test_invalid_saved_geometry_falls_back_to_a_centered_default(self) -> None:
        window = QWidget()
        self.addCleanup(window.deleteLater)
        available = self.app.primaryScreen().availableGeometry()
        with patch.object(pcfg, 'window_geometry', 'not-a-geometry'):
            MainWindow.restoreMacWindowGeometry(window)
        self.assertEqual(window.geometry().center(), available.center())
        self.assertLessEqual(window.width(), available.width())
        self.assertFalse(window.isMaximized())

        window.setGeometry(40, 50, 640, 480)
        saved = bytes(window.saveGeometry().toBase64()).decode('ascii')
        restored = QWidget()
        self.addCleanup(restored.deleteLater)
        with patch.object(pcfg, 'window_geometry', saved):
            MainWindow.restoreMacWindowGeometry(restored)
        self.assertEqual(restored.size(), window.size())

    @unittest.skipUnless(shared.ON_MACOS, 'native Settings window is macOS-only')
    def test_macos_settings_window_is_native_and_ignores_outside_clicks(self) -> None:
        outside = QWidget()
        outside.resize(120, 80)
        outside.show()
        panel = ConfigPanel()
        self.addCleanup(outside.deleteLater)
        self.addCleanup(panel.deleteLater)
        panel.show()
        self.app.processEvents()

        self.assertFalse(
            panel.windowFlags() & Qt.WindowType.FramelessWindowHint
        )
        self.assertFalse(panel.title_bar.isVisible())
        QTest.mouseClick(outside, Qt.MouseButton.LeftButton)
        self.app.processEvents()
        self.assertTrue(panel.isVisible())

        QTest.keyClick(panel, Qt.Key.Key_Escape)
        self.app.processEvents()
        self.assertFalse(panel.isVisible())


if __name__ == '__main__':
    unittest.main()
