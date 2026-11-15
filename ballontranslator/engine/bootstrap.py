"""Headless application bootstrap without CLI restarts or batch stdin prompts."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from ballontranslator.utils.textblock import TextBlock


def bootstrap() -> tuple:
    os.environ.setdefault('QT_API', 'pyqt6')
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    if os.sys.platform == 'darwin':
        # Cocoa uses 72 logical DPI. Match both Qt's point-size device and LDPI,
        # including existing GUI HTML whose font sizes are stored in points.
        os.environ['QT_FONT_DPI'] = '72'
    from ballontranslator.utils import shared
    shared.HEADLESS = True  # MainWindow chooses its Qt base at import time.
    shared.ENGINE_IMAGES_LOCAL = True
    shared.ENGINE_TRANSLATION_AUTO_THINKING_DISABLED = True
    os.environ.setdefault('HF_HOME', str(Path(shared.PROGRAM_PATH) / 'data/models'))
    from ballontranslator.launch import parser, setup_locks, application_arguments
    args = parser.parse_args([])
    args.headless = True
    shared.args = args
    shared.load_cache()
    from ballontranslator.utils.logger import logger
    # Legacy failures log raw provider exception bodies; protocol errors are safe.
    logger.disabled = True
    from ballontranslator.utils import config
    config.load_config(os.environ.get('BALLOONTRANS_ENGINE_CONFIG', shared.CONFIG_PATH))
    pcfg = config.pcfg
    from qtpy.QtCore import QLocale, Signal, Slot, Qt
    shared.DEFAULT_DISPLAY_LANG = QLocale.system().name().replace('en_CN', 'zh_CN')
    from qtpy import API_NAME, QT_VERSION
    from qtpy.QtWidgets import QApplication
    from qtpy.QtGui import QFontDatabase, QFont
    shared.FLAG_QT6 = API_NAME.endswith('6')
    shared.USE_PYSIDE6 = API_NAME == 'PySide6'
    app = QApplication(application_arguments(['engine'], headless=True,
        platform_name=os.sys.platform, qt_version=QT_VERSION, font_backend='default'))
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName('BalloonsTranslator')
    from ballontranslator.utils.version import APP_VERSION
    app.setApplicationVersion(APP_VERSION)
    # QTextDocument's point sizes use the offscreen screen's logical DPI too.
    shared.LDPI = app.primaryScreen().logicalDotsPerInch()
    database = QFontDatabase if shared.FLAG_QT6 else QFontDatabase()
    system = sorted(database.families(), key=str.casefold)
    from ballontranslator.utils.font_registry import build_font_registry, ensure_font_registry_overrides
    fonts = Path(shared.PROGRAM_PATH) / 'fonts'
    paths = sorted(str(p.resolve()) for p in fonts.rglob('*')
                   if p.suffix.lower() in {'.ttf', '.otf', '.ttc', '.pfb'})
    overrides = ensure_font_registry_overrides(shared.PROGRAM_PATH)
    shared.FONT_REGISTRY = build_font_registry(database, paths, system, locale=pcfg.display_lang,
        font_registry_path=str(overrides) if overrides is not None else None)
    shared.FONT_FAMILIES = set(database.families())
    from ballontranslator.ui.text_engine.font_family import register_qt_font_family_aliases
    register_qt_font_family_aliases(database.families(), database.styles)
    font = app.font()
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.NoSubpixelAntialias)
    app.setFont(font)
    shared.DEFAULT_FONT_FAMILY = shared.APP_DEFAULT_FONT = font.family()
    setup_locks()
    from ballontranslator.ui.mainwindow import MainWindow

    class EngineWindow(MainWindow):
        quality_compose_requested = Signal(object)

        @Slot(object)
        def compose_quality_on_main(self, job: dict) -> None:
            # Native text layout/scene objects retain their existing Qt-thread
            # ownership. Only this short render crosses back from page work.
            try:
                if job['stop'].is_set() or time.monotonic() >= job['deadline']:
                    return
                from .manga_quality import compose_page
                job['result'] = compose_page(job['image'], job['draft'], job['blocks'],
                                             job['refit'], force_outline=job['outline'])
            except Exception as error:
                job['error'] = error
            finally:
                job['done'].set()

        def compose_quality(
            self, image: np.ndarray, draft: np.ndarray, blocks: list[TextBlock], refit: set[int] | None,
            outline: set[int] | None, deadline: float, stop: threading.Event,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            from ballontranslator.modules.exceptions import LLMRequestStopped
            job = dict(image=image, draft=draft, blocks=blocks, refit=refit, outline=outline,
                       deadline=deadline, stop=stop, done=threading.Event())
            self.quality_compose_requested.emit(job)
            while not job['done'].wait(.05):
                if stop.is_set():
                    raise LLMRequestStopped()
                if time.monotonic() >= deadline:
                    raise TimeoutError('Enhance composition timed out.')
            if 'error' in job:
                raise job['error']
            if 'result' not in job:
                raise LLMRequestStopped()
            return job['result']

        def process_quality_page(self, name: str, image: np.ndarray, blocks: list[TextBlock]) -> None:
            worker = self.module_manager.imgtrans_thread
            draft = worker.manga_drafts[name]
            normal = self.compose_quality(image, draft, blocks, None, None,
                                          float('inf'), worker.stop_event)
            session = getattr(self, 'enhance_session', None)
            if session is not None:
                draft, normal = session.process(image, draft, blocks, normal,
                    stop=worker.stop_event, translator=worker.translator, compose=self.compose_quality)
            worker.manga_drafts[name] = draft
            worker.manga_composed[name] = normal

        def on_pagtrans_finished(self, page_index: int) -> None:
            worker = self.module_manager.imgtrans_thread
            if not worker.manga_quality_enabled:
                return super().on_pagtrans_finished(page_index)
            name = self.imgtrans_proj.idx2pagename(page_index)
            draft = worker.manga_drafts.pop(name, None)
            if draft is not None and self.imgtrans_proj.get_page_progress(name):
                from .manga_quality import save_page
                try:
                    save_page(self, name, draft, composed=worker.manga_composed.pop(name, None))
                except Exception as error:
                    from ballontranslator.ui import module_manager
                    module_manager._create_page_error_dialog(
                        error, 'Page could not be rendered.', 'MangaQualityRenderFailed', name,
                    )

        def run_batch(self, exec_dirs: str = '', **kwargs: object) -> None:
            self.exec_dirs = []

        def run_next_dir(self) -> None:
            pass

        def on_imgtrans_pipeline_finished(self) -> None:
            self.backup_blkstyles.clear()
            self._run_imgtrans_wo_textstyle_update = False
            self._render_only = False
            self._render_global_format = None

        def on_create_errdialog(self, error_msg: str, detail_traceback: str = '',
                                exception_type: str = '') -> None:
            # Never create a modal dialog in an unattended process.
            pass

    old_recent, old_update = pcfg.open_recent_on_startup, pcfg.check_update_on_startup
    pcfg.open_recent_on_startup = pcfg.check_update_on_startup = False
    try:
        window = EngineWindow(app, pcfg, **vars(args))
    finally:
        # Keep startup-only overrides out of subsequent explicit config saves.
        pcfg.open_recent_on_startup, pcfg.check_update_on_startup = old_recent, old_update
    window.save_on_page_changed = False
    window.quality_compose_requested.connect(window.compose_quality_on_main, Qt.ConnectionType.QueuedConnection)
    window.module_manager.imgtrans_thread.native_quality = True
    window.hide()
    return app, window
