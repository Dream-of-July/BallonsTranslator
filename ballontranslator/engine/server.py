"""Qt-owned protocol operations; rendering stays on the application's real path."""
from __future__ import annotations

import copy
import math
import importlib.util
import os
import shutil
import sys
import threading
import time
from pathlib import Path

from qtpy.QtCore import QObject, Signal, Slot, Qt, QTimer
from qtpy.QtGui import QImageReader

from ballontranslator.utils import config, shared, llm_profiles
from .transport import ProtocolError, Writer, decode


class Relay(QObject):
    progress = Signal(str, int)
    preparation = Signal(str, dict)

    def __init__(self, name: str, parent: QObject) -> None:
        super().__init__(parent)
        self.name = name

    @Slot(int)
    def stage(self, done: int) -> None:
        self.progress.emit(self.name, done)

    @Slot(dict)
    def prepare(self, payload: dict) -> None:
        self.preparation.emit(self.name, payload)


class Server(QObject):
    """Own one window and dispatch all requests on its Qt thread.

    >>> Server.METHODS.count('hello')
    1
    """
    request = Signal(object)
    eof = Signal()
    failure = Signal(object)
    tested = Signal(object)
    probed = Signal(object)
    exported = Signal(object)
    METHODS = ('hello', 'open_project', 'get_page', 'run', 'cancel', 'update_blocks',
               'delete_blocks', 'list_fonts', 'get_settings', 'set_settings',
               'create_profile', 'update_profile', 'delete_profile', 'test_profile',
               'probe_vision', 'export', 'shutdown')

    def __init__(self, app: QObject, window: QObject, writer: Writer) -> None:
        super().__init__()
        self.app, self.window, self.writer = app, window, writer
        self.manager = window.module_manager
        self.active = False
        self.rendering = False
        self.exporting = False
        self.cancelled = False
        self.finishing = False
        self.quitting = False
        self.pages: list[str] = []
        self.failed: set[str] = set()
        self.written: set[str] = set()
        self.stalled = False
        self.last_progress = time.monotonic()
        self.stage_counts: dict[str, int] = {}
        self.preparing: set[QObject] = set()
        self.revisions: dict[str, int] = {}
        self.pending: tuple | None = None
        self.tests: set[threading.Thread] = set()
        self.request.connect(self.dispatch, Qt.ConnectionType.QueuedConnection)
        self.eof.connect(self.quit, Qt.ConnectionType.QueuedConnection)
        self.failure.connect(self.on_failure, Qt.ConnectionType.QueuedConnection)
        self.tested.connect(self.on_tested, Qt.ConnectionType.QueuedConnection)
        self.probed.connect(self.on_probed, Qt.ConnectionType.QueuedConnection)
        self.exported.connect(self.on_exported, Qt.ConnectionType.QueuedConnection)
        self.manager.imgtrans_pipeline_finished.connect(self.pipeline_finished)
        window.imsave_thread.img_writed.connect(self.image_written)
        self.relays: list[Relay] = []
        thread = self.manager.imgtrans_thread
        for stage in ('detect', 'ocr', 'translate', 'inpaint'):
            relay = Relay(stage, self)
            getattr(thread, 'update_' + stage + '_progress').connect(relay.stage)
            relay.progress.connect(self.stage_progress)
            self.relays.append(relay)
        self.manager.translate_thread.progress_changed.connect(self.relays[2].stage)
        self.preparation_workers = {
            'detector': self.manager.textdetect_thread,
            'ocr': self.manager.ocr_thread,
            'translator': self.manager.translate_thread,
            'inpainter': self.manager.inpaint_thread,
            'packages': self.manager.package_install_thread,
        }
        for name, worker in self.preparation_workers.items():
            worker.finished.connect(self.preparation_finished)
            relay = Relay(name, self)
            signal = (worker.package_prepare_progress if name == 'packages'
                      else worker.module_prepare_progress)
            signal.connect(relay.prepare)
            relay.preparation.connect(self.prepare_progress)
            self.relays.append(relay)
        # Capture page identity before the legacy headless path drops it. Never
        # forward exception text (providers may include credentials in errors).
        from ballontranslator.ui import module_manager
        from ballontranslator.ui import mainwindow, io_thread
        self.old_general_errors = [(module, module.create_error_dialog)
                                   for module in (module_manager, mainwindow, io_thread)]
        for module, callback in self.old_general_errors:
            module.create_error_dialog = self.capture_general_error
        self.operation_failed = False
        self.old_page_error = module_manager._create_page_error_dialog
        module_manager._create_page_error_dialog = self.capture_page_error
        self.poller = QTimer(self)
        self.poller.setInterval(20)
        self.poller.timeout.connect(self.settle)
        self.poller.start()
        self.watchdog = QTimer(self)
        self.watchdog.setInterval(100)
        self.watchdog.timeout.connect(self.check_stall)

    @property
    def project(self) -> object:
        proj = self.window.imgtrans_proj
        if not proj.directory:
            raise ProtocolError('no_project', 'Open a project first.')
        return proj

    def start_reader(self) -> None:
        threading.Thread(target=self.read, daemon=True, name='engine-input').start()

    def read(self) -> None:
        for line in sys.stdin:
            try:
                self.request.emit(decode(line))
            except ProtocolError as error:
                self.error(None, error)
        self.eof.emit()

    def error(self, request_id: object, error: ProtocolError) -> None:
        payload = {'code': error.code, 'message': str(error)}
        if error.data is not None:
            payload['data'] = error.data
        self.writer.send({'id': request_id, 'error': payload})

    @Slot(object)
    def dispatch(self, request: dict) -> None:
        request_id = request.get('id')
        try:
            if self.quitting:
                raise ProtocolError('busy', 'Engine is shutting down.')
            method = request.get('method')
            params = request.get('params', {})
            if method not in self.METHODS or not isinstance(params, dict) or 'id' not in request:
                raise ProtocolError('bad_request', 'Invalid method or parameters.')
            allowed = {
                'open_project': {'path'}, 'get_page': {'name'}, 'run': {'pages', 'mode'},
                'update_blocks': {'page', 'edits', 'autofit'}, 'delete_blocks': {'page', 'indexes'},
                'set_settings': {'source_language', 'target_language', 'translator', 'output'},
                'create_profile': {'template'}, 'update_profile': {'id', 'name', 'base_url', 'model', 'api_key'},
                'delete_profile': {'id'}, 'test_profile': {'id'}, 'probe_vision': {'profile_id'},
                'export': {'destination', 'what'},
            }.get(method, set())
            if set(params) - allowed:
                raise ProtocolError('bad_request', 'Unknown parameter.')
            if method in ('update_blocks', 'delete_blocks'):
                self.edit(request_id, params, delete=method == 'delete_blocks')
                return
            if method == 'test_profile':
                self.test_profile(request_id, params)
                return
            if method == 'probe_vision':
                self.probe_vision(request_id, params)
                return
            if method == 'export':
                self.export(request_id, params)
                return
            result = getattr(self, method)(params)
            self.writer.send({'id': request_id, 'result': result})
            if method == 'shutdown':
                self.quit()
        except ProtocolError as error:
            self.error(request_id, error)
        except (KeyError, TypeError, ValueError, IndexError, OverflowError):
            self.error(request_id, ProtocolError('bad_request', 'Invalid parameters.'))
        except Exception:
            self.error(request_id, ProtocolError('internal', 'Engine operation failed.'))

    def idle(self) -> None:
        if self.active or self.rendering or self.exporting or self.window.imsave_thread.isRunning() or self.window.imsave_thread.im_save_list:
            raise ProtocolError('busy', 'An operation is already running.')

    @staticmethod
    def string(params: dict, key: str) -> str:
        value = params[key]
        if not isinstance(value, str):
            raise ValueError(key)
        return value

    def blocks(self, name: str) -> list:
        if name not in self.project.pages:
            raise ProtocolError('not_found', 'Page does not exist.')
        return self.project.pages[name]

    def page(self, name: str) -> dict:
        blocks = self.blocks(name)
        proj = self.project
        size = QImageReader(str(Path(proj.directory) / name)).size()
        result = Path(proj.get_result_path(name))
        inpainted = Path(proj.get_inpainted_path(name, get_last_modified=True))
        return dict(name=name, source=str(Path(proj.directory) / name), width=size.width(),
                    height=size.height(), result=str(result) if result.is_file() else None,
                    inpainted=str(inpainted) if inpainted.is_file() else None,
                    done=bool(proj.get_page_progress(name)), revision=self.revisions.get(name, 0),
                    blocks=[self.block(i, blk) for i, blk in enumerate(blocks)])

    @staticmethod
    def block(index: int, blk: object) -> dict:
        fmt = blk.fontformat
        x1, y1, x2, y2 = blk.xyxy
        return dict(index=index, rect=[float(v) for v in blk.bounding_rect()], angle=float(blk.angle),
                    source_rect=[float(v) for v in (x1, y1, max(0, x2 - x1), max(0, y2 - y1))],
                    review_reason=blk.manga_review,
                    source_text=blk.get_text(), translation=blk.translation, vertical=bool(blk.vertical),
                    font_family=fmt.font_family or shared.DEFAULT_FONT_FAMILY, font_size=float(fmt.font_size),
                    color=[max(0, min(255, round(float(value)))) for value in fmt.frgb],
                    alignment=('left', 'center', 'right')[int(fmt.alignment)],
                    bold=bool(fmt.font_weight >= 600), italic=bool(fmt.italic))

    def hello(self, params: dict) -> dict:
        from ballontranslator.utils.version import APP_VERSION
        installed = importlib.util.find_spec('torch') is not None
        device = None
        if installed:
            from ballontranslator.modules import base as module_runtime
            module_runtime.refresh_torch_device_info()
            device = module_runtime.DEFAULT_DEVICE
        ready = self.models_ready()
        return dict(protocol=1, engine_version=APP_VERSION,
                    capabilities=['keep_original', 'vision_probe'],
                    device=device,
                    torch_installed=installed, models_ready=ready)

    def models_ready(self) -> bool:
        from .manga_quality import local_module
        from ballontranslator.modules.prepare_local_files import _missing_module_requirements
        for key in ('textdetector', 'ocr', 'inpainter'):
            name = local_module(key, getattr(config.pcfg.module, key))
            spec = self.manager._registry_for_module_key(key).get_spec(name)
            if spec is None or not spec.available or _missing_module_requirements(spec):
                return False
            for group in spec.download_file_list or []:
                files = group.get('save_files', group.get('files', []))
                if isinstance(files, str):
                    files = [files]
                directory = Path(shared.PROGRAM_PATH) / group.get('save_dir', '')
                if any(not (directory / file).is_file() for file in files):
                    return False
        return True

    def open_project(self, params: dict) -> dict:
        self.idle()
        path = Path(self.string(params, 'path'))
        if not path.is_absolute():
            raise ValueError('path')
        if not path.is_dir():
            raise ProtocolError('not_found', 'Folder does not exist.')
        self.operation_failed = False
        self.window.openDir(str(path))
        if self.operation_failed or Path(self.project.directory) != path:
            raise ProtocolError('internal', 'Project could not be opened.')
        self.revisions = {}
        # The folder name is what the user recognises; proj_name() is the JSON file stem.
        return dict(path=str(path), name=path.name,
                    pages=[self.page(name) for name in self.project.pages])

    def get_page(self, params: dict) -> dict:
        return self.page(self.string(params, 'name'))

    def profile(self, profile_id: str) -> object:
        profile = llm_profiles.profile_by_id(config.pcfg.module.llm_profiles, profile_id)
        if profile is None or profile.backend == 'codex':
            raise ProtocolError('not_found', 'Profile does not exist.')
        return profile

    @staticmethod
    def public_profile(profile: object) -> dict:
        return {**{key: getattr(profile, key) for key in
                   ('id', 'name', 'base_url', 'model', 'model_options', 'require_api_key')},
                'has_api_key': bool(llm_profiles.resolve_api_key(profile)),
                'vision': ('yes' if profile.vision_probe_result is True else
                           'no' if profile.vision_probe_result is False else 'unknown')}

    def preflight(self) -> None:
        cfg = config.pcfg.module
        if cfg.enable_translate and cfg.translator == 'LLMTranslator':
            try:
                profile = self.profile(cfg.translator_llm_id)
            except ProtocolError:
                raise ProtocolError('translator_not_ready', 'Select an LLM profile.', {'missing': 'model'})
            for missing in ('base_url', 'model', 'api_key'):
                value = (llm_profiles.resolve_api_key(profile) if missing == 'api_key'
                         else getattr(profile, missing))
                if missing == 'api_key' and not profile.require_api_key:
                    continue
                if not value or not value.strip():
                    raise ProtocolError('translator_not_ready', 'Translator setup is incomplete.',
                                        {'missing': missing})

    def run(self, params: dict) -> dict:
        self.idle()
        pages = params.get('pages', list(self.project.pages))
        mode = params.get('mode', 'all')
        if mode not in ('all', 'continue') or not isinstance(pages, list) or any(not isinstance(p, str) for p in pages):
            raise ValueError('run')
        for page in pages:
            self.blocks(page)
        self.preflight()
        self.pages = list(dict.fromkeys(pages))
        if mode == 'continue':
            self.pages = [p for p in self.pages if not self.project.get_page_progress(p)]
        self.active, self.cancelled, self.finishing = True, False, False
        self.stalled = False
        self.last_progress = time.monotonic()
        self.stage_counts.clear()
        try:
            seconds = float(os.environ.get('BALLOONTRANS_ENGINE_STALL_SECONDS', '600'))
        except (ValueError, OverflowError):
            seconds = 600.0
        self.stall_seconds = seconds if math.isfinite(seconds) and seconds > 0 else 600.0
        self.watchdog.start()
        self.failed.clear()
        self.written.clear()
        try:
            # Existing continue_mode ignores explicit ranges. Filter identically
            # here, then use the explicit-range path to preserve the contract.
            if self.pages:
                self.window.on_run_imgtrans(pages_to_process=self.pages)
            else:
                self.finishing = True
        except Exception:
            self.active = False
            self.watchdog.stop()
            raise
        return {'started': True}

    def cancel(self, params: dict) -> dict:
        if self.active:
            self.cancelled = True
            self.finishing = True
            # A package worker has no interrupt API. Detach its follow-up queue
            # so successful installation cannot restart inference after Cancel.
            self.manager._pending_batch_package_queue = []
            self.manager._pending_batch_package_success = None
            self.manager._pending_batch_package_failure = None
            self.manager.stopImgtransPipeline()
            self.manager.cancelModulePreparation()
            if self.manager._pending_prepare_success is not None or self.manager._pending_prepare_failure is not None:
                self.manager._finish_pending_prepare_failure()
        return {'cancelling': True}

    @Slot()
    def pipeline_finished(self) -> None:
        if self.active:
            self.finishing = True

    @Slot()
    def preparation_finished(self) -> None:
        self.preparing.discard(self.sender())

    @Slot()
    def check_stall(self) -> None:
        if not self.active or self.stalled or self.cancelled:
            return
        # Pending callbacks cover silent initial imports/model loads, including
        # the gaps between queued preparation workers. An inference worker alone
        # must not suppress this watchdog (notably parallel translation).
        preparing = (self.manager.package_install_thread.isRunning()
                     or any(worker.isRunning() for worker in self.preparing)
                     or self.manager._pending_prepare_success is not None
                     or self.manager._pending_prepare_failure is not None
                     or self.manager._pending_batch_package_success is not None
                     or self.manager._pending_batch_package_failure is not None)
        now = time.monotonic()
        if preparing:
            self.last_progress = now
        elif now - self.last_progress >= self.stall_seconds:
            self.stalled = True
            self.failed.update(page for page in self.pages if page not in self.written)
            self.writer.send(dict(event='error', message='The run stopped because a stage stopped responding.'))
            self.cancel({})
            # This is a failed run, rather than a user cancellation. Keep the
            # usual worker/save drain so a following run cannot race old work.
            self.cancelled = False
            self.watchdog.stop()

    @Slot(str, int)
    def stage_progress(self, stage: str, done: int) -> None:
        if self.active:
            if done > self.stage_counts.get(stage, -1):
                self.stage_counts[stage] = done
                self.last_progress = time.monotonic()
            self.writer.send(dict(event='stage_progress', stage=stage, done=done, total=len(self.pages)))

    @Slot(str, dict)
    def prepare_progress(self, module: str, payload: dict) -> None:
        self.preparing.add(self.preparation_workers[module])
        if self.active:
            self.last_progress = time.monotonic()
        event = payload.get('event', '')
        phase = ('downloading' if event in ('file_progress', 'file_start', 'package_download_progress')
                 else 'installing' if 'package' in event else 'loading')
        output = dict(event='prepare_progress', module=module, phase=phase,
                      message={'downloading': 'Downloading module files.',
                               'installing': 'Installing module packages.',
                               'loading': 'Loading module.'}[phase])
        for source, target in (('downloaded', 'done'), ('total', 'total')):
            if isinstance(payload.get(source), (int, float)):
                output[target] = payload[source]
        self.writer.send(output)

    def capture_page_error(self, exception: Exception, error_msg: str, exception_type: str,
                           page_key: str | None = None, page_label: str = 'Page') -> None:
        self.failure.emit(page_key)

    def capture_general_error(self, exception: Exception, error_msg: str | None = None,
                              exception_type: str | None = None) -> None:
        self.operation_failed = True
        self.failure.emit(None)

    @Slot(object)
    def on_failure(self, page: str | None) -> None:
        if self.active:
            if page:
                self.failed.add(page)
            if self.stalled:
                return
            payload = dict(event='error', message='Pipeline processing failed.')
            if page:
                payload['page'] = page
            self.writer.send(payload)

    @Slot(str)
    def image_written(self, page: str) -> None:
        if not page:
            return
        self.revisions[page] = self.revisions.get(page, 0) + 1
        if self.active:
            self.last_progress = time.monotonic()
            self.written.add(page)
            self.writer.send(dict(event='page_done', page=page, revision=self.revisions[page]))

    @Slot()
    def settle(self) -> None:
        saver = self.window.imsave_thread
        if saver.isRunning() or saver.im_save_list:
            return
        if self.pending:
            request_id, page, snapshots, edited, baseline = self.pending
            failed_save = str(Path(self.project.get_result_path(page)).with_suffix('')) in saver.failed_paths
            if self.revisions.get(page, 0) <= baseline and not failed_save:
                return
            self.pending = None
            # The GUI captures all HTML on save. Preserve untouched records and
            # clear edited HTML after capture so later edits use canonical fields.
            blocks = self.project.pages[page]
            for index, snapshot in snapshots.items():
                blocks[index] = snapshot
            for index in edited:
                blocks[index].rich_text = ''
            try:
                self.project.save()
                if str(Path(self.project.get_result_path(page)).with_suffix('')) in saver.failed_paths:
                    raise OSError('Image save failed')
                self.writer.send({'id': request_id, 'result': self.page(page)})
            except Exception:
                self.error(request_id, ProtocolError('internal', 'Page could not be saved.'))
            finally:
                self.rendering = False
        workers = (self.manager.imgtrans_thread, self.manager.translate_thread,
                   self.manager.textdetect_thread, self.manager.ocr_thread,
                   self.manager.inpaint_thread, self.manager.package_install_thread)
        if self.active and self.finishing and not any(w.isRunning() for w in workers):
            for page in self.pages:
                if not self.cancelled and page not in self.written and page not in self.failed:
                    self.failed.add(page)
                    self.writer.send(dict(event='error', page=page, message='Page did not produce a result image.'))
            self.active = False
            self.watchdog.stop()
            self.writer.send(dict(event='run_finished', cancelled=self.cancelled,
                                  failed_pages=[p for p in self.pages if p in self.failed]))
        if self.quitting and not self.active and not self.rendering and not self.exporting and not any(w.isRunning() for w in workers) and not self.tests:
            self.app.quit()

    def edit(self, request_id: object, params: dict, delete: bool) -> None:
        self.idle()
        page = self.string(params, 'page')
        blocks = self.blocks(page)
        edits = params.get('indexes' if delete else 'edits')
        if not isinstance(edits, list):
            raise ValueError('edits')
        indexes = edits if delete else [e['index'] for e in edits]
        if any(type(i) is not int or i < 0 or i >= len(blocks) for i in indexes):
            raise ValueError('index')
        if len(set(indexes)) != len(indexes) or type(params.get('autofit', False)) is not bool:
            raise ValueError('indexes')
        updated = copy.deepcopy(blocks)
        for edit in ([] if delete else edits):
            self.apply_edit(updated[edit['index']], edit)
        from .manga_quality import draft_path, save_page
        draft_file = draft_path(self.project.directory, page)
        quality_page = any(b.manga_area is not None for b in blocks) and draft_file.is_file()
        if not delete and any('keep_original' in edit for edit in edits) and not quality_page:
            raise ProtocolError('bad_request', 'Keep Original requires a manga-quality page with a repair draft. Analyse the page again first.')
        # Select before replacing project data: page changes may save the old scene.
        self.window.save_on_page_changed = False
        try:
            self.window.pageList.setCurrentRow(self.project.pagename2idx(page))
        finally:
            self.window.save_on_page_changed = False
        if delete:
            updated = [blk for i, blk in enumerate(updated) if i not in indexes]
            indexes = []
        if quality_page:
            from ballontranslator.utils.io_utils import imread
            draft = imread(str(draft_file))
            if draft is None or draft.ndim != 3 or draft.shape[-1] != 4:
                raise ProtocolError('internal', 'The page needs to be analysed again.')
            refit = {i for i in indexes if params.get('autofit', False)
                     or blocks[i].manga_review is not None}
            if not delete:
                refit.update(edit['index'] for edit in edits if 'keep_original' in edit)
            baseline = self.revisions.get(page, 0)
            previous_blocks = blocks[:]
            self.rendering = True
            try:
                blocks[:] = updated
                save_page(self.window, page, draft, refit)
                self.pending = (request_id, page, {}, indexes, baseline)
            except Exception:
                blocks[:] = previous_blocks
                self.rendering = False
                raise
            return
        self.project.pages[page][:] = updated
        snapshots = {i: copy.deepcopy(b) for i, b in enumerate(updated) if i not in indexes}
        manager = self.window.st_manager
        previous = manager.auto_textlayout_flag
        self.rendering = True
        baseline = self.revisions.get(page, 0)
        try:
            manager.auto_textlayout_flag = False
            manager.updateSceneTextitems()
            if params.get('autofit', False) and not delete:
                manager.auto_textlayout_flag = True
                for i in indexes:
                    item = manager.textblk_item_list[i]
                    if not item.blk.vertical:
                        manager.layout_textblk(item, text=item.blk.translation)
            if not self.window.saveCurrentPage(update_scene_text=True, save_proj=True, save_rst_only=True):
                raise ProtocolError('internal', 'Page could not be rendered.')
            self.pending = (request_id, page, snapshots, indexes, baseline)
        except Exception:
            self.rendering = False
            raise
        finally:
            manager.auto_textlayout_flag = previous

    @staticmethod
    def apply_edit(blk: object, edit: dict) -> None:
        allowed = {'index', 'source_text', 'translation', 'rect', 'angle', 'font_family', 'font_size',
                   'color', 'alignment', 'bold', 'italic', 'keep_original'}
        if set(edit) - allowed:
            raise ValueError('edit')
        for key, value in edit.items():
            if key in ('source_text', 'translation', 'font_family') and not isinstance(value, str):
                raise ValueError(key)
            if key in ('bold', 'italic', 'keep_original') and type(value) is not bool:
                raise ValueError(key)
            if key in ('angle', 'font_size'):
                if type(value) not in (int, float) or not math.isfinite(value) or (key == 'font_size' and value <= 0):
                    raise ValueError(key)
            if key in ('rect', 'color'):
                length = 4 if key == 'rect' else 3
                if not isinstance(value, list) or len(value) != length or any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
                    raise ValueError(key)
                if key == 'rect' and min(value[2:]) <= 0:
                    raise ValueError(key)
                if key == 'color' and any(type(v) is not int or not 0 <= v <= 255 for v in value):
                    raise ValueError(key)
        fmt = blk.fontformat
        if 'keep_original' in edit:
            blk.manga_keep_original = edit['keep_original']
            blk.manga_review = 'protected' if edit['keep_original'] else None
        if 'source_text' in edit:
            blk.text = [edit['source_text']]
            blk.manga_ocr_confidence = None
        for key in ('translation', 'angle'):
            if key in edit:
                setattr(blk, key, edit[key])
        if 'rect' in edit:
            blk._bounding_rect = edit['rect'][:]
            if blk.manga_area is None:
                blk.sync_xyxy_from_bounding_rect()
        for key in ('font_family', 'font_size', 'italic'):
            if key in edit:
                setattr(fmt, key, edit[key])
        if 'color' in edit:
            fmt.frgb = edit['color'][:]
        if 'alignment' in edit:
            fmt.alignment = ('left', 'center', 'right').index(edit['alignment'])
        if 'bold' in edit:
            fmt.font_weight = 700 if edit['bold'] else 400
        blk.rich_text = ''

    def list_fonts(self, params: dict) -> dict:
        return dict(fonts=[dict(family=e.canonical_family, display=e.display_family, source=e.source)
                           for e in shared.FONT_REGISTRY.entries(
                               only_custom=config.pcfg.let_show_only_custom_fonts_flag,
                               excluded=config.pcfg.excluded_fonts)], default=shared.DEFAULT_FONT_FAMILY)

    def get_settings(self, params: dict) -> dict:
        cfg = config.pcfg.module
        metadata = self.manager.translator_metadata()
        return dict(source_language=cfg.translate_source, target_language=cfg.translate_target,
                    languages=dict(source=metadata['supported_src_list'], target=metadata['supported_tgt_list']),
                    translator=dict(kind='llm' if cfg.translator == 'LLMTranslator' else 'google' if cfg.translator == 'google' else 'other',
                                    module=cfg.translator, profile_id=cfg.translator_llm_id if cfg.translator == 'LLMTranslator' else None),
                    profiles=[self.public_profile(p) for p in cfg.llm_profiles if p.backend != 'codex'],
                    provider_templates=llm_profiles.provider_template_names(),
                    output=dict(format=config.pcfg.imgsave_ext, quality=config.pcfg.imgsave_quality))

    def persist(self) -> None:
        if not config.save_config():
            raise ProtocolError('internal', 'Settings could not be saved.')

    def set_settings(self, params: dict) -> dict:
        self.idle()
        cfg = config.pcfg.module
        candidate = copy.deepcopy(cfg)
        translator = params.get('translator', {})
        if translator:
            kind = translator['kind']
            if kind not in ('google', 'llm'):
                raise ValueError('translator')
            candidate.translator = 'google' if kind == 'google' else 'LLMTranslator'
            if kind == 'llm':
                candidate.translator_llm_id = self.profile(translator['profile_id']).id
        meta = self.manager.translator_metadata(candidate.translator)
        for key, attr, choices in (('source_language', 'translate_source', meta['supported_src_list']),
                                   ('target_language', 'translate_target', meta['supported_tgt_list'])):
            if key in params:
                if params[key] not in choices:
                    raise ValueError(key)
                setattr(candidate, attr, params[key])
        output = params.get('output', {})
        ext, quality = output.get('format', config.pcfg.imgsave_ext), output.get('quality', config.pcfg.imgsave_quality)
        if ext not in ('.png', '.jpg', '.webp') or type(quality) is not int or not 0 <= quality <= 100:
            raise ValueError('output')
        loaded = self.manager.translator
        if loaded is not None and loaded.name == candidate.translator:
            loaded.set_source(candidate.translate_source)
            loaded.set_target(candidate.translate_target)
        for attr in ('translator', 'translator_llm_id', 'translate_source', 'translate_target'):
            setattr(cfg, attr, getattr(candidate, attr))
        config.pcfg.imgsave_ext, config.pcfg.imgsave_quality = ext, quality
        self.persist()
        return self.get_settings({})

    def create_profile(self, params: dict) -> dict:
        self.idle()
        profile = llm_profiles.profile_from_template(self.string(params, 'template'))
        config.pcfg.module.llm_profiles.append(profile)
        self.persist()
        return self.public_profile(profile)

    def update_profile(self, params: dict) -> dict:
        self.idle()
        profile = self.profile(self.string(params, 'id'))
        for key in ('name', 'base_url', 'model', 'api_key'):
            if key in params:
                self.string(params, key)
        if any(key in params and params[key] != getattr(profile, key) for key in ('base_url', 'model')):
            profile.vision_probe_result = None
        for key in ('name', 'base_url', 'model'):
            if key in params:
                setattr(profile, key, params[key])
        if 'api_key' in params:
            llm_profiles.store_api_key(profile, params['api_key'])
        self.persist()
        return self.public_profile(profile)

    def delete_profile(self, params: dict) -> dict:
        self.idle()
        profile = self.profile(self.string(params, 'id'))
        config.pcfg.module.llm_profiles.remove(profile)
        self.persist()
        return self.get_settings({})

    def test_profile(self, request_id: object, params: dict) -> None:
        profile = copy.deepcopy(self.profile(self.string(params, 'id')))
        proxy = config.pcfg.module.translator_params.get('LLMTranslator', {}).get('proxy', '')
        if isinstance(proxy, dict):
            proxy = proxy.get('value', '')
        worker = threading.Thread(target=self.check_connection, args=(request_id, profile, proxy), daemon=True)
        self.tests.add(worker)
        worker.start()

    def check_connection(self, request_id: object, profile: object, proxy: str) -> None:
        try:
            from ballontranslator.modules.llm_chat import check_profile_connection
            result = check_profile_connection(profile, proxy=proxy)
            # Provider error strings are untrusted; return stable safe messages.
            payload = dict(ok=result.ok, message='Connection successful.' if result.ok else 'Connection failed.',
                           kind=result.kind, models=result.models)
        except Exception:
            payload = dict(ok=False, message='Connection failed.', kind='internal', models=[])
        self.tested.emit((threading.current_thread(), request_id, payload))

    @Slot(object)
    def on_tested(self, result: tuple) -> None:
        worker, request_id, payload = result
        self.tests.discard(worker)
        self.writer.send({'id': request_id, 'result': payload})

    def probe_vision(self, request_id: object, params: dict) -> None:
        profile = copy.deepcopy(self.profile(self.string(params, 'profile_id')))
        proxy = config.pcfg.module.translator_params.get('LLMTranslator', {}).get('proxy', '')
        if isinstance(proxy, dict):
            proxy = proxy.get('value', '')
        worker = threading.Thread(target=self.check_vision, args=(request_id, profile, proxy), daemon=True)
        self.tests.add(worker)
        worker.start()

    def check_vision(self, request_id: object, profile: llm_profiles.LLMProfile, proxy: str) -> None:
        try:
            from .vision_probe import probe_profile_vision
            from ballontranslator.modules.llm_chat import LLMConnectionError
            from ballontranslator.modules.exceptions import LLMApiKeyRequiredError
            try:
                payload = dict(vision=probe_profile_vision(profile, proxy=proxy))
            except LLMApiKeyRequiredError:
                payload = dict(ok=False, message='Connection failed.', kind='auth', models=[])
            except LLMConnectionError as error:
                payload = dict(ok=False, message='Connection failed.', kind=error.kind, models=[])
        except Exception:
            payload = dict(ok=False, message='Connection failed.', kind='internal', models=[])
        self.probed.emit((threading.current_thread(), request_id, profile, payload))

    @Slot(object)
    def on_probed(self, result: tuple) -> None:
        worker, request_id, snapshot, payload = result
        self.tests.discard(worker)
        try:
            if 'vision' in payload:
                profile = self.profile(snapshot.id)
                # A settings edit while the request was pending invalidates its result.
                if any(getattr(profile, key) != getattr(snapshot, key)
                       for key in ('base_url', 'model', 'vision_model')):
                    raise ProtocolError('busy', 'Profile changed during the vision probe. Try again.')
                previous = profile.vision_probe_result, profile.support_vision
                profile.vision_probe_result = profile.support_vision = payload['vision'] == 'yes'
                try:
                    self.persist()
                except Exception:
                    profile.vision_probe_result, profile.support_vision = previous
                    raise
            self.writer.send({'id': request_id, 'result': payload})
        except ProtocolError as error:
            self.error(request_id, error)
        except Exception:
            self.error(request_id, ProtocolError('internal', 'Settings could not be saved.'))

    def export(self, request_id: object, params: dict) -> None:
        self.idle()
        proj = self.project
        destination = Path(self.string(params, 'destination'))
        what = params['what']
        if not destination.is_absolute() or what not in ('images', 'translation_txt', 'source_txt'):
            raise ValueError('export')
        self.exporting = True
        threading.Thread(target=self.copy_export,
                         args=(request_id, proj, destination, what), daemon=True).start()

    def copy_export(self, request_id: object, proj: object, destination: Path, what: str) -> None:
        try:
            destination.mkdir(parents=True, exist_ok=True)
            if what == 'images':
                paths = [Path(proj.get_result_path(p)) for p in proj.pages]
                paths = [p for p in paths if p.is_file()]
            else:
                target = 'translation' if what == 'translation_txt' else 'source'
                proj.dump_txt(target, '.txt')
                paths = [Path(proj.dump_txt_path(target, '.txt'))]
            files = []
            for source in paths:
                target = destination / source.name
                if source.resolve() != target.resolve():
                    shutil.copy2(source, target)
                files.append(str(target))
            self.exported.emit((request_id, files))
        except Exception:
            self.exported.emit((request_id, None))

    @Slot(object)
    def on_exported(self, payload: tuple) -> None:
        request_id, files = payload
        self.exporting = False
        if files is None:
            self.error(request_id, ProtocolError('internal', 'Export could not be completed.'))
        else:
            self.writer.send({'id': request_id, 'result': {'files': files}})

    def shutdown(self, params: dict) -> dict:
        return {'ok': True}

    @Slot()
    def quit(self) -> None:
        self.quitting = True
        self.cancel({})

    def dispose(self) -> None:
        from ballontranslator.ui import module_manager
        module_manager._create_page_error_dialog = self.old_page_error
        for module, callback in self.old_general_errors:
            module.create_error_dialog = callback
        self.poller.stop()
        self.watchdog.stop()
        self.window.imsave_thread.wait_for_saves(None)
        self.window.deleteLater()
        from qtpy.QtCore import QCoreApplication, QEvent
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
