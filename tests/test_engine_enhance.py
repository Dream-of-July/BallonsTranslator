"""Offline enhanced-mode contracts, including the real Qt render/save bridge."""
import copy
import io
import json
import os
import socket
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from ballontranslator.engine import manga_enhance as enhance
from ballontranslator.engine import manga_quality as quality
from ballontranslator.modules.llm_chat import LLMChatResult
from ballontranslator.utils import config, shared
from ballontranslator.utils.llm_profiles import default_profile
from ballontranslator.utils.textblock import TextBlock
from ballontranslator.utils.io_utils import json_dump_nested_obj

_QT_ENGINE = None


@pytest.fixture(scope='module', autouse=True)
def engine(tmp_path_factory):
    from ballontranslator.engine.bootstrap import bootstrap
    from ballontranslator.engine.server import Server
    from ballontranslator.engine.transport import Writer
    global _QT_ENGINE
    root = tmp_path_factory.mktemp('engine-enhance')
    with patch.dict(os.environ, BALLOONTRANS_ENGINE_CONFIG=str(root / 'settings.json')), \
            patch.object(socket.socket, 'connect', side_effect=AssertionError('Network forbidden in offline tests')):
        app, window = bootstrap()
        sink = io.StringIO()
        server = Server(app, window, Writer(sink))
        _QT_ENGINE = SimpleNamespace(app=app, window=window, server=server, sink=sink, root=root)
        yield _QT_ENGINE
        server.dispose()


def session(enabled=True, vision=True, *, profile_id=None, translator='LLMTranslator'):
    profile = default_profile('LM Studio')
    profile.model = 'offline-model'
    profile.vision_probe_result = vision
    cfg = config.ModuleConfig(llm_profiles=[profile], translator=translator,
        translator_llm_id=profile.id, enhance_enabled=enabled, enhance_profile_id=profile_id)
    events = []
    return enhance.EnhanceSession(cfg, events.append), events


def page(*, unreadable=False):
    from test_native_manga_quality import comic_page
    image, mask, merged = comic_page()
    blocks = quality.identify_regions(image, [merged])
    for block in blocks:
        block.text, block.translation = ['原文'], '中文'
    if unreadable:
        blocks[0].text = ['']
    draft = quality.prepare_background(image, mask, blocks, None)
    normal = quality.compose_page(image, draft, blocks)
    return image, draft, blocks, normal


def compose(image, draft, blocks, changed, outlined, deadline, cancelled):
    return _QT_ENGINE.window.compose_quality(image, draft, blocks, changed, outlined, deadline, cancelled)


class Translator:
    calls = []

    def __init__(self, source='日本語', target='简体中文', **params):
        self.lang_source, self.lang_target, self.params = source, target, params

    def translate_textblk_lst(self, blocks):
        type(self).calls.append([b.get_text() for b in blocks])
        for block in blocks:
            block.translation = '修正'


def reply(index):
    return dict(index=index, is_dialogue=True, source_text_correct=True,
                corrected_text='', residue=False, legible=True)


def response(items):
    return LLMChatResult(content=json.dumps({'blocks': items}))


def process(session, data, *, stop=None, translator=None, composer=compose):
    image, draft, blocks, normal = data
    run_stop = stop or threading.Event()
    def invoke():
        return session.process(image, draft, blocks, normal, stop=run_stop,
                               translator=translator or Translator(), compose=composer)
    if threading.current_thread() is not threading.main_thread():
        return invoke()
    results, errors = [], []
    def run():
        try:
            results.append(invoke())
        except BaseException as error:
            errors.append(error)
    worker = threading.Thread(target=run)
    worker.start()
    deadline = time.monotonic() + 10
    while worker.is_alive() and time.monotonic() < deadline:
        _QT_ENGINE.app.processEvents()
        time.sleep(.005)
    if worker.is_alive():
        run_stop.set()
    worker.join(1)
    assert not worker.is_alive()
    if errors:
        raise errors[0]
    return results[0]


@pytest.mark.parametrize('enabled,vision', [(False, True), (True, None), (True, False)])
def test_no_images_or_requests_without_explicit_enabled_measured_yes(enabled, vision):
    run, events = session(enabled, vision)
    data = page(unreadable=True)
    with patch.object(enhance, 'encode_chat_image') as encode, \
            patch.object(enhance.EnhanceRequester, 'request_chat_completion') as request:
        draft, output = process(run, data)
    assert draft is data[1] and output is data[3]
    encode.assert_not_called()
    request.assert_not_called()
    assert len(events) == 1 and events[0]['event'] == 'log'


def test_non_llm_has_no_implicit_profile_and_explicit_profile_is_used():
    run, events = session(translator='google')
    assert run.profile is None
    profile = default_profile('LM Studio')
    profile.vision_probe_result = True
    cfg = config.ModuleConfig(llm_profiles=[profile], translator='google',
        enhance_enabled=True, enhance_profile_id=profile.id)
    assert enhance.EnhanceSession(cfg, events.append).profile.id == profile.id


def test_settings_roundtrip_partial_null_unknown_and_config_defaults(engine):
    server = engine.server
    profile = default_profile('LM Studio')
    with patch.object(config.pcfg.module, 'llm_profiles', [profile]), \
            patch.object(config.pcfg.module, 'enhance_enabled', False), \
            patch.object(config.pcfg.module, 'enhance_profile_id', None), \
            patch.object(server, 'persist'):
        assert server.get_settings({})['enhance'] == {'enabled': False, 'profile_id': None}
        assert server.set_settings({'enhance': {'enabled': True, 'profile_id': profile.id}})['enhance'] == {
            'enabled': True, 'profile_id': profile.id}
        assert server.set_settings({'enhance': {}})['enhance']['profile_id'] == profile.id
        assert server.set_settings({'enhance': {'profile_id': None}})['enhance'] == {
            'enabled': True, 'profile_id': None}
        engine.sink.seek(0)
        engine.sink.truncate()
        for i, payload in enumerate(({'profile_id': 'missing'}, {'enabled': 1}, {'profile_id': 1}, {'unknown': True})):
            server.dispatch({'id': i, 'method': 'set_settings', 'params': {'enhance': payload}})
        assert all(json.loads(line)['error']['code'] == 'bad_request'
                   for line in engine.sink.getvalue().splitlines())
        assert server.get_settings({})['enhance'] == {'enabled': True, 'profile_id': None}
    old = config.ModuleConfig()
    assert not old.enhance_enabled and old.enhance_profile_id is None
    assert not TextBlock().manga_enhanced and not TextBlock().manga_non_dialogue
    # Actual persisted representation uses defaulted module fields.
    saved = config.ModuleConfig(enhance_enabled=True, enhance_profile_id=profile.id)
    persisted = config.json_dump_program_config(config.ProgramConfig(module=saved))
    restored = config.ProgramConfig(**json.loads(persisted)).module
    assert restored.enhance_enabled and restored.enhance_profile_id == profile.id


def test_clean_page_never_encodes():
    run, _ = session()
    data = page()
    # Flat verified bubbles have no suspicious flags.
    assert not enhance.suspicious_blocks(data[2])
    with patch.object(enhance, 'encode_chat_image') as encode:
        process(run, data)
    encode.assert_not_called()


def test_selection_caps_and_only_suspicious_crops():
    blocks = [TextBlock(xyxy=[i, i, i + 20, i + 20], manga_review='unreadable') for i in range(30)]
    blocks += [TextBlock(xyxy=[0, 0, i + 1, i + 1], manga_review='skipped') for i in range(10)]
    selected = enhance.suspicious_blocks(blocks)
    assert len(selected) == 24
    assert [i for i in selected if i >= 30] == [39, 38, 37, 36, 35, 34]
    # No area/geometry is needed unless a block is actually selected.
    image = np.full((800, 900, 3), 255, np.uint8)
    draft = np.dstack((image, np.zeros(image.shape[:2], np.uint8)))
    normal = image.copy(), image.copy(), draft[..., 3]
    blocks.append(TextBlock(xyxy=[500, 500, 510, 510]))
    run, _ = session()
    payloads = []
    def respond(self, profile, args):
        payloads.append((profile, args))
        ids = [json.loads(p['text'])['index'] for p in args['messages'][1]['content']
               if p['type'] == 'text' and p['text'].startswith('{')]
        return response([reply(i) for i in ids])
    with patch.object(enhance.EnhanceRequester, 'request_chat_completion', respond), \
            patch.object(enhance, 'encode_chat_image', wraps=enhance.encode_chat_image) as encode:
        process(run, (image, draft, blocks, normal))
    assert max(enhance._crop(image, TextBlock(xyxy=[0, 0, 800, 700])).shape[:2]) == 512
    assert len(payloads) == 2 and encode.call_count == 48
    assert all(max(call.args[0].shape[:2]) <= 512 for call in encode.call_args_list)
    ids = [json.loads(p['text'])['index'] for _, args in payloads for p in args['messages'][1]['content']
           if p['type'] == 'text' and p['text'].startswith('{')]
    assert ids == selected and 40 not in ids
    assert all(profile.id == run.profile.id for profile, _ in payloads)
    assert all(args['response_format']['type'] == 'json_schema' for _, args in payloads)


@pytest.mark.parametrize('is_dialogue', [True, False])
def test_corrected_unreadable_is_retranslated_and_drawn(is_dialogue):
    data = page(unreadable=True)
    run, events = session()
    Translator.calls.clear()
    with patch.object(enhance.EnhanceRequester, 'request_chat_completion',
                      return_value=response([reply(0) | {'is_dialogue': is_dialogue,
                          'corrected_text': '修正文', 'source_text_correct': False}])):
        _, output = process(run, data)
    block = data[2][0]
    assert Translator.calls == [['修正文']]
    assert block.get_text() == '修正文' and block.translation == '修正', events
    assert block.manga_enhanced and block.manga_review is None
    assert np.any(output[1] != data[3][1])
    assert 'corrected=1' in [e['message'] for e in events if e['event'] == 'log'][-1]


def test_not_dialogue_preserves_original_and_marks_skipped():
    data = page(unreadable=True)
    block = data[2][0]
    block.manga_area = quality._plain_area(block, data[0].shape[1], data[0].shape[0])
    assert not quality._inside_bubble(block, data[0].shape[:2])
    run, _ = session()
    with patch.object(enhance.EnhanceRequester, 'request_chat_completion',
                      return_value=response([reply(0) | {'is_dialogue': False}])):
        _, output = process(run, data)
    assert data[2][0].manga_review == 'skipped' and data[2][0].manga_enhanced
    assert data[2][0].manga_non_dialogue
    assert np.array_equal(output[1][:, 120:], data[0][:, 120:])
    assert quality.source_review(data[2][0]) == 'skipped'


@pytest.mark.parametrize('unreadable', [True, False])
@pytest.mark.parametrize('tier', ['pale', 'nonwhite'])
def test_in_bubble_negative_verdict_preserves_review_and_drawn_pixels(unreadable, tier):
    if tier == 'pale':
        data = page(unreadable=unreadable)
    else:
        import cv2
        y, x = np.indices((240, 320))
        art = np.stack((35 + x * .2, 55 + y * .15, 70 + x * .1), axis=2).astype(np.uint8)
        image = art.copy()
        bubble = np.zeros(image.shape[:2], np.uint8)
        cv2.ellipse(bubble, (105, 120), (66, 78), 0, 0, 360, 255, -1)
        image[bubble != 0] = ((art[bubble != 0].astype(float) + 255) * .5).astype(np.uint8)
        cv2.ellipse(image, (105, 120), (66, 78), 0, 0, 360, (15, 15, 15), 3)
        cv2.rectangle(image, (98, 96), (110, 143), (8, 8, 8), -1)
        line = [[92, 90], [116, 90], [116, 149], [92, 149]]
        block = quality.identify_regions(image, [TextBlock(
            xyxy=[92, 90, 116, 149], lines=[line], _detected_font_size=24,
            text=['' if unreadable else '原文'], translation='中文')])[0]
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[96:144, 98:111] = 255
        draft = quality.prepare_background(image, mask, [block], None)
        data = image, draft, [block], quality.compose_page(image, draft, [block])
    block = data[2][0]
    assert quality._inside_bubble(block, data[0].shape[:2])
    assert block.manga_review == ('unreadable' if unreadable else None)
    if not unreadable:
        block.manga_complex_background = True
    before = json_dump_nested_obj(data[2])
    run, _ = session()
    with patch.object(enhance.EnhanceRequester, 'request_chat_completion',
                      return_value=response([reply(0) | {'is_dialogue': False}])):
        draft, output = process(run, data)
    assert draft is data[1] and output is data[3]
    assert json_dump_nested_obj(data[2]) == before
    assert not block.manga_non_dialogue


@pytest.mark.parametrize('source', ['…', '……', '!?', ' — -- ♡ ★ \t', ' \t\n'])
def test_symbols_only_are_skipped_and_never_sent(source):
    image, draft, blocks, _ = page()
    blocks[0].text, blocks[0].translation = [source], source
    normal = quality.compose_page(image, draft, blocks)
    assert blocks[0].manga_review == 'skipped'
    blocks[0].manga_complex_background = True
    blocks[0].manga_line_confidence = [.1]
    assert 0 not in enhance.suspicious_blocks(blocks)
    assert np.array_equal(normal[1][:, 120:], image[:, 120:])
    run, _ = session()
    with patch.object(enhance, 'encode_chat_image') as encode, \
            patch.object(enhance.EnhanceRequester, 'request_chat_completion') as request:
        process(run, (image, draft, blocks, normal))
    encode.assert_not_called()
    request.assert_not_called()
    blocks[0].manga_review = 'untranslated'
    assert 0 not in enhance.suspicious_blocks(blocks)


@pytest.mark.parametrize('vertical', [True, False])
def test_request_explains_empty_ocr_and_sends_detected_orientation(vertical):
    data = page(unreadable=True)
    data[2][0].src_is_vertical = vertical
    run, _ = session()
    with patch.object(enhance.EnhanceRequester, 'request_chat_completion',
                      return_value=response([reply(0)])) as request:
        process(run, data)
    args = request.call_args.args[1]
    prompt = args['messages'][0]['content']
    assert 'OCR is empty or wrong' in prompt
    assert 'source_text_correct\nMUST be false' in prompt
    hint = json.loads(args['messages'][1]['content'][0]['text'])
    assert hint['ocr_source'] == ''
    assert hint['orientation'] == ('vertical' if vertical else 'horizontal')
    assert hint['inside_bubble'] is True


@pytest.mark.parametrize('mode', ['unset', 'missing', 'existing', 'unwritable'])
def test_debug_output_is_opt_in_text_only_and_does_not_change_result(tmp_path, monkeypatch, mode):
    monkeypatch.delenv('BALLOONTRANS_ENHANCE_DEBUG_DIR', raising=False)
    directory = tmp_path / 'diagnostics'
    if mode != 'unset':
        monkeypatch.setenv('BALLOONTRANS_ENHANCE_DEBUG_DIR', str(directory))
    if mode in {'existing', 'unwritable'}:
        directory.mkdir()
    data = page(unreadable=True)
    run, _ = session()
    raw = response([reply(0) | {'is_dialogue': False}])
    with patch.object(enhance.EnhanceRequester, 'request_chat_completion', return_value=raw), \
            patch.object(Path, 'open', side_effect=OSError) if mode == 'unwritable' else \
            nullcontext():
        _, output = process(run, data)
    assert output is data[3] and data[2][0].manga_review == 'unreadable'
    files = list(tmp_path.rglob('*.json'))
    if mode != 'existing':
        assert not files
        if mode == 'missing':
            assert not directory.exists()
        return
    assert len(files) == 1
    text = files[0].read_text()
    diagnostic = json.loads(text)
    record = diagnostic['requests'][0]
    assert record['raw_reply'] == raw.content
    assert record['verdicts']['0']['is_dialogue'] is False
    assert json.loads(record['request_text'][1])['ocr_source'] == ''
    assert diagnostic['blocks']['0'] == {
        'review_before': 'unreadable', 'action': 'unchanged', 'review_after': 'unreadable'}
    assert diagnostic['outcome'] == 'completed'
    assert all(value not in text for value in ('image_url', 'data:image', 'base64', 'headers', 'api_key'))


def test_debug_records_applied_correction_and_each_page(tmp_path, monkeypatch):
    monkeypatch.setenv('BALLOONTRANS_ENHANCE_DEBUG_DIR', str(tmp_path))
    run, _ = session()
    with patch.object(enhance.EnhanceRequester, 'request_chat_completion', return_value=response([
            reply(0) | {'corrected_text': '修正文', 'source_text_correct': False}])):
        process(run, page(unreadable=True))
        process(run, page())
    diagnostics = [json.loads(file.read_text()) for file in tmp_path.glob('*.json')]
    assert len(diagnostics) == 2
    corrected = next(d for d in diagnostics if d['requests'])
    assert corrected['blocks']['0']['action'] == 'applied'
    assert corrected['blocks']['0']['changes'] == ['corrected']
    assert corrected['blocks']['0']['review_after'] is None


def test_debug_records_invalid_raw_reply_without_changing_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv('BALLOONTRANS_ENHANCE_DEBUG_DIR', str(tmp_path))
    run, _ = session()
    data = page(unreadable=True)
    with patch.object(enhance.EnhanceRequester, 'request_chat_completion',
                      return_value=LLMChatResult(content='invalid json')):
        _, output = process(run, data)
    assert output is data[3]
    diagnostic = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert diagnostic['outcome'] == 'review failed'
    assert diagnostic['requests'][0]['raw_reply'] == 'invalid json'
    assert diagnostic['requests'][0]['verdicts'] == {}
    assert diagnostic['blocks']['0']['action'] == 'unchanged'


def test_residue_invokes_widened_repair_and_illegible_forces_outline_backing():
    data = page()
    data[2][0].manga_complex_background = True
    run, events = session()
    calls = []
    def render(*args):
        calls.append(args[4])
        return compose(*args)
    with patch.object(enhance.EnhanceRequester, 'request_chat_completion',
                      return_value=response([reply(0) | {'residue': True, 'legible': False}])), \
            patch.object(quality, '_repair_complex', wraps=quality._repair_complex) as repair:
        _, output = process(run, data, composer=render)
    assert any(call.kwargs['widened'] for call in repair.call_args_list)
    assert calls == [{0}]
    assert data[2][0].manga_outline_applied and data[2][0].manga_backing_applied
    assert 're-erased=1' in [e['message'] for e in events if e['event'] == 'log'][-1] and 'restyled=1' in [e['message'] for e in events if e['event'] == 'log'][-1]
    assert np.any(output[1] != data[3][1])


def test_bad_fit_withholds_with_normal_reason_and_other_block_unchanged():
    data = page(unreadable=True)
    before_other = json_dump_nested_obj(data[2][1])
    class LongTranslator(Translator):
        def translate_textblk_lst(self, blocks):
            for block in blocks:
                block.translation = '放不下的超长文字。' * 80
    run, events = session()
    with patch.object(enhance.EnhanceRequester, 'request_chat_completion',
                      return_value=response([reply(0) | {'corrected_text': '修正文'}])):
        _, output = process(run, data, translator=LongTranslator())
    assert data[2][0].manga_review == 'overflow' and not data[2][0].manga_enhanced
    assert json_dump_nested_obj(data[2][1]) == before_other
    assert np.array_equal(output[1], data[3][1])
    assert 'abandoned=1' in [e['message'] for e in events if e['event'] == 'log'][-1]


@pytest.mark.parametrize('failure', ['json', 'missing', 'timeout', 'http', 'second-batch'])
def test_failures_keep_normal_result_and_run_continues(failure):
    data = page(unreadable=True)
    if failure == 'second-batch':
        data[2].extend(copy.deepcopy(data[2][0]) for _ in range(12))
    before = json_dump_nested_obj(data[2])
    run, events = session()
    responses = [LLMChatResult(content='not json')]
    if failure == 'missing':
        responses = [response([{'index': 0}])]
    elif failure == 'timeout':
        responses = [TimeoutError('private provider details')]
    elif failure == 'http':
        responses = [RuntimeError('HTTP failure: private provider details')]
    elif failure == 'second-batch':
        responses = [response([reply(0) | {'corrected_text': '修正文'}]), RuntimeError('HTTP failure')]
    with patch.object(enhance.EnhanceRequester, 'request_chat_completion', side_effect=responses):
        draft, output = process(run, data)
    assert draft is data[1] and output is data[3]
    assert json_dump_nested_obj(data[2]) == before
    assert any(e['event'] == 'log' and 'skipped' in e['message'] for e in events)
    assert not any(e['event'] == 'error' for e in events)
    assert 'private provider details' not in json.dumps(events)
    # Same run can still enhance the next page.
    next_page = page(unreadable=True)
    with patch.object(enhance.EnhanceRequester, 'request_chat_completion',
                      return_value=response([reply(0) | {'corrected_text': '修正文'}])):
        process(run, next_page)
    assert next_page[2][0].manga_enhanced


def test_parse_drops_wrong_types_and_unrequested_indexes():
    content = json.dumps({'blocks': [reply(0), reply(1) | {'residue': 'true'}, reply(99), reply(True)]})
    assert set(enhance._parse('```json\n' + content + '\n```', [0, 1])) == {0}
    with pytest.raises(ValueError):
        enhance._parse(json.dumps({'blocks': [reply(0), reply(0)]}), [0])


@pytest.mark.parametrize('cancel', [False, True])
def test_bounded_total_page_budget_and_cancel_no_late_publication(cancel):
    data = page(unreadable=True)
    before = json_dump_nested_obj(data[2])
    run, events = session()
    run.params['request timeout'] = .15 if not cancel else 5
    entered, release, stop = threading.Event(), threading.Event(), threading.Event()
    def blocked(*args):
        entered.set()
        release.wait(3)
        return response([reply(0) | {'corrected_text': '修正文'}])
    results = []
    with patch.object(enhance.EnhanceRequester, 'request_chat_completion', blocked):
        worker = threading.Thread(target=lambda: results.append(process(run, data, stop=stop)))
        start = time.monotonic()
        worker.start()
        try:
            assert entered.wait(1)
            if cancel:
                stop.set()
            worker.join(1)
            assert not worker.is_alive() and time.monotonic() - start < 1
            assert results[0][1] is data[3]
        finally:
            release.set()
            worker.join(2)
    assert json_dump_nested_obj(data[2]) == before
    assert not any(e['event'] == 'error' for e in events)


def test_translation_request_is_still_text_only_when_enhance_enabled():
    from ballontranslator.modules.translators.trans_llm import LLMTranslator
    from ballontranslator.modules.llm_chat import LLMChatRequester
    data = page(unreadable=True)
    run, _ = session()
    profile = default_profile('LM Studio')
    translator = LLMTranslator('日本語', '简体中文')
    translator.name = 'LLMTranslator'
    calls = []
    def translate_request(self, profile, args, **kwargs):
        calls.append(args)
        return LLMChatResult(content='{"translations":[{"id":1,"translation":"修正"}]}')
    with patch.object(config.pcfg.module, 'llm_profiles', [profile]), \
            patch.object(config.pcfg.module, 'translator_llm_id', profile.id), \
            patch.object(config.pcfg.module, 'llm_translate_vision', True), \
            patch.object(shared, 'ENGINE_IMAGES_LOCAL', True), \
            patch.object(LLMChatRequester, 'request_chat_completion', translate_request), \
            patch.object(enhance.EnhanceRequester, 'request_chat_completion',
                         return_value=response([reply(0) | {'corrected_text': '修正文'}])):
        process(run, data, translator=translator)
    assert calls
    assert 'image_url' not in json.dumps(calls) and 'data:image' not in json.dumps(calls)
    assert data[2][0].manga_enhanced


def test_qt_bridge_and_real_save_order(engine, tmp_path):
    from PIL import Image
    from qtpy.QtCore import QThread
    server, window, app = engine.server, engine.window, engine.app
    data = page(unreadable=True)
    Image.fromarray(data[0]).save(tmp_path / '001.png')
    project_json = tmp_path / ('imgtrans_' + tmp_path.name + '.json')
    project_json.write_text(json_dump_nested_obj({'pages': {'001.png': data[2]},
                                       'current_img': '001.png', 'image_info': {}}))
    server.open_project({'path': str(tmp_path)})
    engine.sink.seek(0)
    engine.sink.truncate()
    worker = window.module_manager.imgtrans_thread
    worker.manga_quality_enabled = True
    worker.manga_drafts['001.png'] = data[1]
    # Actual pipeline QThread invokes the engine hook. The response stub must
    # execute away from Qt's main thread; native composition stays on it.
    class PageWorker(QThread):
        def run(self):
            window.process_quality_page('001.png', data[0], server.project.pages['001.png'])
    run, _ = session()
    run.emit = server.enhance_event
    window.enhance_session = run
    thread = PageWorker()
    seen = []
    def respond(*args):
        seen.append(QThread.currentThread() is not app.thread())
        return response([reply(0) | {'corrected_text': '修正文'}])
    server.active, server.pages, server.finishing = True, ['001.png'], False
    server.cancelled = False
    server.failed.clear()
    server.written.clear()
    # Use the real page bridge, but a translator stub for correction.
    with patch.object(type(worker), 'translator', new=property(lambda self: Translator())), \
            patch.object(enhance.EnhanceRequester, 'request_chat_completion', respond):
        thread.start()
        deadline = time.monotonic() + 8
        while thread.isRunning() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.005)
        assert thread.wait(1000)
    try:
        assert seen == [True]
        server.project.update_page_progress('001.png', config.RunStatus.FIN_ALL)
        window.on_pagtrans_finished(0)
        window.imsave_thread.wait_for_saves(None)
        server.finishing = True
        deadline = time.monotonic() + 3
        while server.active and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.005)
        events = [json.loads(line) for line in engine.sink.getvalue().splitlines()]
        stages = [i for i, e in enumerate(events) if e.get('stage') == 'enhance']
        done = [i for i, e in enumerate(events) if e.get('event') == 'page_done']
        finished = [i for i, e in enumerate(events) if e.get('event') == 'run_finished']
        assert len(done) == 1 and stages[0] < done[0] < finished[0]
        assert events[stages[0]]['total'] == 1
        assert events[finished[0]]['failed_pages'] == []
        assert not any(e.get('event') == 'error' for e in events)
    finally:
        server.active = False
        worker.manga_quality_enabled = False


@pytest.mark.parametrize('outcome', ['success', 'http', 'cancel', 'low-vram'])
def test_pipeline_thread_finishes_pages_after_review(engine, tmp_path, outcome):
    """Real pipeline counters/signals and saver; only inference/request IO stubbed."""
    from contextlib import ExitStack
    from PIL import Image
    from ballontranslator.ui import module_manager
    app, window, server = engine.app, engine.window, engine.server
    data = page(unreadable=True)
    names = ['001.png', '002.png']
    for name in names:
        Image.fromarray(data[0]).save(tmp_path / name)
    (tmp_path / ('imgtrans_' + tmp_path.name + '.json')).write_text(json_dump_nested_obj({
        'pages': {name: data[2] for name in names}, 'current_img': names[0], 'image_info': {}}))
    server.open_project({'path': str(tmp_path)})
    engine.sink.seek(0)
    engine.sink.truncate()
    worker = window.module_manager.imgtrans_thread
    class PipelineTranslator(Translator):
        low_vram_mode = outcome == 'low-vram'
        def is_computational_intensive(self):
            return False
        def translate_textblk_lst(self, blocks, **kwargs):
            super().translate_textblk_lst(blocks)
        def on_page_translation_finished(self, *args):
            pass
    detector = SimpleNamespace(detect=lambda *args: (data[1][..., 3].copy(), copy.deepcopy(data[2])))
    ocr = SimpleNamespace(run_ocr=lambda image, blocks: blocks)
    translator = PipelineTranslator()
    started, release = threading.Event(), threading.Event()
    requests = []
    def respond(self, profile, args):
        requests.append(args)
        started.set()
        if outcome == 'cancel':
            release.wait(3)
        if outcome == 'http':
            raise RuntimeError('HTTP failure')
        return response([reply(0) | {'corrected_text': '修正文'}])
    profile = default_profile('LM Studio')
    profile.vision_probe_result = True
    def start_pipeline(**kwargs):
        window.module_manager.last_finished_index = -1
        worker.runImgtransPipeline(server.project, kwargs['pages_to_process'])
    with ExitStack() as stack:
        for key, value in dict(enable_detect=True, enable_ocr=True, enable_translate=True,
                enable_inpaint=True, translate_source='日本語', keep_exist_textlines=False,
                translator='google', enhance_enabled=True, enhance_profile_id=profile.id,
                llm_profiles=[profile]).items():
            stack.enter_context(patch.object(config.pcfg.module, key, value))
        stack.enter_context(patch.object(config.pcfg, 'restore_ocr_empty', False))
        for name, value in [('textdetector', detector), ('ocr', ocr),
                            ('inpainter', None), ('translator', translator)]:
            stack.enter_context(patch.object(type(worker), name,
                new=property(lambda self, v=value: v)))
        stack.enter_context(patch.object(window.module_manager.translate_thread, 'module', translator))
        stack.enter_context(patch.object(window.module_manager.translate_thread, 'translator', translator))
        stack.enter_context(patch.object(window, 'on_run_imgtrans', side_effect=start_pipeline))
        stack.enter_context(patch.object(quality, 'enhance_detection',
            side_effect=lambda image, mask, blocks, *args: (mask, blocks)))
        stack.enter_context(patch.object(module_manager, 'unload_modules'))
        stack.enter_context(patch.object(enhance.EnhanceRequester, 'request_chat_completion', respond))
        try:
            assert server.run({}) == {'started': True}
            deadline = time.monotonic() + 8
            cancelled = False
            while (server.active or worker.isRunning()) and time.monotonic() < deadline:
                app.processEvents()
                if outcome == 'cancel' and started.is_set() and not cancelled:
                    server.cancel({})
                    cancelled = True
                time.sleep(.005)
            assert worker.wait(1000) and not server.active
            events = [json.loads(line) for line in engine.sink.getvalue().splitlines()]
            assert not any(e.get('event') == 'error' for e in events), json.dumps([e for e in events if e.get('event') == 'error'])
            done = [e for e in events if e.get('event') == 'page_done']
            finished = [e for e in events if e.get('event') == 'run_finished']
            assert len(finished) == 1 and finished[0]['failed_pages'] == []
            if outcome == 'cancel':
                assert finished[0]['cancelled'] and not done
            else:
                assert not finished[0]['cancelled']
                assert [e['page'] for e in done] == names
                assert len(requests) == 2
                assert events[-1]['event'] == 'run_finished'
                if outcome == 'http':
                    assert all(not b.manga_enhanced for blocks in server.project.pages.values() for b in blocks)
                else:
                    assert all(blocks[0].manga_enhanced for blocks in server.project.pages.values())
        finally:
            release.set()
            worker.requestStop()
            worker.wait(2000)
            server.active = False
            app.processEvents()


@pytest.mark.parametrize('flag', ['manga_complex_background', 'manga_residue_retried',
                                  'manga_outline_applied', 'manga_backing_applied', 'confidence'])
def test_drawn_suspicious_flags(flag):
    block = TextBlock()
    if flag == 'confidence':
        block.manga_line_confidence = [1., quality.MANGA_RETRY_CONFIDENCE - .01]
    else:
        setattr(block, flag, True)
    assert enhance.suspicious_blocks([TextBlock(), block]) == [1]


def test_failed_repair_restores_source_and_other_correction_survives():
    image, draft, blocks, _ = page(unreadable=True)
    blocks[1].text = ['']
    normal = quality.compose_page(image, draft, blocks)
    run, _ = session()
    class MixedTranslator(Translator):
        def translate_textblk_lst(self, blocks):
            for block in blocks:
                block.translation = '超长译文。' * 100 if block.get_text() == '長文' else '修正'
    with patch.object(enhance.EnhanceRequester, 'request_chat_completion', return_value=response([
            reply(0) | {'corrected_text': '長文'}, reply(1) | {'corrected_text': '短文'}])):
        _, output = process(run, (image, draft, blocks, normal), translator=MixedTranslator())
    assert blocks[0].manga_review == 'overflow' and not blocks[0].manga_enhanced
    assert blocks[1].manga_review is None and blocks[1].manga_enhanced
    assert np.array_equal(output[1][:, 120:], image[:, 120:])
    assert np.any(output[1][:, :100] != image[:, :100])


def test_residue_check_failure_withholds_original():
    data = page(unreadable=True)
    run, _ = session()
    with patch.object(enhance.EnhanceRequester, 'request_chat_completion', return_value=response([
            reply(0) | {'corrected_text': '修正文', 'residue': True}])), \
            patch.object(quality, '_residue_found', return_value=True):
        _, output = process(run, data)
    assert data[2][0].manga_review == 'geometry' and not data[2][0].manga_enhanced
    assert np.array_equal(output[1], data[3][1])


def test_manual_edits_can_release_non_dialogue(engine):
    block = TextBlock(text=['原文'], manga_non_dialogue=True)
    engine.server.apply_edit(block, {'source_text': '修正'})
    assert not block.manga_non_dialogue and quality.source_review(block) is None
    block.manga_non_dialogue = True
    engine.server.apply_edit(block, {'keep_original': True})
    assert quality.source_review(block) == 'protected'
    engine.server.apply_edit(block, {'keep_original': False})
    assert quality.source_review(block) is None


def test_failed_reerase_of_drawn_block_restores_original_and_withholds():
    data = page()
    data[2][0].manga_complex_background = True
    run, _ = session()
    with patch.object(enhance.EnhanceRequester, 'request_chat_completion',
                      return_value=response([reply(0) | {'residue': True}])), \
            patch.object(quality, '_residue_found', return_value=True):
        _, output = process(run, data)
    assert data[2][0].manga_review == 'geometry' and data[2][0].manga_erase_failed
    assert data[2][0].manga_enhanced
    assert np.array_equal(output[1][:, 120:], data[0][:, 120:])
    assert np.array_equal(output[1][:, :100], data[3][1][:, :100])
    assert data[2][1].manga_review is None
