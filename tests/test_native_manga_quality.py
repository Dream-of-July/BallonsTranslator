import os
from collections.abc import Iterator
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

if TYPE_CHECKING:
    from qtpy.QtWidgets import QApplication

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


@pytest.fixture(scope='module', autouse=True)
def application() -> Iterator['QApplication']:
    from qtpy.QtWidgets import QApplication
    from ballontranslator.utils import shared

    app = QApplication.instance() or QApplication([])
    previous_dpi = shared.LDPI
    # Like engine bootstrap, match point/pixel conversion to the actual screen.
    # A shared QApplication has already fixed Qt's DPI for this process.
    shared.LDPI = app.primaryScreen().logicalDotsPerInch()
    try:
        yield app
    finally:
        shared.LDPI = previous_dpi


def comic_page():
    from ballontranslator.utils.textblock import TextBlock

    image = np.full((180, 240, 3), (125, 92, 45), dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    lines = []
    for x in (65, 157):
        cv2.ellipse(image, (x, 70), (38, 44), 0, 0, 360, (12, 15, 10), -1)
        cv2.ellipse(image, (x, 70), (36, 42), 0, 0, 360, (248, 250, 252), -1)
        cv2.rectangle(image, (x - 4, 49), (x + 4, 89), (15, 22, 38), -1)
        cv2.rectangle(mask, (x - 5, 48), (x + 5, 90), 255, -1)
        lines.append([[x - 6, 47], [x + 6, 47], [x + 6, 91], [x - 6, 91]])
    cv2.line(mask, (18, 24), (204, 24), 255, 2)  # detector false positive on art
    merged = TextBlock(xyxy=[59, 47, 163, 91], lines=lines, language='ja',
                       src_is_vertical=True, _detected_font_size=12)
    merged.vertical = True
    return image, mask, merged


def test_split_adjacent_speech_and_protect_unreadable_text():
    from ballontranslator.engine.manga_quality import (
        identify_regions, prepare_background, compose_page,
    )

    image, mask, merged = comic_page()
    blocks = identify_regions(image, [merged])
    assert len(blocks) == 2
    right, left = blocks
    assert right.xyxy[0] > left.xyxy[2]
    right.text, right.translation = ['原文'], '中文'
    left.text, left.translation = [''], '猜测'
    draft = prepare_background(image, mask, blocks, None)
    cleaned, result, applied = compose_page(image, draft, blocks)
    assert np.any(applied[:, 135:180])
    assert not np.any(applied[:, :95])
    assert np.array_equal(cleaned[applied == 0], image[applied == 0])
    assert np.array_equal(cleaned[20:30], image[20:30])
    assert np.array_equal(result[:, :95], image[:, :95])
    assert np.any(result != cleaned)
    assert left.manga_review == 'unreadable'


def test_overflow_keeps_original_pixels_and_translation_draft():
    from ballontranslator.engine.manga_quality import (
        identify_regions, prepare_background, compose_page,
    )

    image, mask, merged = comic_page()
    block = identify_regions(image, [merged])[0]
    block.text = ['文字']
    block.translation = '這是無法放進這個小氣泡的長譯文。' * 25
    draft = prepare_background(image, mask, [block], None)
    cleaned, result, applied = compose_page(image, draft, [block])
    assert np.array_equal(cleaned, image)
    assert np.array_equal(result, image)
    assert not np.any(applied)
    assert block.translation.startswith('這是')
    assert block.manga_review == 'overflow'


@pytest.mark.parametrize('source', ['…', '……', '!?', ' — -- ♡ ★ \t', ' \t\n'])
@pytest.mark.parametrize('keep_original', [False, True])
def test_symbols_only_source_keeps_original_and_is_not_a_review_item(source, keep_original):
    from ballontranslator.engine.manga_quality import identify_regions, prepare_background, compose_page

    image, mask, merged = comic_page()
    block = identify_regions(image, [merged])[0]
    block.text, block.translation = [source], source
    block.manga_keep_original = keep_original
    draft = prepare_background(image, mask, [block], None)
    cleaned, result, applied = compose_page(image, draft, [block])
    assert block.manga_review == ('protected' if keep_original else 'skipped')
    assert np.array_equal(cleaned, image) and np.array_equal(result, image)
    assert not applied.any()


def test_symbols_only_source_ink_is_preserved_when_another_block_overlaps():
    import copy
    from ballontranslator.engine.manga_quality import identify_regions, prepare_background, compose_page

    image, mask, merged = comic_page()
    block = identify_regions(image, [merged])[0]
    other = copy.deepcopy(block)
    block.text, block.translation = ['…'], '…'
    other.text, other.translation = ['原文'], '中文'
    draft = prepare_background(image, mask, [block, other], None)
    cleaned, result, applied = compose_page(image, draft, [block, other])
    assert block.manga_review == 'skipped'
    assert other.manga_review == 'geometry'
    assert np.array_equal(cleaned, image) and np.array_equal(result, image)
    assert not applied.any()


@pytest.mark.parametrize('source', ['', 'あ…', 'カ!?', '漢字…', 'A—', '1!', '１?', 'é…'])
def test_empty_ocr_or_source_with_letters_and_digits_still_needs_review(source):
    from ballontranslator.engine.manga_quality import identify_regions, prepare_background, compose_page

    image, mask, merged = comic_page()
    block = identify_regions(image, [merged])[0]
    block.text, block.translation = [source], source
    draft = prepare_background(image, mask, [block], None)
    compose_page(image, draft, [block])
    assert block.manga_review == ('unreadable' if not source else 'untranslated')


def test_user_protected_block_precedes_unreadable_review_and_can_be_released():
    from ballontranslator.engine.manga_quality import identify_regions, prepare_background, compose_page

    image, mask, merged = comic_page()
    block = identify_regions(image, [merged])[0]
    block.text, block.translation = [''], '中文'
    draft = prepare_background(image, mask, [block], None)
    block.manga_keep_original = True
    cleaned, result, applied = compose_page(image, draft, [block])
    assert block.manga_review == 'protected'
    assert np.array_equal(cleaned, image)
    assert np.array_equal(result, image)
    assert not np.any(applied)
    block.manga_keep_original = False
    compose_page(image, draft, [block])
    assert block.manga_review == 'unreadable'


def test_two_existing_blocks_in_joined_outline_keep_distinct_layout_areas():
    from ballontranslator.engine.manga_quality import identify_regions
    from ballontranslator.utils.textblock import TextBlock

    image, _, merged = comic_page()
    cv2.rectangle(image, (96, 64), (127, 76), (248, 250, 252), -1)
    blocks = []
    for line in merged.lines:
        points = np.asarray(line, dtype=np.int32)
        xyxy = [*points.min(axis=0), *points.max(axis=0)]
        blocks.append(TextBlock(xyxy=xyxy, lines=[line]))
    left, right = identify_regions(image, blocks)
    left_area, right_area = (np.asarray(blk.manga_area, dtype=np.int32) for blk in (left, right))
    assert cv2.pointPolygonTest(left_area, (65, 70), False) == 1
    assert cv2.pointPolygonTest(right_area, (157, 70), False) == 1
    assert cv2.pointPolygonTest(left_area, (157, 70), False) == -1
    assert cv2.pointPolygonTest(right_area, (65, 70), False) == -1


def test_open_white_bubble_erases_the_unclipped_stroke_and_keeps_the_panel_line():
    from ballontranslator.engine.manga_quality import compose_page, prepare_background
    from ballontranslator.utils.textblock import TextBlock

    image = np.full((120, 240, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (50, 44), (61, 65), (5, 5, 5), -1)
    cv2.rectangle(image, (97, 43), (102, 66), (5, 5, 5), -1)
    cv2.line(image, (205, 15), (205, 100), (8, 8, 8), 2)
    detector = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.rectangle(detector, (58, 47), (61, 63), 255, -1)
    cv2.rectangle(detector, (97, 47), (102, 63), 255, -1)
    area = [[58, 47], [196, 47], [196, 63], [58, 63]]
    block = TextBlock(xyxy=[58, 47, 196, 63], lines=[area], manga_area=area,
                      _detected_font_size=16, text=['日本語'], translation='中文')

    draft = prepare_background(image, detector, [block], None)
    cleaned, result, mask = compose_page(image, draft, [block])
    assert block.manga_review is None
    assert np.any(result != cleaned)
    assert np.all(cleaned[44:66, 50:62] == 255)
    assert np.all(cleaned[43:67, 97:103] == 255)
    assert np.array_equal(cleaned[:, 203:208], image[:, 203:208])
    assert np.array_equal(cleaned[mask == 0], image[mask == 0])


@pytest.mark.parametrize(
    ('family', 'weight', 'vertical'), [('Songti SC', 400, False), ('PingFang SC', 600, True)],
)
def test_local_font_match_uses_original_glyphs_and_preserves_a_manual_edit(family, weight, vertical):
    from qtpy.QtCore import QPointF
    from qtpy.QtGui import QFont, QFontDatabase, QImage, QPainter

    from ballontranslator.engine.manga_font import FontMatcher
    from ballontranslator.engine.manga_quality import compose_page, prepare_background
    from ballontranslator.utils.textblock import TextBlock

    if family not in QFontDatabase.families():
        pytest.skip(f'{family} is a macOS font')
    font = QFont(family)
    font.setWeight(QFont.Weight(weight))
    font.setPixelSize(48)
    pixels = QImage(256, 256, QImage.Format.Format_Grayscale8)
    pixels.fill(255)
    text = '海風吹葉'
    painter = QPainter(pixels)
    painter.setFont(font)
    if vertical:
        for i, char in enumerate(text):
            painter.drawText(QPointF(26, 52 + i * 48), char)
    else:
        painter.drawText(QPointF(18, 56), text)
    painter.end()
    bits = pixels.constBits()
    if hasattr(bits, 'setsize'):
        bits.setsize(pixels.sizeInBytes())
    gray = np.frombuffer(bits, np.uint8).reshape(256, pixels.bytesPerLine())[:, :256].copy()
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mask = np.where(gray < 220, np.uint8(255), np.uint8(0))
    x1, y1, x2, y2 = (20, 5, 83, 213) if vertical else (11, 12, 216, 68)
    block = TextBlock(xyxy=[x1, y1, x2, y2], lines=[[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]],
                      text=[text], translation='海風吹叶', manga_area=[[7, 3], [240, 3], [240, 242], [7, 242]],
                      src_is_vertical=vertical, _detected_font_size=44)
    block.vertical = vertical
    matcher = FontMatcher()
    assert matcher.rank(image, mask, block)[0] == (family, weight)
    translation = block.translation
    block.translation = '海風吹叶🧬'
    assert matcher.rank(image, mask, block) == []
    block.translation = translation
    draft = prepare_background(image, mask, [block], None)
    cleaned, result, applied = compose_page(image, draft, [block])
    assert block.manga_review is None
    assert block.font_family == family and block.fontformat.font_weight == weight
    assert np.any(result != cleaned)
    assert np.array_equal(cleaned[applied == 0], image[applied == 0])

    block.font_family = 'Hiragino Sans GB'
    block.fontformat.font_weight = 400
    compose_page(image, draft, [block], refit={0})
    assert block.font_family == 'Hiragino Sans GB'
    assert block.fontformat.font_weight == 400


def test_confidence_fields_default_and_project_json_round_trip(tmp_path):
    import json
    from ballontranslator.utils.proj_imgtrans import ProjImgTrans, TextBlkEncoder
    from ballontranslator.utils.textblock import TextBlock

    cv2.imwrite(str(tmp_path / 'page.png'), np.full((48, 48, 3), 255, np.uint8))
    project = ProjImgTrans(str(tmp_path))
    legacy = {'xyxy': [0, 0, 40, 40], 'text': ['source'], 'manga_ocr_confidence': .1}
    current = dict(legacy, manga_line_confidence=[.85, .1], manga_line_text=['source', ''],
                   manga_partial=True)
    project.load_from_dict({'pages': {'page.png': [legacy, current]}})
    old, new = project.pages['page.png']
    assert old.manga_line_confidence == []
    assert old.manga_line_text == []
    assert old.manga_partial is False
    assert old.manga_ocr_confidence == .1
    assert old.manga_complex_background is False
    assert old.manga_residue_retried is False
    assert old.manga_erase_failed is False
    assert old.manga_contrast_before is None and old.manga_contrast_after is None
    assert old.manga_background_luminance is None and old.manga_background_spread is None
    assert not old.manga_outline_applied and not old.manga_backing_applied
    new.manga_contrast_after = 7.2
    new.manga_outline_applied = True
    new.manga_backing_applied = True
    old.manga_line_confidence.append(.9)
    assert TextBlock().manga_line_confidence == []
    payload = json.loads(json.dumps(project.to_dict(), cls=TextBlkEncoder))
    project.load_from_dict(payload)
    loaded = project.pages['page.png'][1]
    assert loaded.manga_line_confidence == new.manga_line_confidence
    assert loaded.manga_line_text == new.manga_line_text
    assert loaded.manga_partial is True
    assert loaded.manga_contrast_after == 7.2
    assert loaded.manga_outline_applied and loaded.manga_backing_applied


def test_retry_merges_better_lines_without_losing_good_readings():
    from ballontranslator.engine.manga_quality import retry_uncertain_ocr, source_review
    from ballontranslator.utils.textblock import TextBlock

    lines = [[[10, 10], [20, 10], [20, 40], [10, 40]],
             [[30, 10], [40, 10], [40, 40], [30, 40]],
             [[50, 10], [60, 10], [60, 40], [50, 40]]]
    block = TextBlock(xyxy=[10, 10, 60, 40], lines=lines, manga_area=lines[0],
                      text='甲丙', manga_line_text=['甲', '', '丙'],
                      manga_line_confidence=[.95, .4, .75], manga_partial=True)

    class RetryOCR:
        def run_ocr(self, image, blocks):
            retried = blocks[0]
            assert retried.manga_line_confidence == []
            retried.manga_line_confidence = [.8, .85, .72]
            retried.manga_line_text = ['wrong', '乙', 'worse']
            retried.text = 'wrong乙worse'
            return blocks

    retry_uncertain_ocr(np.full((100, 100, 3), 255, np.uint8), [block], RetryOCR())
    assert block.text == ['甲', '乙', '丙']
    assert block.manga_line_confidence == [.95, .85, .75]
    assert block.manga_partial is False
    assert block.lines == lines
    assert source_review(block) is None


def test_retry_does_not_replace_an_accepted_line_with_empty_or_worse_text():
    from ballontranslator.engine.manga_quality import retry_uncertain_ocr
    from ballontranslator.utils.textblock import TextBlock

    line = [[10, 10], [20, 10], [20, 40], [10, 40]]
    block = TextBlock(lines=[line, line], manga_area=line, text=['甲'],
                      manga_line_text=['甲', ''], manga_line_confidence=[.8, .1], manga_partial=True)

    class RetryOCR:
        def run_ocr(self, image, blocks):
            retried = blocks[0]
            retried.manga_line_confidence = [.99, .2]
            retried.manga_line_text = ['', '']
            return blocks

    retry_uncertain_ocr(np.full((60, 60, 3), 255, np.uint8), [block], RetryOCR())
    assert block.text == ['甲']
    assert block.manga_line_confidence == [.8, .1]
    assert block.manga_partial is True


def box_block(bounds, translation, size=24, vertical=True, source_bounds=None):
    from ballontranslator.utils.textblock import TextBlock

    x1, y1, x2, y2 = bounds
    area = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    u1, v1, u2, v2 = source_bounds or bounds
    line = [[u1, v1], [u2, v1], [u2, v2], [u1, v2]]
    block = TextBlock(xyxy=[u1, v1, u2, v2], lines=[line], manga_area=area,
                      text=['原文'], translation=translation, src_is_vertical=vertical,
                      _detected_font_size=size)
    block.font_size = size
    block.vertical = vertical
    return block


def test_vertical_balances_real_columns_and_preserves_translation():
    from ballontranslator.engine.manga_quality import _fit, _layout_runs, _balanced_breaks, _NO_LINE_START
    from ballontranslator.ui.text_engine.item import TextBlkItem
    from ballontranslator.ui.text_engine.rendering.indexing import _grapheme_count
    from qtpy.QtWidgets import QGraphicsScene

    text = '天地玄黃宇宙，洪荒日月盈昃辰宿列張寒來暑往秋收冬藏'
    block = box_block((10, 10, 85, 200), text)
    rendered = _fit(block, np.asarray(block.manga_area, np.int32), True)
    assert rendered is not None
    scene = QGraphicsScene()
    item = TextBlkItem(block, 0)
    scene.addItem(item)
    runs = _layout_runs(item)
    lengths = [_grapheme_count(run) for run in runs]
    assert len(runs) > 1
    assert all(run[0] not in _NO_LINE_START for run in runs[1:])
    assert lengths[-1] != 1
    assert max(lengths) - min(lengths) <= 2
    assert _balanced_breaks(runs)
    assert ''.join(runs) == text
    assert block.translation == text and '\n' not in block.translation
    scene.clear()


@pytest.mark.parametrize('runs', [
    ['天地', '，玄'], ['天地「', '玄黃宇'], ['天地…', '…玄黃'],
    ['天地—', '—玄黃'], ['天地', 'ゃ玄'], ['天地', 'ー玄'], ['天地玄', '黃'],
])
def test_rejects_prohibited_breaks(runs):
    from ballontranslator.engine.manga_quality import _balanced_breaks

    assert not _balanced_breaks(runs)


def test_search_at_old_minimum_fits_without_any_ink_outside_area():
    import copy
    from ballontranslator.engine.manga_quality import _fit, _glyph_silhouette, _area_mask, _bounds

    block = box_block((10, 10, 75, 130), ('天地玄黃宇宙洪荒日月盈昃辰宿列張' * 4)[:35], size=36)
    # The previous smallest candidate used the full box and 48% size.
    old = copy.deepcopy(block)
    old.font_size *= .48
    old._bounding_rect = [13, 13, 60, 115]
    pixels, _, _ = _glyph_silhouette(old)
    assert pixels.shape[1] > 61 or pixels.shape[0] > 116
    points = np.asarray(block.manga_area, np.int32)
    pixels, x, y = _fit(block, points, True)
    assert 9 <= block.font_size <= 36 * .48
    assert block.letter_spacing == 1.0
    area = _area_mask(points, _bounds(points))
    h, w = pixels.shape[:2]
    assert not np.any((pixels[..., 3] != 0) & (area[y - 10:y - 10 + h, x - 10:x - 10 + w] == 0))


def test_readable_floor_fallback_fits_large_source_font_deterministically() -> None:
    import copy
    from ballontranslator.engine.manga_quality import _fit, _area_mask, _bounds

    block = box_block((10, 10, 75, 130), ('天地玄黃宇宙洪荒日月盈昃辰宿列張' * 4)[:35], size=72)
    repeated = copy.deepcopy(block)
    points = np.asarray(block.manga_area, np.int32)
    rendered = _fit(block, points, True)
    again = _fit(repeated, points, True)
    assert rendered is not None and again is not None
    assert 9 <= block.font_size < 72 * .40
    assert block.letter_spacing == 1.0
    assert block.fontformat == repeated.fontformat
    assert block.bounding_rect() == repeated.bounding_rect()
    pixels, x, y = rendered
    assert np.array_equal(pixels, again[0]) and (x, y) == again[1:]
    assert np.any(pixels[..., 3])
    x1, y1, x2, y2 = _bounds(points)
    h, w = pixels.shape[:2]
    assert x1 <= x and y1 <= y and x + w <= x2 and y + h <= y2
    area = _area_mask(points, (x1, y1, x2, y2), inset=2)
    assert not np.any((pixels[..., 3] != 0) & (area[y - y1:y - y1 + h, x - x1:x - x1 + w] == 0))


def test_readable_floor_withholds_and_keeps_original_pixels():
    from ballontranslator.engine.manga_quality import prepare_background, compose_page

    image = np.full((100, 100, 3), 250, np.uint8)
    mask = np.zeros((100, 100), np.uint8)
    cv2.rectangle(mask, (25, 25), (35, 60), 255, -1)
    image[mask != 0] = 15
    block = box_block((20, 20, 60, 80), '天地玄黃宇宙洪荒' * 20, size=12)
    draft = prepare_background(image, mask, [block], None)
    cleaned, result, applied = compose_page(image, draft, [block])
    assert block.manga_review == 'overflow'
    assert block.font_size == 12
    assert np.array_equal(cleaned, image) and np.array_equal(result, image)
    assert not applied.any()


def test_collision_retries_once_and_repositions_inside_remaining_area(monkeypatch):
    import ballontranslator.engine.manga_quality as quality

    image = np.full((180, 280, 3), 250, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    blocks = [box_block((10, 10, 200, 160), '天地玄黃宇宙洪荒日月盈昃辰宿', source_bounds=(20, 35, 26, 100)),
              box_block((30, 10, 220, 160), '天地玄黃宇宙洪荒日月盈昃辰宿', source_bounds=(208, 35, 214, 100))]
    for block in blocks:
        x1, y1, x2, y2 = block.xyxy
        mask[y1:y2, x1:x2] = 255
    image[mask != 0] = 15
    calls = []
    original_fit = quality._fit

    def observed_fit(*args, **kwargs):
        rendered = original_fit(*args, **kwargs)
        calls.append((args[4] if len(args) > 4 else None, rendered))
        return rendered

    monkeypatch.setattr(quality, '_fit', observed_fit)
    draft = quality.prepare_background(image, mask, blocks, None)
    cleaned, result, applied = quality.compose_page(image, draft, blocks)
    assert all(block.manga_review is None for block in blocks)
    retries = [(available, render) for available, render in calls if available is not None]
    assert len(retries) == 1 and retries[0][1] is not None
    available, (pixels, x, y) = retries[0]
    h, w = pixels.shape[:2]
    assert not np.any((pixels[..., 3] != 0) & ~available[y - 10:y - 10 + h, x - 30:x - 30 + w])
    assert applied.any() and np.any(result != cleaned)


def test_collision_with_protected_ink_withholds_after_one_retry(monkeypatch):
    import ballontranslator.engine.manga_quality as quality

    image = np.full((150, 150, 3), 80, np.uint8)
    valid = np.zeros(image.shape[:2], np.uint8)
    valid[10:140, 10:140] = 255
    target = box_block((10, 10, 139, 139), '天地玄黃')
    protected = box_block((10, 10, 139, 139), '')
    protected.text = ['@synthetic']
    draft = np.dstack((np.full_like(image, 250), valid))
    calls = []
    original_fit = quality._fit

    def observed_fit(*args, **kwargs):
        calls.append(len(args) > 4)
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(quality, '_fit', observed_fit)
    cleaned, result, applied = quality.compose_page(image, draft, [target, protected])
    assert calls == [False, True]
    assert target.manga_review == 'geometry'
    assert protected.manga_review == 'protected'
    assert not applied.any()
    assert np.array_equal(cleaned, image) and np.array_equal(result, image)


def test_single_line_strip_only_scales_and_never_wraps():
    from ballontranslator.engine.manga_quality import _fit, _layout_runs
    from ballontranslator.ui.text_engine.item import TextBlkItem
    from qtpy.QtWidgets import QGraphicsScene

    text = '天地玄黃宇宙洪荒日月盈昃辰宿列張寒來'
    block = box_block((10, 10, 310, 45), text, size=28, vertical=False)
    spacing = (block.letter_spacing, block.line_spacing)
    assert _fit(block, np.asarray(block.manga_area, np.int32), True) is not None
    scene = QGraphicsScene()
    item = TextBlkItem(block, 0)
    scene.addItem(item)
    assert _layout_runs(item) == [text]
    assert block.font_size < 28
    assert (block.letter_spacing, block.line_spacing) == spacing
    scene.clear()


def test_horizontal_rows_are_balanced_without_prohibited_start():
    from ballontranslator.engine.manga_quality import _fit, _layout_runs, _balanced_breaks
    from ballontranslator.ui.text_engine.item import TextBlkItem
    from qtpy.QtWidgets import QGraphicsScene

    block = box_block((10, 10, 130, 110), '天地玄黃宇宙，洪荒日月盈昃辰宿列張寒來暑往', vertical=False)
    assert _fit(block, np.asarray(block.manga_area, np.int32), True) is not None
    scene = QGraphicsScene()
    item = TextBlkItem(block, 0)
    scene.addItem(item)
    runs = _layout_runs(item)
    assert len(runs) > 1 and _balanced_breaks(runs)
    assert ''.join(runs) == block.translation
    scene.clear()


def test_page_font_consensus_uses_majority_family_and_median_weight(monkeypatch):
    from ballontranslator.engine.manga_font import FontMatcher
    from ballontranslator.engine.manga_quality import compose_page

    image = np.full((160, 500, 3), 250, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    blocks = [box_block((10 + i * 160, 10, 150 + i * 160, 149), '天地玄黃', size=24,
                        source_bounds=(40 + i * 160, 35, 55 + i * 160, 90)) for i in range(3)]
    for block in blocks:
        x1, y1, x2, y2 = block.xyxy
        mask[y1:y2, x1:x2] = 255
    image[mask != 0] = 15
    ranks = {blocks[0].xyxy[0]: [('PingFang SC', 400)],
             blocks[1].xyxy[0]: [('PingFang SC', 800)],
             blocks[2].xyxy[0]: [('Songti SC', 600)]}
    monkeypatch.setattr(FontMatcher, 'rank', lambda self, image, mask, block: ranks[block.xyxy[0]])
    draft = np.dstack((np.full_like(image, 250), mask))
    compose_page(image, draft, blocks)
    assert all(block.manga_review is None for block in blocks)
    assert {(block.font_family, block.fontformat.font_weight) for block in blocks} == {('PingFang SC', 400)}


def test_complex_background_uses_sequential_crops_and_commits_only_accepted_blocks():
    import copy
    from ballontranslator.engine.manga_quality import prepare_background, compose_page
    from ballontranslator.modules.inpaint.base import enlarge_window

    y, x = np.indices((280, 360))
    image = np.stack(((x + y) % 140, (2 * x + y) % 140, (x + 2 * y) % 140), axis=2).astype(np.uint8)
    background = image.copy()
    mask = np.zeros(image.shape[:2], np.uint8)
    blocks = [box_block((25, 35, 110, 120), '天地', size=16),
              box_block((220, 140, 305, 225), '天地玄黃宇宙洪荒' * 60, size=16)]
    for block in blocks:
        x1, y1, x2, y2 = block.xyxy
        mask[y1 + 10:y2 - 10, x1 + 20:x1 + 30] = 255
    image[mask != 0] = 0
    calls = []

    class CropInpainter:
        def all_model_loaded(self):
            return True

        def memory_safe_inpaint(self, crop, ink):
            left, top, right, bottom = crops[len(calls)]
            calls.append(crop.shape[:2])
            # Repair ink correctly, but deliberately corrupt the context:
            # the new residue check must pass without publishing that context.
            repaired = np.full_like(crop, 230)
            repaired[ink != 0] = background[top:bottom, left:right][ink != 0]
            return repaired

    crops = []
    for block in blocks:
        x1, y1, width, height = cv2.boundingRect(np.asarray(block.manga_area, np.int32))
        crops.append(enlarge_window((x1, y1, x1 + width, y1 + height), 360, 280, ratio=1.7))
    outside = np.ones(mask.shape, bool)
    for left, top, right, bottom in crops:
        outside[top:bottom, left:right] = False
    draft = prepare_background(image, mask, blocks, CropInpainter())
    assert len(calls) == 2
    assert all(shape != image.shape[:2] for shape in calls)
    assert np.array_equal(draft[outside, :3], image[outside])
    copies = copy.deepcopy(blocks)
    first = compose_page(image, draft, blocks)
    second = compose_page(image, draft, copies)
    assert blocks[0].manga_review is None and blocks[1].manga_review == 'overflow'
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    cleaned, result, applied = first
    assert np.array_equal(cleaned[outside], image[outside])
    assert np.array_equal(result[outside], image[outside])
    assert np.array_equal(result[140:226, 220:306], image[140:226, 220:306])
    assert np.array_equal(cleaned[applied == 0], image[applied == 0])
    # The same draft remains usable for update_blocks, including a later fit.
    blocks[1].translation = '日月'
    cleaned, result, applied = compose_page(image, draft, blocks, refit={1})
    assert blocks[1].manga_review is None and applied[140:226, 220:306].any()



def test_protected_complex_ink_does_not_load_or_call_inpainter():
    from ballontranslator.engine.manga_quality import prepare_background

    image = np.full((120, 120, 3), 80, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[40:60, 40:60] = 255
    target = box_block((20, 20, 100, 100), '天地')
    protected = box_block((20, 20, 100, 100), '')
    protected.text = ['@synthetic']

    class UnusedInpainter:
        def all_model_loaded(self):
            pytest.fail('Protected ink must not initialize a model')

    draft = prepare_background(image, mask, [target, protected], UnusedInpainter())
    assert not draft[..., 3].any()
    assert np.array_equal(draft[..., :3], image)


def test_second_tier_finds_translucent_bubble_but_rejects_busy_art():
    from ballontranslator.engine.manga_quality import identify_regions, _inside_bubble
    from ballontranslator.utils.textblock import TextBlock

    y, x = np.indices((240, 320))
    art = np.stack((35 + x * .2, 55 + y * .15, 70 + x * .1), axis=2).astype(np.uint8)
    image = art.copy()
    bubble = np.zeros(image.shape[:2], np.uint8)
    cv2.ellipse(bubble, (105, 120), (66, 78), 0, 0, 360, 255, -1)
    image[bubble != 0] = ((art[bubble != 0].astype(float) + 255) * .5).astype(np.uint8)
    cv2.ellipse(image, (105, 120), (66, 78), 0, 0, 360, (15, 15, 15), 3)
    cv2.rectangle(image, (98, 96), (110, 143), (8, 8, 8), -1)
    line = [[92, 90], [116, 90], [116, 149], [92, 149]]
    block = TextBlock(xyxy=[92, 90, 116, 149], lines=[line], _detected_font_size=24)
    found = identify_regions(image, [block])[0]
    assert _inside_bubble(found, image.shape[:2])
    contour = np.asarray(found.manga_area, np.int32)
    assert all(cv2.pointPolygonTest(contour, tuple(map(float, p)), False) >= 0 for p in line)
    assert cv2.pointPolygonTest(contour, (55., 120.), False) >= 0

    busy = np.random.default_rng(7).integers(20, 190, image.shape, dtype=np.uint8)
    direct = TextBlock(xyxy=block.xyxy, lines=[line], _detected_font_size=24)
    assert not _inside_bubble(identify_regions(busy, [direct])[0], busy.shape[:2])


def test_pale_bubble_area_is_unchanged_by_second_tier():
    from ballontranslator.engine.manga_quality import identify_regions, _speech_shapes, _line_label
    from ballontranslator.utils.textblock import TextBlock

    image, _, merged = comic_page()
    labels, contours = _speech_shapes(image)
    for line in merged.lines:
        points = np.asarray(line)
        block = TextBlock(xyxy=[*points.min(axis=0), *points.max(axis=0)], lines=[line])
        expected = contours[_line_label(line, labels) - 1].tolist()
        assert identify_regions(image, [block])[0].manga_area == expected


def outlined_gradient_page():
    y, x = np.indices((180, 260))
    clean = np.stack((70 + x * .25, 100 + y * .25, 120 + x * .1 + y * .1), axis=2).astype(np.uint8)
    image = clean.copy()
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.putText(image, 'TEXT', (38, 106), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 9)
    cv2.putText(image, 'TEXT', (38, 106), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (5, 5, 5), 3)
    cv2.putText(mask, 'TEXT', (38, 106), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 255, 3)
    block = box_block((15, 45, 225, 140), '天地', size=32, vertical=False,
                      source_bounds=(32, 70, 133, 113))
    block.xyxy = [15, 45, 225, 140]
    return image, mask, clean, block


@pytest.mark.parametrize('detected_size', [32, -1])
def test_outlined_text_on_gradient_is_reconstructed_without_residue(detected_size):
    from ballontranslator.engine.manga_quality import prepare_background, compose_page, _residue_found, _grow, _background_model

    image, mask, clean, block = outlined_gradient_page()
    block._detected_font_size = detected_size

    class UnusedInpainter:
        def all_model_loaded(self):
            pytest.fail('A smooth gradient must not load an inpainter')

    draft = prepare_background(image, mask, [block], UnusedInpainter())
    assert block.manga_complex_background and not block.manga_erase_failed
    changed = np.any(image != clean, axis=2)
    assert np.all(draft[changed, 3] != 0)
    error = np.abs(draft[..., :3].astype(int) - clean.astype(int)).max(axis=2)
    assert error[changed].max() <= 3
    assert np.percentile(error[draft[..., 3] != 0], 99) <= 2
    ink = draft[..., 3] != 0
    area = np.zeros(mask.shape, bool)
    area[46:140, 16:225] = True
    ring = _grow(ink, 20) & ~_grow(ink, 4) & area
    model, spread = _background_model(image, ring)
    assert not _residue_found(draft[..., :3], ink, area, 4, model, spread, ring)
    assert np.array_equal(draft[draft[..., 3] == 0, :3], image[draft[..., 3] == 0])
    cleaned, _, applied = compose_page(image, draft, [block])
    assert block.manga_review is None
    assert np.all(applied[changed] != 0)
    assert np.abs(cleaned[changed].astype(int) - clean[changed].astype(int)).max() <= 3


@pytest.mark.parametrize('missing_mask', [False, True])
def test_wide_outline_on_grey_gradient_uses_backing_beyond_tight_line(missing_mask):
    from ballontranslator.engine.manga_quality import prepare_background, compose_page, _erase_radius, _grow

    y, x = np.indices((180, 260))
    clean = np.stack((145 + x * .12, 140 + y * .15, 136 + x * .08 + y * .06), axis=2).astype(np.uint8)
    image = clean.copy()
    source = np.zeros(image.shape[:2], np.uint8)
    source[81:84, 52:76] = 255
    source[82:106, 62:65] = 255
    source[81:106, 88:91] = 255
    source[81:106, 110:113] = 255
    source[92:95, 88:113] = 255
    outline = _grow(source, 13)
    image[outline] = 255
    image[source != 0] = (25, 145, 18)
    block = box_block((48, 81, 123, 108), '天地', size=16, vertical=False)
    block.fg_colors = [25, 145, 18]
    block.bg_colors = [25, 145, 18]  # OCR often reports the fill twice.
    changed = np.any(image != clean, axis=2)
    assert np.any(changed & ~_grow(source, _erase_radius(block, (48, 81, 124, 109)) * 5))

    class UnusedInpainter:
        def all_model_loaded(self):
            pytest.fail('A recoverable gradient must use the surface model')

    mask = np.zeros_like(source) if missing_mask else source
    draft = prepare_background(image, mask, [block], UnusedInpainter())
    assert block.manga_complex_background and not block.manga_erase_failed
    assert np.all(draft[changed, 3] != 0)
    assert np.abs(draft[changed, :3].astype(int) - clean[changed].astype(int)).max() <= 3
    assert draft[draft[..., 3] != 0, :3].max() < 211
    assert np.array_equal(draft[draft[..., 3] == 0, :3], image[draft[..., 3] == 0])
    cleaned, _, applied = compose_page(image, draft, [block])
    assert block.manga_review is None
    assert np.all(applied[changed] != 0)
    assert np.abs(cleaned[changed].astype(int) - clean[changed].astype(int)).max() <= 3


def test_pale_gradient_uses_distant_backing_instead_of_white_outline():
    from ballontranslator.engine.manga_quality import prepare_background, _unframed_white_ink

    y, x = np.indices((140, 180))
    clean = np.stack((223 + x * .1, 220 + y * .1, 218 + x * .07), axis=2).astype(np.uint8)
    image = clean.copy()
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.putText(image, 'HI', (55, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 15)
    cv2.putText(image, 'HI', (55, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (25, 145, 18), 2)
    cv2.putText(mask, 'HI', (55, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    block = box_block((53, 61, 85, 88), '天地', size=16, vertical=False)
    block.fg_colors = block.bg_colors = [25, 145, 18]
    assert _unframed_white_ink(block, image, mask) is None
    draft = prepare_background(image, mask, [block], None)
    assert not block.manga_erase_failed
    changed = np.any(image != clean, axis=2)
    assert np.all(draft[changed, 3] != 0)
    assert np.abs(draft[changed, :3].astype(int) - clean[changed].astype(int)).max() <= 3


def test_residue_allows_continuing_border_but_rejects_attached_glyph_strokes():
    from ballontranslator.engine.manga_quality import _background_model, _residue_found, _grow

    clean = np.full((120, 180, 3), 170, np.uint8)
    cv2.line(clean, (0, 65), (179, 65), (25, 90, 30), 2)
    original = clean.copy()
    cv2.putText(original, 'T', (80, 64), cv2.FONT_HERSHEY_SIMPLEX, .8, (5, 5, 5), 3)
    ink = np.zeros(clean.shape[:2], bool)
    ink[40:78, 70:114] = True
    area = np.ones_like(ink)
    ring = _grow(ink, 16) & ~_grow(ink, 4)
    model, spread = _background_model(original, ring)
    assert not _residue_found(clean, ink, area, 2, model, spread, ring, original=original)
    assert _residue_found(original, ink, area, 2, model, spread, ring, original=original)
    partial = clean.copy()
    partial[44:59, 87:91] = 5
    assert _residue_found(partial, ink, area, 2, model, spread, ring, original=original)


@pytest.mark.parametrize(('low', 'high'), [(65, 145), (0, 200)])
def test_failed_textured_erasure_retries_once_and_withholds_original(low, high):
    from ballontranslator.engine.manga_quality import prepare_background, compose_page

    image, mask, _, block = outlined_gradient_page()
    texture = np.random.default_rng(11).integers(low, high, image.shape, dtype=np.uint8)
    source = (image.min(axis=2) == 5) | (image.min(axis=2) == 255)
    texture[source] = image[source]
    calls = []

    class UnchangedInpainter:
        def all_model_loaded(self):
            return True

        def memory_safe_inpaint(self, crop, ink):
            calls.append(ink.copy())
            return crop

    draft = prepare_background(texture, mask, [block], UnchangedInpainter())
    assert len(calls) == 2
    assert np.count_nonzero(calls[1]) > np.count_nonzero(calls[0])
    assert block.manga_complex_background and block.manga_residue_retried and block.manga_erase_failed
    assert not draft[..., 3].any()
    assert np.array_equal(draft[..., :3], texture)
    cleaned, result, applied = compose_page(texture, draft, [block])
    assert block.manga_review == 'geometry'
    assert np.array_equal(cleaned, texture) and np.array_equal(result, texture)
    assert not applied.any()


@pytest.mark.parametrize('repair_core', [False, True])
def test_partial_repair_does_not_hide_remaining_strokes_on_busy_art(repair_core):
    from ballontranslator.engine.manga_quality import prepare_background, _grow

    image, mask, clean, block = outlined_gradient_page()
    # Construct the halo explicitly: putText thickness handling varies across
    # OpenCV builds, so this rejection case must guarantee real white strokes.
    image = clean.copy()
    image[_grow(mask, 5)] = 255
    image[mask != 0] = 5
    assert np.any(image.min(axis=2) == 255)
    texture = np.random.default_rng(11).integers(0, 200, image.shape, dtype=np.uint8)
    source = (image.min(axis=2) == 5) | (image.min(axis=2) == 255)
    texture[source] = image[source]
    calls = []

    class PartialInpainter:
        def all_model_loaded(self):
            return True

        def memory_safe_inpaint(self, crop, ink):
            calls.append(1)
            repaired = crop.copy()
            if repair_core:
                # Erase every dark core while leaving the white glyph halo.
                repaired[crop.min(axis=2) == 5] = 100
            else:
                selected = np.flatnonzero(ink)
                repaired.reshape(-1, 3)[selected[::100]] = 100
            return repaired

    draft = prepare_background(texture, mask, [block], PartialInpainter())
    assert len(calls) == 2
    assert block.manga_erase_failed and block.manga_residue_retried
    assert not draft[..., 3].any()
    assert np.array_equal(draft[..., :3], texture)


def test_textured_translucent_bubble_keeps_veil_texture_and_border():
    from ballontranslator.engine.manga_quality import prepare_background, _bounds
    from ballontranslator.modules.inpaint.base import enlarge_window

    y, x = np.indices((220, 300))
    art = np.stack((60 + x % 70, 70 + y % 60, 80 + (x + y) % 80), axis=2).astype(np.uint8)
    bubble = np.zeros(art.shape[:2], np.uint8)
    cv2.ellipse(bubble, (150, 110), (85, 80), 0, 0, 360, 255, -1)
    clean = art.copy()
    clean[bubble != 0] = ((art[bubble != 0].astype(float) + 255) * .5).astype(np.uint8)
    cv2.ellipse(clean, (150, 110), (85, 80), 0, 0, 360, (35, 140, 25), 2)
    image = clean.copy()
    source = np.zeros(bubble.shape, np.uint8)
    source[85:130, 133:137] = 255
    source[85:89, 124:146] = 255
    halo = cv2.dilate(source, np.ones((9, 9), np.uint8)) != 0
    image[halo] = 255
    image[source != 0] = (25, 145, 18)
    block = box_block((65, 30, 235, 190), '天地', size=24, source_bounds=(122, 80, 151, 135))
    contour = cv2.findContours(bubble, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0].reshape(-1, 2)
    block.manga_area = contour.tolist()
    block.fg_colors = block.bg_colors = [25, 145, 18]
    left, top, right, bottom = enlarge_window(_bounds(contour), 300, 220, ratio=1.7)
    calls = []

    class VeilInpainter:
        def all_model_loaded(self):
            return True

        def memory_safe_inpaint(self, crop, ink):
            calls.append(1)
            return clean[top:bottom, left:right].copy()

    draft = prepare_background(image, source, [block], VeilInpainter())
    assert calls == [1] and block.manga_complex_background and not block.manga_erase_failed
    assert np.array_equal(draft[..., :3], clean)
    assert np.ptp(draft[draft[..., 3] != 0, :3], axis=0).max() > 20
    assert not draft[bubble == 0, 3].any()


def test_unreadable_outside_bubble_is_skipped_and_inside_is_reviewed():
    from ballontranslator.engine.manga_quality import identify_regions, prepare_background, compose_page

    image, mask, merged = comic_page()
    inside = identify_regions(image, [merged])[0]
    outside = box_block((8, 130, 80, 164), '', size=16)
    outside.text = ['']
    blocks = [inside, outside]
    inside.text = ['']
    draft = prepare_background(image, mask, blocks, None)
    cleaned, result, applied = compose_page(image, draft, blocks)
    assert inside.manga_review == 'unreadable'
    assert outside.manga_review == 'skipped'
    assert np.array_equal(cleaned, image) and np.array_equal(result, image)
    assert not applied.any()


def test_complex_mask_respects_protected_area_and_resets_retry_state():
    from ballontranslator.engine.manga_quality import prepare_background

    image, mask, _, target = outlined_gradient_page()
    protected = box_block((88, 65, 140, 119), '', size=32, vertical=False)
    protected.manga_keep_original = True
    target.manga_residue_retried = target.manga_erase_failed = True
    draft = prepare_background(image, mask, [target, protected], None)
    assert target.manga_complex_background
    assert not target.manga_residue_retried and not target.manga_erase_failed
    assert draft[65:120, 88:141, 3].sum() == 0
    assert np.array_equal(draft[65:120, 88:141, :3], image[65:120, 88:141])
    assert draft[:, :88, 3].any()


def legibility_page(background, fill=(0, 0, 0), bubble=True):
    image = background.copy()
    block = box_block((20, 20, 219, 159), '天地玄黃', size=28,
                      source_bounds=(70, 55, 85, 120), vertical=False)
    block.fontformat.frgb = list(fill)
    if not bubble:
        block.xyxy = [20, 20, 219, 159]
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[55:120, 70:85] = 255
    image[mask != 0] = fill
    return image, np.dstack((background, mask)), block


@pytest.mark.parametrize(('low', 'high', 'fill', 'stroke'), [
    (125, 240, (0, 0, 0), [255, 255, 255]),
    (5, 95, (255, 255, 255), [0, 0, 0]),
])
def test_busy_lettering_keeps_contrasting_source_fill_and_uses_native_stroke(low, high, fill, stroke):
    from ballontranslator.engine.manga_quality import compose_page, _glyph_silhouette

    y, x = np.indices((180, 240))
    grey = np.where((x // 4 + y // 4) % 2, low, high).astype(np.uint8)
    background = np.repeat(grey[..., None], 3, axis=2)
    image, draft, block = legibility_page(background, fill)
    compose_page(image, draft, [block])
    assert block.manga_review is None
    assert block.fontformat.frgb == list(fill)
    assert block.fontformat.srgb == stroke
    assert block.manga_outline_applied and block.stroke_width >= .08
    assert block.stroke_width * block.font_size >= 1.5
    assert block.manga_contrast_after >= 4.5
    # Edits and subsequent re-renders use the same persisted native stroke.
    rendered = _glyph_silhouette(block)
    assert rendered is not None
    assert np.any(np.all(rendered[0][..., :3] == stroke, axis=2) & (rendered[0][..., 3] > 224))


def test_green_source_on_grey_wall_gets_sufficient_contrast():
    from ballontranslator.engine.manga_quality import compose_page

    image, draft, block = legibility_page(np.full((180, 240, 3), 145, np.uint8), (25, 145, 18), False)
    compose_page(image, draft, [block])
    assert block.manga_review is None
    assert block.manga_contrast_before < 4.5
    assert block.fontformat.frgb == [0, 0, 0]
    assert block.manga_outline_applied
    assert block.manga_contrast_after >= 4.5
    assert not block.manga_backing_applied


@pytest.mark.parametrize('colour', [(255, 255, 255), (241, 245, 232)])
def test_flat_pale_bubble_keeps_fill_and_zero_stroke(colour):
    from ballontranslator.engine.manga_quality import compose_page

    image, draft, block = legibility_page(np.full((180, 240, 3), colour, np.uint8), (12, 22, 10))
    compose_page(image, draft, [block])
    assert block.manga_review is None
    assert block.fontformat.frgb == [12, 22, 10]
    assert block.stroke_width == 0
    assert not block.manga_outline_applied and not block.manga_backing_applied


@pytest.mark.parametrize('bubble', [False, True])
def test_backing_and_outlined_glyphs_stay_inside_bubble_and_preserve_original_edges(bubble):
    from ballontranslator.engine.manga_quality import compose_page, _glyph_silhouette, _area_mask, _bounds

    y, x = np.indices((180, 240))
    grey = np.where((x // 3 + y // 3) % 2, 125, 240).astype(np.uint8)
    image, draft, block = legibility_page(np.repeat(grey[..., None], 3, axis=2), bubble=bubble)
    if bubble:
        ellipse = cv2.ellipse2Poly((120, 90), (99, 69), 0, 0, 360, 10)
        block.manga_area = ellipse.tolist()
    points = np.asarray(block.manga_area, np.int32)
    allowed = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(allowed, [points], 255)
    cleaned, result, applied = compose_page(image, draft, [block])
    assert block.manga_review is None and block.manga_outline_applied
    assert block.manga_backing_applied is bubble
    assert np.array_equal(cleaned[allowed == 0], image[allowed == 0])
    assert np.array_equal(result[allowed == 0], image[allowed == 0])
    assert np.array_equal(cleaned[applied == 0], image[applied == 0])
    pixels, px, py = _glyph_silhouette(block)
    h, w = pixels.shape[:2]
    assert not np.any((pixels[..., 3] != 0) & (allowed[py:py + h, px:px + w] == 0))
    if bubble:
        left, top, right, bottom = _bounds(points)
        interior = _area_mask(points, (left, top, right, bottom), inset=2)
        backing = np.any(cleaned[top:bottom, left:right] != draft[top:bottom, left:right, :3], axis=2)
        assert backing.any() and not np.any(backing & (interior == 0))
        # Partial opacity retains texture instead of replacing it with a plate.
        assert np.ptp(cleaned[top:bottom, left:right][backing], axis=0).max() > 5


def test_removed_source_halo_preserves_its_contrasting_colour():
    from ballontranslator.engine.manga_quality import compose_page, _grow

    background = np.full((180, 240, 3), 100, np.uint8)
    image, draft, block = legibility_page(background, (255, 255, 255), False)
    core = draft[..., 3] != 0
    halo = _grow(core, 4) & ~core
    image[halo] = [0, 0, 20]
    draft[halo, 3] = 255
    compose_page(image, draft, [block])
    assert block.fontformat.frgb == [255, 255, 255]
    assert block.manga_outline_applied
    assert block.fontformat.srgb == [0, 0, 20]


def test_unreadable_art_detection_does_not_reserve_neighbouring_white_bubble_ink():
    from ballontranslator.engine.manga_quality import compose_page, prepare_background

    image, mask, merged = comic_page()
    from ballontranslator.engine.manga_quality import identify_regions
    bubble = identify_regions(image, [merged])[0]
    bubble.text, bubble.translation = ['文字'], '中文'
    art = box_block((10, 10, 210, 140), '', size=24)
    art.text = ['']
    blocks = [bubble, art]
    draft = prepare_background(image, mask, blocks, None)
    _, result, applied = compose_page(image, draft, blocks)
    assert art.manga_review == 'skipped'
    assert bubble.manga_review is None
    assert applied.any() and np.any(result != image)
    assert bubble.stroke_width == 0 and not bubble.manga_backing_applied


@pytest.mark.parametrize('connected', [False, True])
def test_textured_repair_distinguishes_coincident_pixels_from_surviving_stroke(connected):
    from ballontranslator.engine.manga_quality import prepare_background, _grow

    rng = np.random.default_rng(19)
    clean = rng.integers(30, 180, (180, 240, 3), dtype=np.uint8)
    image, _, block = legibility_page(clean, (0, 0, 0), False)
    mask = np.zeros(clean.shape[:2], np.uint8)
    mask[55:120, 70:85] = 255
    repaired = clean.copy()
    # Four chance matches are not a stroke. A connected strip still fails.
    dots = [(65, 74), (77, 78), (91, 74), (106, 79)]
    for y, x in dots:
        repaired[y, x] = 0
    if connected:
        repaired[65:115, 76:79] = 0

    class RepairedTexture:
        def all_model_loaded(self):
            return True

        def memory_safe_inpaint(self, crop, ink):
            from ballontranslator.modules.inpaint.base import enlarge_window
            from ballontranslator.engine.manga_quality import _bounds
            x1, y1, x2, y2 = _bounds(np.asarray(block.manga_area, np.int32))
            left, top, right, bottom = enlarge_window((x1, y1, x2, y2), 240, 180, ratio=1.7)
            return repaired[top:bottom, left:right].copy()

    draft = prepare_background(image, mask, [block], RepairedTexture())
    assert block.manga_erase_failed is connected
    assert bool(draft[..., 3].any()) is not connected


def test_antialiased_dark_source_on_white_is_not_mistaken_for_a_halo():
    from ballontranslator.engine.manga_quality import compose_page

    background = np.full((180, 240, 3), 255, np.uint8)
    image, draft, block = legibility_page(background, (5, 8, 3))
    # A wide range of JPEG/antialias grey values belongs to a single fill.
    for offset, grey in enumerate((25, 65, 125, 145, 175, 205, 225, 240)):
        image[55:120, 70 + offset] = grey
    compose_page(image, draft, [block])
    assert block.manga_review is None
    assert block.fontformat.frgb == [5, 8, 3]
    assert block.stroke_width == 0 and not block.manga_backing_applied


def test_placement_prefers_erased_background_to_artwork_without_withholding():
    from ballontranslator.engine.manga_quality import _fit, _area_mask, _bounds

    background = np.full((180, 240, 3), 230, np.uint8)
    background[20:90, 20:220] = [20, 10, 5]  # artwork at the first distance maximum
    erased = np.zeros(background.shape[:2], np.uint8)
    erased[95:145, 80:155] = 255
    block = box_block((20, 20, 219, 159), '天地', size=20,
                      source_bounds=(80, 95, 155, 145), vertical=False)
    points = np.asarray(block.manga_area, np.int32)
    rendered = _fit(block, points, True, background=background, erased=erased)
    assert rendered is not None
    pixels, x, y = rendered
    h, w = pixels.shape[:2]
    visible = pixels[..., 3] != 0
    assert not np.any(visible & (background[y:y + h, x:x + w, 0] == 20))
    assert np.any(visible & (erased[y:y + h, x:x + w] != 0))


def test_native_stroke_remains_collision_geometry():
    from ballontranslator.engine.manga_quality import _fit, _bounds, _area_mask

    block = box_block((20, 20, 219, 159), '天地玄黃', size=28, vertical=False)
    block.fontformat.srgb = [255, 255, 255]
    block.stroke_width = .14
    points = np.asarray(block.manga_area, np.int32)
    left, top, right, bottom = _bounds(points)
    available = np.ones((bottom - top, right - left), bool)
    available[:, 85:100] = False
    rendered = _fit(block, points, True, available=available)
    assert rendered is not None
    pixels, x, y = rendered
    h, w = pixels.shape[:2]
    assert not np.any((pixels[..., 3] > 0) & ~available[y - top:y - top + h, x - left:x - left + w])


def test_small_translucent_glyph_core_gets_contrast_without_overflowing_tight_detection():
    from ballontranslator.engine.manga_quality import _fit, _drawn_contrast

    background = np.full((100, 80, 3), 255, np.uint8)
    block = box_block((20, 20, 30, 78), '天地玄黃宇宙', size=9)
    block.fontformat.frgb = [8, 130, 6]
    erased = np.zeros(background.shape[:2], np.uint8)
    erased[20:79, 20:31] = 255
    rendered = _fit(block, np.asarray(block.manga_area, np.int32), True,
                    background=background, erased=erased)
    assert rendered is not None
    pixels, x, y = rendered
    h, w = pixels.shape[:2]
    assert 20 <= x and 20 <= y and x + w <= 31 and y + h <= 79
    assert _drawn_contrast(block, pixels, background[y:y + h, x:x + w]) >= 4.5
