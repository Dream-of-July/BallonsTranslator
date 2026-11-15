"""Use local text geometry to couple Japanese typesetting and reversible cleanup."""
from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ballontranslator.utils.textblock import TextBlock
from .manga_font import FontChoice, FontMatcher


LOCAL_MODULES = {'textdetector': 'ctd', 'ocr': 'mit48px', 'inpainter': 'lama_large_512px'}
LOCAL_ALTERNATIVES = {
    'textdetector': {'ctbd', 'ysgyolo', 'ysg_ctd'},
    'ocr': {'mit48px_ctc', 'manga_ocr', 'PaddleOCRVLManga'},
    'inpainter': {'opencv-tela', 'patchmatch', 'aot', 'lama_mpe'},
}


def local_module(key: str, selected: str) -> str:
    if selected in LOCAL_ALTERNATIVES.get(key, ()) or key not in LOCAL_MODULES:
        return selected
    return LOCAL_MODULES[key]


def draft_path(directory: str, name: str) -> Path:
    return Path(directory) / '.manga-quality' / f'{name}.png'


def _bounds(points: np.ndarray) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(points)
    return x, y, x + w, y + h


def _speech_shapes(image: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    # White/near-white interiors retain the dark-ink holes. Filling each
    # external contour retains its ownership, including across touching bubbles.
    pale = (image[..., :3].min(axis=2) >= 211).astype(np.uint8)
    # A narrow gap in hand-drawn ink can join a bubble to the page margin.
    pale = cv2.erode(pale, np.ones((3, 3), np.uint8))
    n, components, stats, _ = cv2.connectedComponentsWithStats(pale, connectivity=4)
    h, w = pale.shape
    labels = np.zeros_like(components)
    contours = []
    for i in range(1, n):
        x, y, bw, bh, pixels = stats[i]
        if min(bw, bh) < 17 or max(bw, bh) < 29 or not 320 < pixels < h * w * .12:
            continue
        if pixels < bw * bh * .35 or bw > w * .66 or bh > h * .66:
            continue
        component = (components[y:y + bh, x:x + bw] == i).astype(np.uint8)
        edges, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not edges:
            continue
        contour = max(edges, key=cv2.contourArea)
        contour = cv2.approxPolyDP(contour, 1.3, True).reshape(-1, 2) + (x, y)
        if len(contour) < 3:
            continue
        contour = contour.astype(np.int32)
        contours.append(contour)
        cv2.fillPoly(labels, [contour], len(contours))
    return labels, contours


def _line_label(line: list, labels: np.ndarray) -> int:
    h, w = labels.shape
    points = np.asarray(line, dtype=np.int32)
    x, y = points.mean(axis=0).astype(int)
    if 0 <= x < w and 0 <= y < h and labels[y, x]:
        return int(labels[y, x])
    return 0


def _plain_area(block: TextBlock, width: int, height: int) -> list[list[int]]:
    x1, y1, x2, y2 = block.xyxy
    if x2 <= x1 or y2 <= y1:
        return []
    x1, x2 = max(0, x1), min(width - 1, x2)
    y1, y2 = max(0, y1), min(height - 1, y2)
    return [[int(x1), int(y1)], [int(x2), int(y1)], [int(x2), int(y2)], [int(x1), int(y2)]]


def _divide_shared_shapes(groups: dict[int, list[TextBlock]], contours: list[np.ndarray]) -> None:
    # A joined outline can still contain multiple CTD blocks (two speaking
    # lobes or a stepped caption). Allocate its room to the source anchors.
    for label, blocks in groups.items():
        if len(blocks) < 2:
            continue
        outline = contours[label - 1]
        x1, y1, x2, y2 = bounds = _bounds(outline)
        interior = _area_mask(outline, bounds) != 0
        y, x = np.ogrid[y1:y2, x1:x2]
        closest = np.full(interior.shape, np.inf, dtype=np.float32)
        assignments = np.full(interior.shape, -1, dtype=np.int32)
        for index, block in enumerate(blocks):
            bx1, by1, bx2, by2 = block.xyxy
            dx = np.maximum(0, np.maximum(bx1 - x, x - bx2))
            dy = np.maximum(0, np.maximum(by1 - y, y - by2))
            center = (x - (bx1 + bx2) / 2) ** 2 + (y - (by1 + by2) / 2) ** 2
            score = dx ** 2 + dy ** 2 + .025 * center
            updated = interior & (score < closest)
            closest[updated] = score[updated]
            assignments[updated] = index
        for index, block in enumerate(blocks):
            parts, _ = cv2.findContours((assignments == index).astype(np.uint8),
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not parts:
                block.manga_area = []
                continue
            part = cv2.approxPolyDP(max(parts, key=cv2.contourArea), 1, True).reshape(-1, 2)
            block.manga_area = (part + (x1, y1)).tolist()


def identify_regions(image: np.ndarray, blocks: list[TextBlock]) -> list[TextBlock]:
    """Assign each source line to its own nearby bubble before OCR joins lines."""
    labels, contours = _speech_shapes(image)
    result = []
    owners = {}
    for block in blocks:
        by_shape = {}
        for line in block.lines:
            by_shape.setdefault(_line_label(line, labels), []).append(line)

        # Tiny background details can also be pale. Keep the measured source
        # geometry when an alleged bubble does not contain most of the line.
        shape_groups = []
        for label, lines in by_shape.items():
            if label:
                contour = contours[label - 1]
                accepted = [
                    line for line in lines
                    if sum(cv2.pointPolygonTest(contour, tuple(map(float, pt)), False) >= 0
                           for pt in line) >= 2
                ]
                if accepted:
                    shape_groups.append((label, accepted))
                residue = [line for line in lines if line not in accepted]
                if residue:
                    shape_groups.append((0, residue))
            else:
                shape_groups.append((0, lines))
        groups = []
        for label, lines in shape_groups:
            existing = next((group for key, group in groups if key == label), None)
            if existing is None:
                groups.append((label, list(lines)))
            else:
                existing.extend(lines)
        if len(groups) > 1:
            if block.src_is_vertical or block.vertical:
                groups.sort(key=lambda entry: -np.asarray(entry[1])[..., 0].mean())
            else:
                groups.sort(key=lambda entry: np.asarray(entry[1])[..., 1].mean())
        for label, lines in groups or [(0, block.lines)]:
            item = block
            if len(groups) > 1:
                item = copy.deepcopy(block)
                item.lines = lines
                item.adjust_bbox(with_bbox=False)
                item._bounding_rect = None
            item.manga_area = contours[label - 1].tolist() if label else _plain_area(item, labels.shape[1], labels.shape[0])
            if label:
                owners.setdefault(label, []).append(item)
            result.append(item)
    _divide_shared_shapes(owners, contours)
    return result


def _boxes_overlap(a: list, b: list) -> bool:
    x1, y1, x2, y2 = a
    u1, v1, u2, v2 = b
    intersection = max(0, min(x2, u2) - max(x1, u1)) * max(0, min(y2, v2) - max(y1, v1))
    small = min((x2 - x1) * (y2 - y1), (u2 - u1) * (v2 - v1))
    return small > 0 and intersection / small >= .48


def enhance_detection(
    image: np.ndarray, mask: np.ndarray, blocks: list[TextBlock], detector: object,
    stopped: Callable[[], bool],
) -> tuple[np.ndarray, list[TextBlock]]:
    """Add distinct local detections from overlapping views and missed bubbles."""
    if mask is None:
        return mask, identify_regions(image, blocks)
    mask = mask.copy()
    h, w = mask.shape
    # Additional crops reuse the loaded CTD model. Other detectors may have
    # their own tiling and grouping already.
    detect_detail = getattr(detector, 'detect_detail', None)
    if detect_detail is not None and min(h, w) >= 512:
        xpad, ypad = w // 12, h // 12
        crops = [(x1, y1, x2, y2)
                 for y1, y2 in ((0, h // 2 + ypad), (h // 2 - ypad, h))
                 for x1, x2 in ((0, w // 2 + xpad), (w // 2 - xpad, w))]
        for x1, y1, x2, y2 in crops:
            if stopped():
                break
            crop_mask, found = detect_detail(image[y1:y2, x1:x2])
            for block in found:
                u1, v1, u2, v2 = map(int, block.xyxy)
                # Partial text at seams is covered by the overlapping crop.
                if (u1 <= 2 and x1 or v1 <= 2 and y1
                        or u2 >= x2 - x1 - 2 and x2 < w
                        or v2 >= y2 - y1 - 2 and y2 < h):
                    continue
                rect = [u1 + x1, v1 + y1, u2 + x1, v2 + y1]
                if any(_boxes_overlap(rect, existing.xyxy) for existing in blocks):
                    continue
                block.xyxy = rect
                block.det_model = detector.name
                block._bounding_rect = None
                block.lines = (np.asarray(block.lines, dtype=np.int32) + (x1, y1)).tolist()
                blocks.append(block)
                target = mask[rect[1]:rect[3], rect[0]:rect[2]]
                target |= crop_mask[v1:v2, u1:u2]
    return mask, identify_regions(image, blocks)


def _polygon(block: TextBlock, shape: tuple[int, int]) -> np.ndarray | None:
    try:
        points = np.asarray(block.manga_area)
        if points.ndim != 2 or points.shape[1] != 2 or not 3 <= len(points) <= 1024:
            return None
        if not np.issubdtype(points.dtype, np.number) or not np.isfinite(points).all():
            return None
        if (points < 0).any() or (points >= (shape[1], shape[0])).any():
            return None
        return points.astype(np.int32)
    except (TypeError, ValueError):
        return None


def _area_mask(points: np.ndarray, bounds: tuple[int, int, int, int], inset=0) -> np.ndarray:
    x1, y1, x2, y2 = bounds
    local = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    cv2.fillPoly(local, [(points - (x1, y1)).astype(np.int32)], 255)
    if inset:
        local = cv2.erode(local, np.ones((inset * 2 + 1,) * 2, np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return local


def _line_support(block: TextBlock, bounds: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bounds
    support = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    for line in block.lines:
        geometry = np.asarray(line, dtype=np.int32) - (x1, y1)
        cv2.fillPoly(support, [geometry.astype(np.int32)], 255)
    return cv2.dilate(support, np.ones((9, 9), dtype=np.uint8))


def _region_ink(
    block: TextBlock, mask: np.ndarray, *, inside_bubble: bool = False,
) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    points = _polygon(block, mask.shape)
    if points is None or not block.lines:
        return None
    bounds = _bounds(points)
    x1, y1, x2, y2 = bounds
    area = _area_mask(points, bounds, inset=1)
    support = area if inside_bubble else _line_support(block, bounds)
    ink = cv2.bitwise_and(area, cv2.bitwise_and(support, mask[y1:y2, x1:x2]))
    return bounds, ink != 0


def _unframed_white_ink(
    block: TextBlock, image: np.ndarray, mask: np.ndarray,
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray] | None:
    """Recover complete strokes near a tight detection box on verified white."""
    tight = _polygon(block, mask.shape)
    if tight is None or not block.lines:
        return None
    x1, y1, x2, y2 = _bounds(tight)
    height, width = mask.shape
    padding = min(16, max(10, round(block._detected_font_size * .5)))
    bounds = (max(0, x1 - padding), max(0, y1 - padding),
              min(width, x2 + padding), min(height, y2 + padding))
    left, top, right, bottom = bounds
    crop = image[top:bottom, left:right, :3]
    rim = np.ones(crop.shape[:2], dtype=bool)
    rim[y1 - top:y2 - top, x1 - left:x2 - left] = False
    surrounding = crop[rim]
    pale = surrounding[surrounding.min(axis=1) >= 225]
    if not len(pale) or len(pale) < .72 * len(surrounding):
        return None
    if np.max(np.ptp(np.percentile(pale, (10, 90), axis=0), axis=0)) > 18:
        return None
    seed = (mask[top:bottom, left:right] != 0) & (_line_support(block, bounds) != 0)
    if not seed.any():
        return None
    backing = np.median(pale, axis=0).astype(np.uint8)
    stroke = np.max(np.abs(crop.astype(np.int16) - backing), axis=2) > 10
    count, labels = cv2.connectedComponents(stroke.astype(np.uint8), connectivity=8)
    selected = np.zeros(count, dtype=bool)
    selected[np.unique(labels[seed])] = True
    # A line of artwork that continues beyond this crop is not a text stroke.
    for edge in (labels[0], labels[-1], labels[:, 0], labels[:, -1]):
        selected[edge] = False
    selected[0] = False
    ink = selected[labels]
    if not ink.any():
        return None
    halo = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return bounds, ink | (halo & ~stroke), backing


MANGA_LINE_MIN_CONFIDENCE = .75
MANGA_RETRY_CONFIDENCE = .9


def source_review(block: TextBlock) -> str | None:
    if block.manga_keep_original:
        return 'protected'
    source = block.get_text().strip()
    if not source:
        return 'unreadable'
    if re.fullmatch(r'[@＠][a-zA-Z0-9_.-]{2,}', source):
        return 'protected'
    if block.manga_line_confidence and not any(
            value >= MANGA_LINE_MIN_CONFIDENCE for value in block.manga_line_confidence):
        return 'unreadable'
    return None


def prepare_background(
    image: np.ndarray, mask: np.ndarray, blocks: list[TextBlock], inpainter: object,
) -> np.ndarray:
    """Stage possible repairs. The last channel records actual candidate ink."""
    original = image[..., :3]
    filled = original.copy()
    allowed = np.zeros(mask.shape, np.uint8)
    complex_ink = np.zeros_like(allowed)
    protected = np.zeros_like(allowed)
    for block in blocks:
        region = _region_ink(block, mask)
        if region is None:
            continue
        (x1, y1, x2, y2), ink = region
        if not ink.any():
            continue
        if source_review(block) == 'protected':
            protected[y1:y2, x1:x2][ink] = 255
            continue
        inside_bubble = block.manga_area != _plain_area(block, mask.shape[1], mask.shape[0])
        if not inside_bubble:
            white = _unframed_white_ink(block, original, mask)
            if white is not None:
                (left, top, right, bottom), complete, backing = white
                filled[top:bottom, left:right][complete] = backing
                allowed[top:bottom, left:right][complete] = 255
                continue
        crop = original[y1:y2, x1:x2]
        points = _polygon(block, mask.shape)
        interior = _area_mask(points, (x1, y1, x2, y2), inset=2) != 0
        pale = (crop.min(axis=2) >= 211) & interior
        pale_fraction = np.count_nonzero(pale) / max(1, np.count_nonzero(interior))
        # A thin heading spanning a light margin and coloured artwork has no
        # reliable common backing plate. Keep it visible for manual review.
        if not inside_bubble and (x2 - x1) >= 6 * (y2 - y1) and .24 < pale_fraction < .74:
            continue
        samples = crop[pale]
        if pale_fraction >= .58 and np.max(np.ptp(np.percentile(samples, (12, 88), axis=0), axis=0)) <= 20:
            base = np.median(samples, axis=0).astype(np.int16)
            stroke = (np.max(np.abs(crop.astype(np.int16) - base), axis=2) > 10) & interior
            if not inside_bubble:
                stroke &= cv2.dilate(ink.astype(np.uint8), np.ones((9, 9), np.uint8)) != 0
            ink = (cv2.dilate(stroke.astype(np.uint8), np.ones((3, 3), np.uint8)) != 0) & interior
            filled[y1:y2, x1:x2][ink] = base.astype(np.uint8)
        else:
            complex_ink[y1:y2, x1:x2][ink] = 255
        allowed[y1:y2, x1:x2][ink] = 255
    complex_ink[protected != 0] = 0
    allowed[protected != 0] = 0
    if np.any(complex_ink):
        if inpainter is None:
            repaired = cv2.inpaint(original.copy(), complex_ink, 3, cv2.INPAINT_TELEA)
        else:
            # The model sees surrounding context, but its output never
            # overwrites pixels outside the independently validated mask.
            if not inpainter.all_model_loaded():
                inpainter.load_model()
            repaired = inpainter.memory_safe_inpaint(original.copy(), complex_ink.copy())
        filled[complex_ink != 0] = repaired[complex_ink != 0]
    filled[allowed == 0] = original[allowed == 0]
    return np.dstack((filled, allowed))


def retry_uncertain_ocr(image: np.ndarray, blocks: list[TextBlock], ocr: object) -> None:
    """Recheck uncertain crops and merge only improved individual readings.

    The parallel text list retains empty slots for rejected lines even when
    OCRBase joins or postprocesses the public block.text value.
    """
    candidates = []
    original = []
    height, width = image.shape[:2]
    for block in blocks:
        confidence = block.manga_line_confidence
        if (not confidence or min(confidence) >= MANGA_RETRY_CONFIDENCE
                or len(confidence) != len(block.lines)
                or len(block.manga_line_text) != len(confidence)):
            continue
        lines = np.asarray(block.lines, dtype=np.float32)
        center = lines.mean(axis=1, keepdims=True)
        padded = np.rint(center + (lines - center) * (1.32 if len(lines) == 1 else 1.18))
        padded = np.clip(padded, (0, 0), (width - 1, height - 1)).astype(np.int32)
        retried = copy.deepcopy(block)
        retried.lines = padded.tolist()
        retried.text = []
        retried.manga_line_confidence = []
        retried.manga_line_text = []
        retried.manga_partial = False
        candidates.append(retried)
        original.append(block)
    if not candidates:
        return
    for block, retried in zip(original, ocr.run_ocr(image, candidates)):
        changed = False
        for i, (confidence, text) in enumerate(zip(retried.manga_line_confidence, retried.manga_line_text)):
            if i >= len(block.manga_line_confidence):
                break
            if (text.strip() and confidence >= MANGA_LINE_MIN_CONFIDENCE
                    and confidence > block.manga_line_confidence[i]):
                block.manga_line_text[i] = text
                block.manga_line_confidence[i] = confidence
                changed = True
        if changed:
            from ballontranslator.modules.ocr.base import postprocess_ocr_text
            from ballontranslator.utils.config import pcfg

            block.text = [text for text in block.manga_line_text if text]
            source = block.get_text()
            processed = postprocess_ocr_text(source, pcfg.ocr_sublist, pcfg.module.ocr_text_postprocess)
            if source != processed:
                block.text = processed
            block.manga_partial = any(value < MANGA_LINE_MIN_CONFIDENCE
                                      for value in block.manga_line_confidence)


def _glyph_silhouette(block: TextBlock) -> tuple[np.ndarray, int, int] | None:
    from qtpy.QtCore import QRectF
    from qtpy.QtGui import QImage, QPainter
    from qtpy.QtWidgets import QGraphicsScene
    from ballontranslator.ui.misc import pixmap2ndarray
    from ballontranslator.ui.text_engine.item import TextBlkItem

    scene = QGraphicsScene()
    item = TextBlkItem(block, 0, show_rect=False)
    item.set_ui_guide_suppressed(True)
    scene.addItem(item)
    bounds = item.sceneBoundingRect().adjusted(-2, -2, 2, 2)
    left, top = math.floor(bounds.left()), math.floor(bounds.top())
    width, height = math.ceil(bounds.right()) - left, math.ceil(bounds.bottom()) - top
    if min(width, height) < 1 or max(width, height) > 8192 or width * height > 8_000_000:
        return None
    layer = QImage(width, height, QImage.Format.Format_ARGB32)
    layer.fill(0)
    painter = QPainter(layer)
    try:
        painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        scene.render(painter, QRectF(0, 0, width, height), QRectF(left, top, width, height))
    finally:
        painter.end()
    raster = pixmap2ndarray(layer, keep_alpha=True).copy()
    y, x = np.where(raster[:, :, 3] != 0)
    if not len(x):
        return None
    return raster[y.min():y.max() + 1, x.min():x.max() + 1], left + x.min(), top + y.min()


def _fit(
    block: TextBlock, points: np.ndarray, refit: bool, font: FontChoice | None = None,
) -> tuple[np.ndarray, int, int] | None:
    from ballontranslator.utils import shared

    bounds = _bounds(points)
    x1, y1, x2, y2 = bounds
    area = _area_mask(points, bounds, inset=1 if min(x2 - x1, y2 - y1) < 28 else 2)
    rect = block.bounding_rect()
    candidate = copy.deepcopy(block)
    candidate.rich_text = ''
    if font is not None:
        candidate.font_family, candidate.fontformat.font_weight = font
    if not candidate.font_family:
        candidate.font_family = shared.DEFAULT_FONT_FAMILY
    detected_size = candidate.font_size if candidate.font_size > 0 else 16
    # The same real Qt text renderer is used to measure the ink and place the
    # accepted pixels. Studying the geometry alone misses strokes and rotation.
    scales = (1, .88, .76, .66, .56, .48) if refit else (1,)
    for scale in scales:
        candidate.font_size = max(7, detected_size * scale)
        if refit:
            # Use the source line's writing direction but keep right-to-left
            # columns readable after replacing the taller/narrower characters.
            candidate.angle = max(-13, min(13, block.angle)) if min(x2 - x1, y2 - y1) > 32 else 0
            candidate.alignment = 1
            if min(x2 - x1, y2 - y1) <= 25 and x2 - x1 > (y2 - y1) * 4:
                candidate.vertical = False
            width, height = (x2 - x1 - 6), (y2 - y1 - 6)
            candidate._bounding_rect = [x1 + 3, y1 + 3, width, height]
        else:
            candidate._bounding_rect = list(rect)
        if min(candidate._bounding_rect[2:]) <= 0:
            return None
        rendered = _glyph_silhouette(candidate)
        if rendered is None:
            continue
        pixels, x, y = rendered
        h, w = pixels.shape[:2]
        if refit:
            x, y = (x1 + x2 - w) // 2, (y1 + y2 - h) // 2
        if x < x1 or y < y1 or x + w > x2 or y + h > y2:
            continue
        section = area[y - y1:y + h - y1, x - x1:x + w - x1]
        if np.any((pixels[:, :, 3] > 0) & (section == 0)):
            continue
        if refit:
            remapped = list(candidate._bounding_rect)
            remapped[0] += int(x - rendered[1])
            remapped[1] += int(y - rendered[2])
            block._bounding_rect = remapped
            block.fontformat = candidate.fontformat
            block.angle = candidate.angle
        return pixels, x, y
    return None


def compose_page(
    image: np.ndarray, draft: np.ndarray, blocks: list[TextBlock], refit: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Commit an erasure and its replacement together only if both fit safely."""
    if draft.shape[:2] != image.shape[:2] or draft.ndim != 3 or draft.shape[2] != 4:
        raise ValueError('Invalid page repair draft.')
    valid = draft[:, :, 3]
    applied = np.zeros_like(valid)
    reserved = np.zeros_like(valid, dtype=bool)
    cleaned = image.copy()
    overlays = []
    lettering = np.zeros_like(reserved)
    matcher = FontMatcher() if refit is None else None
    for index, block in enumerate(blocks):
        block.manga_review = source_review(block)
        points = _polygon(block, valid.shape)
        inside_bubble = block.manga_area != _plain_area(block, valid.shape[1], valid.shape[0])
        region = _region_ink(block, valid, inside_bubble=inside_bubble)
        if not inside_bubble:
            white = _unframed_white_ink(block, image, valid)
            if white is not None:
                (left, top, right, bottom), complete, _ = white
                region = (left, top, right, bottom), complete & (valid[top:bottom, left:right] != 0)
        if block.manga_review is None:
            if not block.translation.strip() or block.translation.strip() == block.get_text().strip():
                block.manga_review = 'untranslated'
            elif points is None or region is None or not region[1].any():
                block.manga_review = 'geometry'
        if block.manga_review is None:
            resize = refit is None or index in refit
            ranked = matcher.rank(image, valid, block) if matcher is not None else []
            for choice in (*ranked, None):
                rendered = _fit(block, points, resize, choice)
                if rendered is not None:
                    break
            if rendered is not None:
                overlays.append((block, region, rendered))
                continue
            block.manga_review = 'overflow'
        if region is not None:
            (x1, y1, x2, y2), ink = region
            reserved[y1:y2, x1:x2] |= ink

    for block, region, rendered in overlays:
        (x1, y1, x2, y2), ink = region
        pixels, x, y = rendered
        h, w = pixels.shape[:2]
        visible = pixels[:, :, 3] != 0
        if (np.any(ink & reserved[y1:y2, x1:x2])
                or np.any((reserved[y:y + h, x:x + w] | lettering[y:y + h, x:x + w]) & visible)):
            block.manga_review = 'geometry'
            continue
        applied[y1:y2, x1:x2][ink] = 255
        lettering[y:y + h, x:x + w] |= visible
    cleaned[applied != 0, :3] = draft[applied != 0, :3]
    result = cleaned.copy()
    for block, _, (pixels, x, y) in overlays:
        if block.manga_review:
            continue
        h, w = pixels.shape[:2]
        crop = result[y:y + h, x:x + w, :3]
        alpha = pixels[:, :, 3:4].astype(np.uint16)
        crop[:] = ((pixels[:, :, :3] * alpha + crop * (255 - alpha) + 127) // 255).astype(np.uint8)
    return cleaned, result, applied


def save_page(window: object, name: str, draft: np.ndarray, refit: set[int] | None = None) -> None:
    """Publish the same page generation through the existing image-save worker."""
    from ballontranslator.utils.config import pcfg

    project = window.imgtrans_proj
    original = project.read_img(name)
    cleaned, result, mask = compose_page(original, draft, project.pages[name], refit)
    project.save()
    saver = window.imsave_thread
    saver.saveImg(str(draft_path(project.directory, name)), draft)
    saver.saveImg(project.get_mask_path(name).rsplit('.', 1)[0] + '.png', mask)
    saver.saveImg(project.get_inpainted_path(name).rsplit('.', 1)[0] + '.png', cleaned)
    saver.saveImg(project.get_result_path(name), result, name,
                  save_params={'ext': pcfg.imgsave_ext, 'quality': pcfg.imgsave_quality})
