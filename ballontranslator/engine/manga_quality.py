"""Use local text geometry to couple Japanese typesetting and reversible cleanup."""
from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ballontranslator.ui.text_engine.item import TextBlkItem

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


def _inside_bubble(block: TextBlock, shape: tuple[int, int]) -> bool:
    return bool(block.manga_area) and block.manga_area != _plain_area(block, shape[1], shape[0])


def _grow(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    return cv2.dilate(mask.astype(np.uint8), kernel) != 0


def _erase_radius(block: TextBlock, bounds: tuple[int, int, int, int]) -> int:
    size = block._detected_font_size
    if size <= 0:
        x1, y1, x2, y2 = bounds
        size = max(8., min(x2 - x1, y2 - y1) / 2)
    return max(1, round(size * .12))


def _background_model(crop: np.ndarray, samples: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a quadratic RGB trend without treating a gradient as texture.

    >>> crop = np.full((12, 12, 3), 80, np.uint8)
    >>> model, spread = _background_model(crop, np.ones((12, 12), bool))
    >>> np.allclose(model, crop) and spread < 1
    True
    """
    h, w = samples.shape
    y, x = np.mgrid[-1:1:complex(h), -1:1:complex(w)].astype(np.float32)
    design = np.stack((np.ones_like(x), x, y, x * x, x * y, y * y), axis=2)
    rows = design[samples]
    values = crop[samples].astype(np.float32)
    if len(rows) < 24:
        return np.zeros_like(crop, dtype=np.float32), float('inf')
    # Bound fitting cost on large bubbles; evaluate the surface only once.
    stride = max(1, math.ceil(len(rows) / 4096))
    rows, values = rows[::stride], values[::stride]
    coefficients = np.linalg.lstsq(rows, values, rcond=None)[0]
    for _ in range(2):
        errors = np.max(np.abs(values - rows @ coefficients), axis=1)
        keep = errors <= max(4., float(np.percentile(errors, 85)))
        coefficients = np.linalg.lstsq(rows[keep], values[keep], rcond=None)[0]
    model = np.clip(design @ coefficients, 0, 255)
    spread = float(np.percentile(np.max(np.abs(values - rows @ coefficients), axis=1), 90))
    return model, spread


def _nonwhite_bubble(image: np.ndarray, block: TextBlock) -> list[list[int]] | None:
    """Find a closed, smooth interior on a text-seeded crop, never the page."""
    if not block.lines:
        return None
    height, width = image.shape[:2]
    points = np.asarray(block.lines, np.int32).reshape(-1, 2)
    tx1, ty1, tx2, ty2 = _bounds(points)
    padding = max(tx2 - tx1, ty2 - ty1, 24)
    bounds = (max(0, tx1 - padding), max(0, ty1 - padding),
              min(width, tx2 + padding), min(height, ty2 + padding))
    x1, y1, x2, y2 = bounds
    crop = image[y1:y2, x1:x2, :3]
    blurred = cv2.GaussianBlur(crop.astype(np.float32), (0, 0), 1.2)
    dx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0) / 4
    dy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1) / 4
    barrier = (np.max(np.hypot(dx, dy), axis=2) > 12).astype(np.uint8)
    barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, labels = cv2.connectedComponents(1 - barrier, connectivity=4)
    support = _line_support(block, bounds) != 0
    candidates, counts = np.unique(labels[support], return_counts=True)
    # A real enclosing interior dominates these seeds. Bound work on noisy art
    # with thousands of tiny components instead of contouring all of them.
    for label in candidates[np.argsort(-counts)][:16]:
        if label == 0:
            continue
        interior = labels == label
        if any(np.any(edge) for edge in (interior[0], interior[-1], interior[:, 0], interior[:, -1])):
            continue
        contours, _ = cv2.findContours(interior.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        contour = cv2.approxPolyDP(contour, 1.3, True).reshape(-1, 2) + (x1, y1)
        area = cv2.contourArea(contour)
        text_area = max(1, (tx2 - tx1) * (ty2 - ty1))
        _, _, bw, bh = cv2.boundingRect(contour)
        if not (1.25 * text_area < area < min(30 * text_area, height * width * .35)):
            continue
        if bw > width * .85 or bh > height * .85 or len(contour) < 3:
            continue
        if any(cv2.pointPolygonTest(contour, tuple(map(float, point)), False) < 0 for point in points):
            continue
        _, spread = _background_model(crop, interior & ~support)
        if spread <= 12:
            return contour.tolist()
    return None


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
            else:
                bubble = _nonwhite_bubble(image, item)
                if bubble is not None:
                    item.manga_area = bubble
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


def _erase_polygon(block: TextBlock, shape: tuple[int, int]) -> np.ndarray | None:
    points = _polygon(block, shape)
    if points is not None and not _inside_bubble(block, shape):
        # Detection boxes bound the glyphs, not their outline or the backing
        # samples. Keep layout geometry unchanged while giving cleanup room.
        x1, y1, x2, y2 = _bounds(points)
        if min(x2 - x1, y2 - y1) > max(16., block._detected_font_size) * 5:
            return points
        padding = max(16, round(max(0., block._detected_font_size)))
        left, top = max(0, x1 - padding), max(0, y1 - padding)
        right, bottom = min(shape[1] - 1, x2 + padding), min(shape[0] - 1, y2 + padding)
        points = np.asarray([[left, top], [right, top], [right, bottom], [left, bottom]], np.int32)
    return points


def _erase_owners(blocks: list[TextBlock], shape: tuple[int, int]) -> np.ndarray:
    """Partition cleanup margins; -1 denotes pixels outside every region.

    >>> _erase_owners([], (1, 3)).tolist()
    [[-1, -1, -1]]
    """
    owners = np.full(shape, -1, np.int32)
    nearest = np.full(shape, np.inf, np.float32)
    for index, block in enumerate(blocks):
        points = _erase_polygon(block, shape)
        if points is None or not block.lines:
            continue
        x1, y1, x2, y2 = bounds = _bounds(points)
        source = np.ones((y2 - y1, x2 - x1), np.uint8)
        for line in block.lines:
            cv2.fillPoly(source, [np.asarray(line, np.int32) - (x1, y1)], 0)
        distance = cv2.distanceTransform(source, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        closer = (distance < nearest[y1:y2, x1:x2]) & (_area_mask(points, bounds) != 0)
        nearest[y1:y2, x1:x2][closer] = distance[closer]
        owners[y1:y2, x1:x2][closer] = index
    return owners


def _source_colour_ink(block: TextBlock, crop: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    """Anchor known, readable source strokes without inventing detections."""
    support = np.zeros(crop.shape[:2], np.uint8)
    for line in block.lines:
        cv2.fillPoly(support, [np.asarray(line, np.int32) - bounds[:2]], 1)
    colour = np.asarray(block.fg_colors, dtype=np.float32)
    if source_review(block) is not None or colour.shape != (3,) or not np.isfinite(colour).all():
        return np.zeros(crop.shape[:2], bool)
    matches = (np.max(np.abs(crop.astype(np.float32) - colour), axis=2) <= 36) & (support != 0)
    # A foreground colour shared by the backing is not an ink measurement.
    if np.count_nonzero(matches) < 3 or np.count_nonzero(matches) > np.count_nonzero(support) * .45:
        matches[:] = False
    return matches


def _region_ink(
    block: TextBlock, mask: np.ndarray, *, inside_bubble: bool = False,
) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    points = _erase_polygon(block, mask.shape)
    if points is None or not block.lines:
        return None
    bounds = _bounds(points)
    x1, y1, x2, y2 = bounds
    area = _area_mask(points, bounds, inset=1)
    support = area if inside_bubble else _line_support(block, bounds)
    if not inside_bubble and block.manga_complex_background:
        # The completed halo may extend beyond CTD's original line support.
        support = _grow(support, _erase_radius(block, bounds) * 5).astype(np.uint8) * 255
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
    padding = max(16, round(max(0., block._detected_font_size)))
    bounds = (max(0, x1 - padding), max(0, y1 - padding),
              min(width, x2 + padding), min(height, y2 + padding))
    left, top, right, bottom = bounds
    crop = image[top:bottom, left:right, :3]
    rim = np.ones(crop.shape[:2], dtype=bool)
    rim[y1 - top:y2 - top, x1 - left:x2 - left] = False
    # White glyph halos often dominate a tight rim. Only distant backing
    # samples can establish paper; a halo is never a backing plate.
    rim &= ~_grow(_line_support(block, bounds), max(3, _erase_radius(block, bounds) * 2))
    surrounding = crop[rim]
    pale = surrounding[surrounding.min(axis=1) >= 225]
    if len(surrounding) < 24 or len(pale) < .9 * len(surrounding):
        return None
    if np.max(np.ptp(np.percentile(pale, (10, 90), axis=0), axis=0)) > 2:
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


def _complete_complex_ink(
    block: TextBlock, crop: np.ndarray, seed: np.ndarray, area: np.ndarray,
    bounds: tuple[int, int, int, int], radius: int, inside_bubble: bool,
) -> tuple[np.ndarray, np.ndarray]:
    detected = seed
    measured = _source_colour_ink(block, crop, bounds) & area
    if measured.any():
        seed = measured
    reach = max(radius * 5, round(max(0., block._detected_font_size) * .9))
    support = (_grow(_line_support(block, bounds), reach) & area
               & _grow(seed, reach))
    # OCR colours are RGB, as are engine images. Include measured extremes:
    # default/missing OCR colours must not hide a coloured fill or halo.
    colours = [np.asarray(block.fg_colors), np.asarray(block.bg_colors)]
    if seed.any():
        colours.append(np.median(crop[seed], axis=0))
    pixels = crop[support]
    backing = area & ~_grow(_line_support(block, bounds), radius * 2)
    if np.count_nonzero(backing) < 24:
        backing = area & ~_grow(seed, radius * 2)
    backing_colour = np.median(crop[backing], axis=0) if backing.any() else None
    if len(pixels):
        luminance = pixels.mean(axis=1)
        ordered = np.sort(luminance)
        for percentile in (5, 95):
            # Use an observed luminance so quantisation/gaps cannot leave an
            # empty colour cluster at an interpolated percentile.
            target = ordered[round((len(luminance) - 1) * percentile / 100)]
            colour = np.median(pixels[np.abs(luminance - target) <= 8], axis=0)
            # In a closed bubble an extreme can be the backing itself. On
            # unframed text, keep the halo even beside a white page margin.
            if (backing_colour is None or np.max(np.abs(colour - backing_colour)) > 18
                    or (not inside_bubble and percentile == 95 and colour.min() >= 225)):
                colours.append(colour)
    matches = seed.copy()
    pixels_float = crop.astype(np.float32)
    for colour in colours:
        if colour.shape == (3,) and np.isfinite(colour).all():
            matches |= np.max(np.abs(pixels_float - colour), axis=2) <= 36
    matches &= area
    count, labels = cv2.connectedComponents(matches.astype(np.uint8), connectivity=8)
    selected = np.zeros(count, bool)
    selected[np.unique(labels[seed])] = True
    selected[0] = False
    complete = (selected[labels] & support) | seed
    continuing = np.zeros(count, bool)
    continuing[np.unique(labels[area & ~support])] = True
    continuing[0] = False
    # A halo can join pale veil/paper or a border. Keep its nearby part, not
    # the whole connected backing feature, so textured bubbles retain context.
    complete &= ~continuing[labels] | _grow(seed, max(radius * 3, round(block._detected_font_size * .25)))
    # Isolated pale halos are also source evidence. Checking only the dark
    # core would certify a repair that leaves a white glyph outline. Other
    # measured extremes can be real texture and must not become ink anchors.
    anchors = (seed & detected) | (complete & ~continuing[labels] & (crop.min(axis=2) >= 225))
    if not anchors.any():
        anchors = detected
    return _grow(complete, radius) & area, anchors


def _residue_found(
    repaired: np.ndarray, ink: np.ndarray, area: np.ndarray, radius: int,
    model: np.ndarray, spread: float, ring: np.ndarray,
    *, original: np.ndarray | None = None,
) -> bool:
    """Reject strokes, allowing unchanged context and continuing backing lines.

    >>> pixels = np.full((12, 12, 3), 100, np.uint8)
    >>> ink = np.zeros((12, 12), bool); ink[4:8, 4:8] = True
    >>> bool(_residue_found(pixels, ink, np.ones_like(ink), 1,
    ...                    pixels.astype(np.float32), 0., ~ink, original=pixels))
    False
    """
    zone = _grow(ink, radius) & area
    if not zone.any() or not ring.any() or not math.isfinite(spread):
        return True
    residual = np.max(np.abs(repaired.astype(np.float32) - model), axis=2)
    grey = cv2.cvtColor(repaired.astype(np.float32), cv2.COLOR_RGB2GRAY)
    energy = np.hypot(cv2.Sobel(grey, cv2.CV_32F, 1, 0), cv2.Sobel(grey, cv2.CV_32F, 0, 1)) / 4
    outliers = residual > max(18., 2.5 * spread + 8)
    edges = energy > max(20., float(np.percentile(energy[ring], 99)) * 1.8 + 8)
    suspicious = outliers | edges
    if original is not None:
        # Untouched context is evidence, not a demand that the backing be
        # flat. Do not count its old border/texture in the mask's outer halo.
        changed = np.max(np.abs(repaired.astype(np.int16) - original.astype(np.int16)), axis=2) > 2
        suspicious &= ink | _grow(changed, 1)
        # Exempt only the straight part of a feature continuing from outside
        # the repair, never its whole component (a glyph may touch a border).
        length = max(9, radius * 6 + 1)
        for kernel in (np.ones((1, length), np.uint8), np.ones((length, 1), np.uint8),
                       np.eye(length, dtype=np.uint8), np.fliplr(np.eye(length, dtype=np.uint8))):
            lines = cv2.morphologyEx((outliers | edges).astype(np.uint8), cv2.MORPH_OPEN, kernel)
            count, labels = cv2.connectedComponents(lines, connectivity=8)
            continues = np.zeros(count, bool)
            continues[np.unique(labels[ring])] = True
            continues[0] = False
            suspicious &= ~continues[labels]
    return np.count_nonzero(suspicious & zone) > max(3, np.count_nonzero(zone) * .015)


def _repair_complex(
    original: np.ndarray, block: TextBlock, bounds: tuple[int, int, int, int],
    seed: np.ndarray, area: np.ndarray, inpainter: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Try a crop twice; publish nothing when the second repair still has ink."""
    from ballontranslator.modules.inpaint.base import enlarge_window

    x1, y1, x2, y2 = bounds
    crop = original[y1:y2, x1:x2]
    radius = _erase_radius(block, bounds)
    ink, source = _complete_complex_ink(block, crop, seed, area, bounds, radius,
                                      _inside_bubble(block, original.shape[:2]))
    for attempt in range(2):
        if attempt:
            block.manga_residue_retried = True
            ink = _grow(ink, radius * 2) & area
        zone = _grow(ink, radius) & area
        # Keep JPEG/antialiased outline fringes out of the fitting samples.
        ring = _grow(zone, max(6, radius * 5)) & area & ~_grow(ink, radius * 2)
        model, spread = _background_model(crop, ring)
        repaired = crop.copy()
        if spread <= 10:
            repaired[ink] = np.rint(model[ink]).astype(np.uint8)
        else:
            layout_bounds = _bounds(_polygon(block, original.shape[:2]))
            left, top, right, bottom = enlarge_window(layout_bounds, original.shape[1], original.shape[0], ratio=1.7)
            left, top = min(left, x1), min(top, y1)
            right, bottom = max(right, x2), max(bottom, y2)
            crop_mask = np.zeros((bottom - top, right - left), np.uint8)
            crop_mask[y1 - top:y2 - top, x1 - left:x2 - left][ink] = 255
            context = original[top:bottom, left:right].copy()
            if inpainter is None:
                result = cv2.inpaint(context, crop_mask, 3, cv2.INPAINT_TELEA)
            else:
                if not inpainter.all_model_loaded():
                    inpainter.load_model()
                result = inpainter.memory_safe_inpaint(context, crop_mask)
            repaired[ink] = result[y1 - top:y2 - top, x1 - left:x2 - left][ink]
        # Busy backing can hide strokes from the ring's statistical threshold.
        # Changing a few source pixels must not certify the remaining glyphs.
        seed_error = np.max(np.abs(crop[source].astype(np.float32) - model[source]), axis=1)
        source_change = np.max(np.abs(repaired[source].astype(np.float32) - crop[source]), axis=1)
        remaining = (source_change < 12) & (seed_error > max(12., spread * .5))
        # On textured art a few unrelated reconstructed pixels can happen to
        # match their old RGB. Only connected remnants establish a surviving
        # stroke; isolated matches must not veto an otherwise clean repair.
        remnants = np.zeros_like(ink, np.uint8)
        remnants[source] = remaining
        _, labels, stats, _ = cv2.connectedComponentsWithStats(remnants, connectivity=8)
        coherent = stats[:, cv2.CC_STAT_AREA] >= 3
        coherent[0] = False
        unchanged = np.count_nonzero(coherent[labels]) > max(3, np.count_nonzero(source) * .015)
        if not _residue_found(repaired, ink, area, radius, model, spread, ring, original=crop) and not unchanged:
            return repaired, ink
    block.manga_erase_failed = True
    return crop, np.zeros_like(ink)


def prepare_background(
    image: np.ndarray, mask: np.ndarray, blocks: list[TextBlock], inpainter: object,
) -> np.ndarray:
    """Stage possible repairs. The last channel records actual candidate ink."""
    original = image[..., :3]
    filled = original.copy()
    allowed = np.zeros(mask.shape, np.uint8)
    protected = np.zeros_like(allowed)
    claims = np.zeros(mask.shape, np.int32)
    owners = _erase_owners(blocks, mask.shape)
    regions = []
    for block in blocks:
        block.manga_complex_background = False
        block.manga_residue_retried = False
        block.manga_erase_failed = False
        region = _region_ink(block, mask)
        guard = None
        if region is not None:
            (x1, y1, x2, y2), ink = region
            if (not ink.any() and source_review(block) is None
                    and block.translation.strip() and block.translation.strip() != block.get_text().strip()):
                # Saved applied masks can omit a previously withheld block.
                # Use its existing OCR colour and line geometry, not a detector.
                ink = _source_colour_ink(block, original[y1:y2, x1:x2], region[0])
                region = region[0], ink
            guard = _grow(ink, _erase_radius(block, region[0]) * 2)
            claims[y1:y2, x1:x2] += guard
        regions.append((region, guard))
        if source_review(block) == 'protected':
            points = _polygon(block, mask.shape)
            if points is not None:
                cv2.fillPoly(protected, [points], 255)
    for index, (block, (region, guard)) in enumerate(zip(blocks, regions)):
        if region is None:
            continue
        (x1, y1, x2, y2), ink = region
        if not ink.any():
            continue
        if source_review(block) == 'protected':
            continue
        inside_bubble = _inside_bubble(block, mask.shape)
        if not inside_bubble:
            white = _unframed_white_ink(block, original, mask)
            if white is not None:
                (left, top, right, bottom), complete, backing = white
                filled[top:bottom, left:right][complete] = backing
                allowed[top:bottom, left:right][complete] = 255
                continue
        crop = original[y1:y2, x1:x2]
        points = _erase_polygon(block, mask.shape)
        interior = _area_mask(points, (x1, y1, x2, y2), inset=2) != 0
        pale = (crop.min(axis=2) >= 211) & interior
        pale_fraction = np.count_nonzero(pale) / max(1, np.count_nonzero(interior))
        # A thin heading spanning a light margin and coloured artwork has no
        # reliable common backing plate. Keep it visible for manual review.
        if not inside_bubble and (x2 - x1) >= 6 * (y2 - y1) and .24 < pale_fraction < .74:
            continue
        backing = interior & ~_grow(_line_support(block, (x1, y1, x2, y2)),
                                    max(3, _erase_radius(block, region[0]) * 2))
        backing_pale = pale & backing
        samples = crop[backing_pale]
        verified_pale = (np.count_nonzero(backing) >= 24
                         and np.count_nonzero(backing_pale) >= .9 * np.count_nonzero(backing))
        flat_limit = 8 if inside_bubble else 2
        if (verified_pale and pale_fraction >= .58
                and np.max(np.ptp(np.percentile(samples, (12, 88), axis=0), axis=0)) <= flat_limit):
            base = np.median(samples, axis=0).astype(np.int16)
            stroke = (np.max(np.abs(crop.astype(np.int16) - base), axis=2) > 10) & interior
            if not inside_bubble:
                stroke &= cv2.dilate(ink.astype(np.uint8), np.ones((9, 9), np.uint8)) != 0
            ink = (cv2.dilate(stroke.astype(np.uint8), np.ones((3, 3), np.uint8)) != 0) & interior
            filled[y1:y2, x1:x2][ink] = base.astype(np.uint8)
        else:
            block.manga_complex_background = True
            # Exclude other owners before segmentation, dilation and sampling.
            area = (_area_mask(points, (x1, y1, x2, y2), inset=1) != 0)
            area &= (protected[y1:y2, x1:x2] == 0) & (claims[y1:y2, x1:x2] <= guard)
            ink &= area
            if not ink.any():
                continue
            repaired, ink = _repair_complex(original, block, (x1, y1, x2, y2), ink, area, inpainter)
            # Ownership limits publication, not the available backing samples.
            ink &= owners[y1:y2, x1:x2] == index
            filled[y1:y2, x1:x2][ink] = repaired[ink]
        allowed[y1:y2, x1:x2][ink] = 255
    allowed[protected != 0] = 0
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


_NO_LINE_START = frozenset('、。，．？！：；…‥」』）〉》】〕ーぁぃぅぇぉっゃゅょゎゕゖァィゥェォッャュョヮヵヶㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿ')
_NO_LINE_END = frozenset('「『（〈《【〔')


def _layout_runs(item: TextBlkItem) -> list[str]:
    """Read actual Qt rows/columns, preserving UTF-16 grapheme boundaries."""
    from ballontranslator.ui.text_engine.rendering.indexing import _utf16_slice

    runs: list[str] = []
    paragraph = item.document().firstBlock()
    while paragraph.isValid():
        layout = paragraph.layout()
        previous_x = None
        for index in range(layout.lineCount()):
            line = layout.lineAt(index)
            text = _utf16_slice(paragraph.text(), line.textStart(), line.textLength())
            if item.blk.vertical and previous_x is not None and abs(line.x() - previous_x) < .5:
                runs[-1] += text
            else:
                runs.append(text)
            previous_x = line.x()
        paragraph = paragraph.next()
    return runs


def _balanced_breaks(runs: list[str], single_line: bool = False) -> bool:
    """Validate real breaks without adding paragraph separators.

    >>> _balanced_breaks(['天地玄', '黃宇宙'])
    True
    >>> _balanced_breaks(['天地玄', '，宇宙'])
    False
    """
    from ballontranslator.ui.text_engine.rendering.indexing import _grapheme_count

    runs = [text.strip() for text in runs if text.strip()]
    if not runs or (single_line and len(runs) != 1):
        return False
    lengths = [_grapheme_count(text) for text in runs]
    if max(lengths) - min(lengths) > 2:
        return False
    if len(runs) > 1 and sum(lengths) > 3 and lengths[-1] == 1:
        return False
    for before, after in zip(runs, runs[1:]):
        if after[0] in _NO_LINE_START or before[-1] in _NO_LINE_END:
            return False
        if before[-1] == after[0] and before[-1] in '…—':
            return False
    return True


def _balanced_boxes(block: TextBlock, width: float, height: float, single_line: bool) -> list[tuple[float, float]]:
    """Measure native advances, then search balanced capacities, fewest flows first.

    >>> boxes = _balanced_boxes(block, 120, 180, False)  # doctest: +SKIP
    >>> all(w <= 120 and h <= 180 for w, h in boxes)  # doctest: +SKIP
    True
    """
    from ballontranslator.ui.text_engine.item import TextBlkItem
    from ballontranslator.ui.text_engine.rendering.indexing import _grapheme_ranges, _grapheme_count

    block._bounding_rect[2:] = [width, height]
    probe = TextBlkItem(copy.deepcopy(block), 0, show_rect=False)
    try:
        runs = _layout_runs(probe)
        n = _grapheme_count(block.translation)
        if not n:
            return []
        if single_line:
            return [(width, height)]
        margin = probe.document().documentMargin() * 2
        if block.vertical:
            # CharFontFormat uses the same glyph metrics and spacing as the
            # vertical layout; punctuation may have a different cell advance.
            metrics = [probe.layout.get_char_fontfmt(0, start)
                       for start, _ in _grapheme_ranges(probe.document().firstBlock().text())]
            cells = [metric.tbr.height() * block.letter_spacing for metric in metrics]
            cell = max(1., float(np.median(cells))) if cells else block.font_size
            column = max(1., float(np.median([metric.tbr.width() for metric in metrics]))
                         * block.line_spacing) if metrics else block.font_size
            maximum = max(1, math.floor((height - margin) / cell))
            first = max(1, min(len(runs), math.ceil(n / maximum)))
            last = min(n, max(first, math.ceil(width / column) + 1))
        else:
            widths, heights = [], []
            paragraph = probe.document().firstBlock()
            while paragraph.isValid():
                layout = paragraph.layout()
                widths.extend(layout.lineAt(i).naturalTextWidth() for i in range(layout.lineCount()))
                heights.extend(layout.lineAt(i).height() for i in range(layout.lineCount()))
                paragraph = paragraph.next()
            cell = max(1., sum(widths) / n)
            first = max(1, len(runs))
            row = max(1., float(np.median(heights)) * block.line_spacing) if heights else block.font_size
            last = min(n, max(first, math.ceil(height / row) + 1))
        boxes = []
        for count in range(first, last + 1):
            balanced = math.ceil(n / count)
            for capacity in (balanced, balanced + 1, balanced - 1):
                if capacity < 1:
                    continue
                extent = capacity * cell + margin + .25
                size = (width, min(height, extent)) if block.vertical else (min(width, extent), height)
                if size not in boxes:
                    boxes.append(size)
        if (width, height) not in boxes:
            boxes.append((width, height))
        return boxes
    finally:
        # These items never enter a scene; destroy their Qt children promptly
        # during a potentially long font/box search.
        from qtpy import sip
        sip.delete(probe)


def _glyph_silhouette(block: TextBlock, *, balanced: bool = False, single_line: bool = False) -> tuple[np.ndarray, int, int] | None:
    from qtpy.QtCore import QRectF
    from qtpy.QtGui import QImage, QPainter
    from qtpy.QtWidgets import QGraphicsScene
    from ballontranslator.ui.misc import pixmap2ndarray
    from ballontranslator.ui.text_engine.item import TextBlkItem

    scene = QGraphicsScene()
    item = TextBlkItem(block, 0, show_rect=False)
    item.set_ui_guide_suppressed(True)
    scene.addItem(item)
    if balanced and not _balanced_breaks(_layout_runs(item), single_line):
        scene.clear()
        return None
    bounds = item.sceneBoundingRect().adjusted(-2, -2, 2, 2)
    left, top = math.floor(bounds.left()), math.floor(bounds.top())
    width, height = math.ceil(bounds.right()) - left, math.ceil(bounds.bottom()) - top
    if min(width, height) < 1 or max(width, height) > 8192 or width * height > 8_000_000:
        scene.clear()
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
    scene.clear()
    y, x = np.where(raster[:, :, 3] != 0)
    if not len(x):
        return None
    return raster[y.min():y.max() + 1, x.min():x.max() + 1], left + x.min(), top + y.min()


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """sRGB relative luminance (linear light, D65).

    >>> _luminance(np.array([[0, 0, 0], [255, 255, 255]])).tolist()
    [0.0, 1.0]
    """
    srgb = np.asarray(rgb, np.float32) / 255
    linear = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)
    return linear @ np.asarray([.2126, .7152, .0722])


def _contrast(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return (np.maximum(first, second) + .05) / (np.minimum(first, second) + .05)


def _source_outline(image: np.ndarray, background: np.ndarray, block: TextBlock) -> list[int] | None:
    """Measure a removed halo, excluding unchanged paper and artwork."""
    points = _polygon(block, image.shape[:2])
    if points is None:
        return None
    bounds = _bounds(points)
    x1, y1, x2, y2 = bounds
    original = image[y1:y2, x1:x2, :3]
    cleaned = background[y1:y2, x1:x2, :3]
    changed = np.max(np.abs(original.astype(np.int16) - cleaned), axis=2) > 24
    support = _line_support(block, bounds) != 0
    luminance = _luminance(original)
    contrasted = _contrast(luminance, _luminance(block.fontformat.frgb)) >= 4.5
    # Intermediate antialiased fill fringes are not an outline. Require a
    # separate extreme paint that also contrasts with the repaired backing.
    distinct = ((luminance < .03) | (luminance > .75))
    known = np.asarray(block.bg_colors)
    if _contrast(_luminance(known), _luminance(block.fontformat.frgb)) >= 4.5:
        distinct |= np.max(np.abs(original.astype(float) - known), axis=2) <= 18
    distinct &= _contrast(luminance, _luminance(cleaned)) >= 1.8
    halo = changed & support & contrasted & distinct
    if np.count_nonzero(halo) < max(6, np.count_nonzero(changed & support) * .08):
        return None
    return np.median(original[halo], axis=0).astype(int).tolist()


def _lettering_style(
    block: TextBlock, pixels: np.ndarray, background: np.ndarray,
    source_outline: list[int] | None,
) -> None:
    """Select editable fill/stroke from the cleaned glyph footprint and rim.

    >>> block = TextBlock()
    >>> glyph = np.zeros((10, 10, 4), np.uint8); glyph[..., 3] = 255
    >>> _lettering_style(block, glyph, np.full((10, 10, 3), 255, np.uint8), None)
    >>> block.stroke_width == 0 and block.manga_contrast_before == 21
    True
    """
    glyphs = pixels[..., 3] > 0
    around = _grow(glyphs, 2)
    luminance = _luminance(background[..., :3])
    samples = luminance[around]
    block.manga_background_luminance = float(np.median(samples))
    block.manga_background_spread = float(np.percentile(samples, 90) - np.percentile(samples, 10))
    fill = block.fontformat.frgb
    before = float(np.percentile(_contrast(_luminance(fill), luminance[glyphs]), 10))
    block.manga_contrast_before = before
    # Small native glyphs can have no fully opaque core. Their actual coverage
    # may make an otherwise adequate nominal fill illegible after blending.
    insufficient = before < 4.5 or _drawn_contrast(block, pixels, background) < 4.5
    if insufficient:
        fill = max(([0, 0, 0], [255, 255, 255]),
                   key=lambda rgb: np.percentile(_contrast(_luminance(rgb), luminance[glyphs]), 10))
        block.fontformat.frgb = fill
    # A pale, flat bubble retains its original fill and zero stroke. Texture
    # and removed source halos are evidence for a stroke, not OCR's bg colour
    # alone (OCR commonly reports the same colour for fill and outline).
    outline = (block.manga_background_spread > .08 or insufficient
               or source_outline is not None or block.stroke_width > 0)
    if outline:
        stroke = [255, 255, 255] if _luminance(fill) < .179 else [0, 0, 0]
        if source_outline is None and block.stroke_width > 0:
            source_outline = block.fontformat.srgb
        if source_outline is not None and _contrast(_luminance(fill), _luminance(source_outline)) >= 4.5:
            stroke = source_outline
        block.fontformat.srgb = stroke
        block.stroke_width = max(.10, 1.5 / block.font_size)
    block.manga_outline_applied = outline


def _fit(
    block: TextBlock, points: np.ndarray, refit: bool, font: FontChoice | None = None,
    available: np.ndarray | None = None,
    *, background: np.ndarray | None = None, erased: np.ndarray | None = None,
    source_outline: list[int] | None = None,
) -> tuple[np.ndarray, int, int] | None:
    """Fit the native painted silhouette, including its editable stroke.

    >>> _fit(TextBlock(), np.array([[0, 0], [0, 1], [1, 1]]), True) is None
    True
    """
    from ballontranslator.utils import shared

    bounds = _bounds(points)
    x1, y1, x2, y2 = bounds
    inset = 1 if min(x2 - x1, y2 - y1) < 28 else 2
    # Unframed detection boxes already tightly bound source ink. Their edge
    # is usable backing, whereas a bubble edge needs its outline clearance.
    if background is not None and not _inside_bubble(block, background.shape[:2]):
        inset = 0
    area = _area_mask(points, bounds, inset=inset)
    if available is not None:
        area[~available] = 0
    if not area.any():
        return None
    erased_colour = None
    if background is not None and erased is not None:
        samples = background[y1:y2, x1:x2, :3][erased[y1:y2, x1:x2] != 0]
        if len(samples):
            erased_colour = np.median(samples, axis=0)
    candidate = copy.deepcopy(block)
    candidate.rich_text = ''
    if font is not None:
        candidate.font_family, candidate.fontformat.font_weight = font
    if not candidate.font_family:
        candidate.font_family = shared.DEFAULT_FONT_FAMILY
    detected_size = candidate.font_size if candidate.font_size > 0 else 16
    # Never turn small source lettering into illegibly tiny output. A manual
    # update without refitting keeps the user's exact size/spacing/rectangle.
    floor = max(9., block.detected_font_size * .55 if block.detected_font_size <= 16 else 9.)
    sizes = list(dict.fromkeys(max(floor, detected_size * scale)
                              for scale in (1, .88, .76, .66, .56, .48, .40))) if refit else [detected_size]
    # Large source fonts can leave every scaled candidate above the readable
    # floor. Continue in pixel steps so a larger compact fit precedes the floor.
    if refit:
        smallest = sizes[-1]
        sizes.extend(max(floor, smallest - step)
                     for step in range(1, math.ceil(smallest - floor) + 1))
    distance = cv2.distanceTransform(area, cv2.DIST_L2, 5)
    cy, cx = np.unravel_index(np.argmax(distance), distance.shape)
    centers = [(x1 + cx, y1 + cy)]
    # A small deterministic neighbourhood also handles flat distance maxima
    # (rectangles/strips). Keep the bbox centre as a later candidate.
    step = max(2, min(12, round(float(distance[cy, cx]) * .25)))
    centers.extend((x1 + cx + dx * step, y1 + cy + dy * step)
                   for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0),
                                  (-1, -1), (1, -1), (-1, 1), (1, 1)))
    centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
    if background is not None:
        bx1, by1, bx2, by2 = block.xyxy
        centers.append(((bx1 + bx2) / 2, (by1 + by2) / 2))
    single_line = (len(block.lines) == 1 and not block.src_is_vertical
                   and x2 - x1 >= 4 * (y2 - y1))
    spacing = [(block.letter_spacing, block.line_spacing)]
    if refit and not single_line:
        compact = (min(block.letter_spacing, 1.), min(block.line_spacing, 1.1))
        if compact != spacing[0]:
            spacing.append(compact)
    for size in sizes:
        candidate.font_size = size
        for letters, lines in spacing:
            candidate.fontformat.letter_spacing = letters
            candidate.line_spacing = lines
            if refit:
                candidate.angle = max(-13, min(13, block.angle)) if min(x2 - x1, y2 - y1) > 32 else 0
                candidate.alignment = 1
                if single_line:
                    candidate.vertical = False
                width, height = x2 - x1 - 6, y2 - y1 - 6
                if min(width, height) <= 0:
                    return None
                candidate._bounding_rect = [x1 + 3, y1 + 3, width, height]
                boxes = _balanced_boxes(candidate, width, height, single_line)
            else:
                candidate._bounding_rect = list(block.bounding_rect())
                boxes = [tuple(candidate._bounding_rect[2:])]
            for width, height in boxes:
                candidate._bounding_rect[2:] = [width, height]
                rendered = _glyph_silhouette(candidate, balanced=refit, single_line=single_line)
                if rendered is None:
                    continue
                pixels, original_x, original_y = rendered
                h, w = pixels.shape[:2]
                if w > x2 - x1 or h > y2 - y1:
                    continue
                positions = [(round(cx - w / 2), round(cy - h / 2)) for cx, cy in centers] if refit else [(original_x, original_y)]
                positions = [(x, y) for x, y in positions
                             if x1 <= x and y1 <= y and x + w <= x2 and y + h <= y2]
                if erased_colour is not None:
                    ink = pixels[..., 3] != 0
                    def placement_cost(position: tuple[int, int]) -> tuple[float, float]:
                        px, py = position
                        zone = erased[py:py + h, px:px + w] != 0
                        similar = np.max(np.abs(background[py:py + h, px:px + w, :3].astype(float) - erased_colour), axis=2) <= 36
                        return float(np.mean((~zone & ~similar)[ink])), -float(np.mean(zone[ink]))
                    positions.sort(key=placement_cost)
                for x, y in positions:
                    styled = candidate
                    rendered_pixels, rx, ry = pixels, original_x, original_y
                    px, py, ph, pw = x, y, h, w
                    if background is not None:
                        styled = copy.deepcopy(candidate)
                        _lettering_style(styled, pixels, background[y:y + h, x:x + w], source_outline)
                        if (styled.stroke_width != candidate.stroke_width
                                or list(styled.fontformat.frgb) != list(candidate.fontformat.frgb)
                                or list(styled.fontformat.srgb) != list(candidate.fontformat.srgb)):
                            styled_render = _glyph_silhouette(styled, balanced=refit, single_line=single_line)
                            if styled_render is None:
                                continue
                            rendered_pixels, rx, ry = styled_render
                            ph, pw = rendered_pixels.shape[:2]
                            px, py = (round(x + (w - pw) / 2), round(y + (h - ph) / 2)) if refit else (rx, ry)
                    if px < x1 or py < y1 or px + pw > x2 or py + ph > y2:
                        continue
                    section = area[py - y1:py + ph - y1, px - x1:px + pw - x1]
                    if np.any((rendered_pixels[:, :, 3] > 0) & (section == 0)):
                        continue
                    if refit:
                        remapped = list(styled._bounding_rect)
                        remapped[0] += int(px - rx)
                        remapped[1] += int(py - ry)
                        block._bounding_rect = remapped
                        block.angle = styled.angle
                    block.fontformat = copy.deepcopy(styled.fontformat)
                    block.manga_contrast_before = styled.manga_contrast_before
                    block.manga_background_luminance = styled.manga_background_luminance
                    block.manga_background_spread = styled.manga_background_spread
                    block.manga_outline_applied = styled.manga_outline_applied
                    return rendered_pixels, px, py
    return None


def _soft_backing(
    block: TextBlock, points: np.ndarray, rendered: tuple[np.ndarray, int, int],
    cleaned: np.ndarray, applied: np.ndarray, forbidden: np.ndarray,
) -> None:
    """Publish a feathered text-sized veil, clipped to the bubble interior."""
    if not block.manga_outline_applied or not _inside_bubble(block, applied.shape):
        return
    pixels, x, y = rendered
    h, w = pixels.shape[:2]
    visible = pixels[..., 3] > 64
    edge = visible & ~cv2.erode(visible.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if not edge.any():
        return
    luminance = _luminance(cleaned[y:y + h, x:x + w, :3]).astype(np.float32)
    mean = cv2.blur(luminance, (5, 5))
    spread = np.sqrt(np.maximum(0, cv2.blur(luminance ** 2, (5, 5)) - mean ** 2))
    weak = (_contrast(_luminance(block.fontformat.srgb), luminance) < 3) | (spread > .06)
    if np.mean(weak[edge]) < .20:
        return
    left, top, right, bottom = bounds = _bounds(points)
    inside = _area_mask(points, bounds, inset=2) != 0
    inside &= ~forbidden[top:bottom, left:right]
    crop = cleaned[top:bottom, left:right, :3]
    if not inside.any():
        return
    colour = np.median(crop[inside], axis=0)
    margin = max(3, round(block.font_size * .35))
    yy, xx = np.ogrid[top:bottom, left:right]
    distance = np.maximum.reduce(np.broadcast_arrays(x - xx, xx - (x + w - 1),
                                                     y - yy, yy - (y + h - 1)))
    weight = np.clip(1 - np.maximum(0, distance) / margin, 0, 1)
    # Smoothstep feathers only the outer margin; clipping happens afterwards,
    # so filtering cannot leak even one pixel across the bubble outline.
    weight = (.72 * weight ** 2 * (3 - 2 * weight) * inside)[..., None]
    crop[:] = np.rint(crop * (1 - weight) + colour * weight).astype(np.uint8)
    published = weight[..., 0] > 0
    applied[top:bottom, left:right][published] = 255
    block.manga_backing_applied = bool(published.any())


def _drawn_contrast(block: TextBlock, rendered: np.ndarray, backing: np.ndarray) -> float:
    """10th-percentile core contrast against the adjacent backing or stroke.

    Antialias fringes are excluded. For outlined letters the adjacent paint is
    measured from opaque stroke pixels of the real Qt raster, not from a
    theoretical replacement of the page luminance.

    >>> glyph = np.array([[[0, 0, 0, 255]]], np.uint8)
    >>> _drawn_contrast(TextBlock(), glyph, np.full((1, 1, 3), 255, np.uint8))
    21.0
    """
    alpha = rendered[..., 3:4].astype(np.float32) / 255
    composited = rendered[..., :3] * alpha + backing[..., :3] * (1 - alpha)
    fill = np.asarray(block.fontformat.frgb)
    core = (rendered[..., 3] >= 224) & (np.max(np.abs(rendered[..., :3].astype(float) - fill), axis=2) <= 24)
    if not core.any():
        return 1.
    comparison = _luminance(backing[..., :3])[core]
    if block.manga_outline_applied:
        stroke = np.asarray(block.fontformat.srgb)
        rim = (rendered[..., 3] >= 224) & (np.max(np.abs(rendered[..., :3].astype(float) - stroke), axis=2) <= 24)
        if rim.any():
            comparison = np.median(_luminance(composited)[rim])
    return float(np.percentile(_contrast(_luminance(composited)[core], comparison), 10))


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
    owners = _erase_owners(blocks, valid.shape)
    # Rectangular detected interiors are narration boxes, even when their
    # pale connected component was found by the speech-bubble detector.
    kinds = []
    for block in blocks:
        points = _polygon(block, valid.shape)
        if points is None:
            kinds.append(False)
            continue
        left, top, right, bottom = _bounds(points)
        rectangular = cv2.contourArea(points) / max(1, (right - left - 1) * (bottom - top - 1)) >= .92
        kinds.append(not rectangular and _inside_bubble(block, valid.shape))
    fonts = FontMatcher().page_choices(image, valid, blocks, kinds) if refit is None else {}
    source_outlines = [_source_outline(image, draft, block) if source_review(block) is None else None
                       for block in blocks]
    for index, block in enumerate(blocks):
        block.manga_contrast_before = block.manga_contrast_after = None
        block.manga_background_luminance = block.manga_background_spread = None
        block.manga_outline_applied = block.manga_backing_applied = False
        block.manga_review = source_review(block)
        points = _polygon(block, valid.shape)
        inside_bubble = _inside_bubble(block, valid.shape)
        if block.manga_review == 'unreadable' and not inside_bubble:
            block.manga_review = 'skipped'
        if block.manga_review is None and block.manga_erase_failed:
            block.manga_review = 'geometry'
        region = _region_ink(block, valid, inside_bubble=inside_bubble)
        if block.manga_review == 'skipped':
            # An unreadable art detection has no source ink to reserve. Its
            # expanded box can otherwise borrow a neighbouring bubble's mask
            # and veto that bubble's independently verified white repair.
            region = None
        if region is not None and block.manga_complex_background:
            (x1, y1, x2, y2), ink = region
            region = region[0], ink & (owners[y1:y2, x1:x2] == index)
        if not inside_bubble and block.manga_review != 'skipped':
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
            candidate = copy.deepcopy(block)
            rendered = _fit(candidate, points, resize, fonts.get(index),
                            background=draft, erased=valid, source_outline=source_outlines[index])
            if rendered is not None:
                overlays.append([index, block, candidate, region, rendered])
                continue
            block.manga_review = 'overflow'
        if region is not None:
            (x1, y1, x2, y2), ink = region
            reserved[y1:y2, x1:x2] |= ink

    retried = set()
    for entry in overlays:
        index, block, candidate, region, rendered = entry
        (x1, y1, x2, y2), ink = region
        pixels, x, y = rendered
        h, w = pixels.shape[:2]
        visible = pixels[:, :, 3] != 0
        source_collision = np.any(ink & reserved[y1:y2, x1:x2])
        if source_collision or np.any((reserved[y:y + h, x:x + w] | lettering[y:y + h, x:x + w]) & visible):
            points = _polygon(block, valid.shape)
            left, top, right, bottom = _bounds(points)
            remaining = ~(reserved[top:bottom, left:right] | lettering[top:bottom, left:right])
            candidate = copy.deepcopy(block)
            # Exactly one re-placement search, starting from the original size
            # and spacing. Source erasure itself may never cross protected ink.
            retried.add(index)
            retry = _fit(candidate, points, refit is None or index in refit, fonts.get(index), remaining,
                         background=draft, erased=valid, source_outline=source_outlines[index])
            if source_collision or retry is None:
                block.manga_review = 'geometry'
                reserved[y1:y2, x1:x2] |= ink
                continue
            entry[2], entry[4] = candidate, retry
            pixels, x, y = retry
            h, w = pixels.shape[:2]
            visible = pixels[:, :, 3] != 0
        lettering[y:y + h, x:x + w] |= visible

    # A later rejected block can reveal original ink under an earlier
    # placement. Withholding is monotonic; never publish that earlier overlap.
    changed = True
    while changed:
        changed = False
        for entry in overlays:
            index, block, _, region, (pixels, x, y) = entry
            if block.manga_review:
                continue
            (x1, y1, x2, y2), ink = region
            h, w = pixels.shape[:2]
            if (np.any(ink & reserved[y1:y2, x1:x2])
                    or np.any(reserved[y:y + h, x:x + w] & (pixels[:, :, 3] != 0))):
                lettering[y:y + h, x:x + w][pixels[:, :, 3] != 0] = False
                retry = None
                if index not in retried:
                    retried.add(index)
                    points = _polygon(block, valid.shape)
                    left, top, right, bottom = _bounds(points)
                    remaining = ~(reserved[top:bottom, left:right] | lettering[top:bottom, left:right])
                    candidate = copy.deepcopy(block)
                    retry = _fit(candidate, points, refit is None or index in refit, fonts.get(index), remaining,
                                 background=draft, erased=valid, source_outline=source_outlines[index])
                if retry is not None and not np.any(ink & reserved[y1:y2, x1:x2]):
                    entry[2], entry[4] = candidate, retry
                    pixels, x, y = retry
                    h, w = pixels.shape[:2]
                    lettering[y:y + h, x:x + w] |= pixels[:, :, 3] != 0
                else:
                    block.manga_review = 'geometry'
                    reserved[y1:y2, x1:x2] |= ink
                    changed = True
    for _, block, candidate, region, _ in overlays:
        if block.manga_review:
            continue
        (x1, y1, x2, y2), ink = region
        applied[y1:y2, x1:x2][ink] = 255
        block._bounding_rect = candidate._bounding_rect
        block.fontformat = candidate.fontformat
        block.angle = candidate.angle
        block.manga_contrast_before = candidate.manga_contrast_before
        block.manga_background_luminance = candidate.manga_background_luminance
        block.manga_background_spread = candidate.manga_background_spread
        block.manga_outline_applied = candidate.manga_outline_applied
    cleaned[applied != 0, :3] = draft[applied != 0, :3]
    for _, block, _, _, rendered in overlays:
        if not block.manga_review:
            _soft_backing(block, _polygon(block, valid.shape), rendered, cleaned, applied, reserved)
    result = cleaned.copy()
    for _, block, _, _, (pixels, x, y) in overlays:
        if block.manga_review:
            continue
        h, w = pixels.shape[:2]
        crop = result[y:y + h, x:x + w, :3]
        block.manga_contrast_after = _drawn_contrast(block, pixels, crop)
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
