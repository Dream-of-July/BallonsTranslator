"""Match source lettering against the fonts the local renderer can use."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from unicodedata import east_asian_width, normalize

import cv2
import numpy as np
from qtpy.QtGui import QFont, QFontDatabase, QRawFont

from ballontranslator.ui.text_engine.font_family import qfont_with_family
from ballontranslator.utils import shared
from ballontranslator.utils.font_registry import normalize_key
from ballontranslator.utils.fontformat import FontWeight, font_weight_from_qt, font_weight_to_qt
from ballontranslator.utils.textblock import TextBlock


FontChoice = tuple[str, FontWeight]
_GLYPH_SIZE = 32
_LINE_HEIGHT = 48
_MAX_SAMPLES = 12


def _profile(ink: np.ndarray) -> np.ndarray | None:
    x, y, width, height = cv2.boundingRect(ink)
    if min(width, height) < 3:
        return None
    scale = (_GLYPH_SIZE - 6) / max(width, height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (_GLYPH_SIZE - w) // 2, (_GLYPH_SIZE - h) // 2
    square = np.zeros((_GLYPH_SIZE, _GLYPH_SIZE), dtype=np.float32)
    square[top:top + h, left:left + w] = cv2.resize(
        ink[y:y + height, x:x + width].astype(np.float32), (w, h), interpolation=cv2.INTER_AREA,
    ) / 255
    return cv2.GaussianBlur(square, (5, 5), .9)


def _projected_glyphs(
    ink: np.ndarray, text: str, vertical: bool,
) -> list[tuple[str, np.ndarray]]:
    chars = [ch for ch in normalize('NFKC', text) if not ch.isspace()]
    if not chars or len(chars) > 32:
        return []
    x, _, width, height = cv2.boundingRect(ink)
    units = np.array([1 if vertical or east_asian_width(ch) in 'WF' else .65 for ch in chars])
    step = width / units.sum()
    if height < 14 or not 11 <= step <= _LINE_HEIGHT * 1.55:
        return []
    projection = np.count_nonzero(ink[:, x:x + width], axis=0)
    cuts = [0]
    for position in np.cumsum(units)[:-1] * step:
        center = round(position)
        radius = round(step * .25)
        choices = np.arange(max(1, center - radius), min(width, center + radius + 1))
        cost = projection[choices - 1] + projection[choices] + 9 * abs(choices - position) / step
        cuts.append(int(choices[np.argmin(cost)]))
    cuts.append(width)
    samples = []
    for ch, left, right in zip(chars, cuts, cuts[1:]):
        if not ch.isalpha() or east_asian_width(ch) not in 'WF':
            continue
        # A cut through a joined stroke is weak evidence about either glyph.
        if ((left > 0 and min(projection[left - 1:left + 1]) > 5)
                or (right < width and min(projection[right - 1:right + 1]) > 5)):
            continue
        glyph = ink[:, x + left:x + right]
        if glyph.size == 0:
            continue
        if vertical:
            glyph = cv2.rotate(glyph, cv2.ROTATE_90_CLOCKWISE)
        profile = _profile(glyph)
        if profile is not None:
            samples.append((ch, profile))
    return samples


def _source_glyphs(
    image: np.ndarray, mask: np.ndarray, block: TextBlock,
) -> list[tuple[str, np.ndarray]]:
    lines = block.text
    if len(block.lines) == 1:
        lines = [block.get_text()]
    if not isinstance(lines, list) or len(lines) != len(block.lines):
        return []
    samples = []
    for index, text in enumerate(lines):
        if not text:
            continue
        try:
            strip = block.get_transformed_region(image, index, _LINE_HEIGHT, maxwidth=1100)
            support = block.get_transformed_region(mask, index, _LINE_HEIGHT, maxwidth=1100) > 16
        except cv2.error:
            continue
        if support.ndim != 2 or not support.any() or np.mean(support) > .75:
            continue
        background = np.median(strip[~support], axis=0)
        contrast = np.max(abs(strip.astype(np.int16) - background), axis=2).astype(np.uint8)
        contrast[~support] = 0
        threshold, _ = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        ink = np.where(contrast > max(threshold, 15), np.uint8(255), np.uint8(0))
        samples.extend(_projected_glyphs(ink, text, bool(block.src_is_vertical)))
    if len(samples) > _MAX_SAMPLES:
        samples = [samples[i] for i in np.linspace(0, len(samples) - 1, _MAX_SAMPLES, dtype=int)]
    return samples


@dataclass
class _Face:
    choice: FontChoice
    raw: QRawFont
    glyphs: dict[str, np.ndarray | None] = field(default_factory=dict)

    def glyph(self, char: str) -> np.ndarray | None:
        if char not in self.glyphs:
            index = self.raw.glyphIndexesForString(char)[0]
            alpha = self.raw.alphaMapForGlyph(index, QRawFont.AntialiasingType.PixelAntialiasing)
            self.glyphs[char] = None
            if not alpha.isNull() and alpha.depth() == 8:
                bits = alpha.constBits()
                if hasattr(bits, 'setsize'):
                    bits.setsize(alpha.sizeInBytes())
                raster = np.frombuffer(bits, np.uint8).reshape(alpha.height(), alpha.bytesPerLine())
                self.glyphs[char] = _profile(raster[:, :alpha.width()])
        return self.glyphs[char]


def _installed_faces() -> list[_Face]:
    from ballontranslator.utils.config import pcfg

    db = QFontDatabase if shared.FLAG_QT6 else QFontDatabase()
    scripts = QFontDatabase.WritingSystem
    families = set(db.families(scripts.SimplifiedChinese)) | set(db.families(scripts.TraditionalChinese))
    registry = shared.FONT_REGISTRY
    if registry is None:
        entries = [(family, family, {normalize_key(family)}) for family in sorted(families)]
    else:
        entries = []
        for entry in registry.entries(only_custom=pcfg.let_show_only_custom_fonts_flag,
                                      excluded=pcfg.excluded_fonts):
            if entry.source != 'custom' and entry.qt_family not in families:
                continue
            aliases = entry.lookup_keys()
            if entry.is_pseudo_group:
                entries.extend((face.storage_family, face.qt_family, aliases) for face in entry.faces)
            else:
                entries.append((entry.canonical_family, entry.qt_family, aliases))
    found = []
    seen = set()
    for family, qt_family, aliases in entries:
        if not db.isSmoothlyScalable(qt_family):
            continue
        weights = {font_weight_from_qt(db.weight(qt_family, style)) for style in db.styles(qt_family)}
        for weight in sorted(weights or {FontWeight.Normal}):
            font = QFont()
            font.setWeight(QFont.Weight(font_weight_to_qt(weight)))
            font = qfont_with_family(font, family)
            font.setPixelSize(_LINE_HEIGHT)
            raw = QRawFont.fromFont(font)
            if not raw.isValid() or normalize_key(raw.familyName()) not in aliases:
                continue
            key = (raw.familyName(), raw.styleName())
            if key not in seen and raw.supportsCharacter(ord('中')):
                found.append(_Face((family, weight), raw))
                seen.add(key)
    return found


class FontMatcher:
    """A page-local collection of eligible faces with lazily rendered glyphs."""

    def __init__(self) -> None:
        self.faces: list[_Face] | None = None

    def rank(self, image: np.ndarray, mask: np.ndarray, block: TextBlock) -> list[FontChoice]:
        samples = _source_glyphs(image, mask, block)
        if len(samples) < 2:
            return []
        if self.faces is None:
            self.faces = _installed_faces()
        required = {ord(ch) for ch in block.translation if ch.isprintable() and not ch.isspace()}
        matches = []
        for face in self.faces:
            if not all(face.raw.supportsCharacter(code) for code in required):
                continue
            scores = []
            for char, source in samples:
                if not face.raw.supportsCharacter(ord(char)):
                    continue
                glyph = face.glyph(char)
                if glyph is None:
                    continue
                source_ink, glyph_ink = source.sum(), glyph.sum()
                shape = np.abs(source - glyph).sum() / np.maximum(source, glyph).sum()
                density = abs(source_ink - glyph_ink) / max(source_ink, glyph_ink)
                scores.append(.8 * shape + .2 * density)
            if len(scores) < max(2, (len(samples) + 1) // 2):
                continue
            # Handle a stray OCR error or ruby annotation without trusting a single perfect glyph.
            score = .75 * np.median(scores) + .25 * np.mean(scores)
            score += .12 * (1 - len(scores) / len(samples))
            if score < .65:
                matches.append((score, face.choice))
        return [choice for _, choice in sorted(matches, key=lambda match: match[0])[:2]]

    def page_choices(
        self, image: np.ndarray, mask: np.ndarray, blocks: list[TextBlock], kinds: list[bool],
    ) -> dict[int, FontChoice]:
        """Use a majority family and median weight per bubble/narration group.

        An explicit source bold weight plus agreement from glyph matching is
        required for a bold exception. Missing evidence never invents emphasis.
        """
        ranks = [self.rank(image, mask, block) for block in blocks]
        choices = {}
        for kind in (False, True):
            indices = [i for i, value in enumerate(kinds) if value == kind]
            votes = [ranks[i][0] for i in indices if ranks[i]]
            if not votes:
                common = (shared.DEFAULT_FONT_FAMILY, FontWeight.Normal)
            else:
                # Counter preserves page order for deterministic ties.
                family = Counter(face[0] for face in votes).most_common(1)[0][0]
                weights = sorted(weight for name, weight in votes if name == family)
                common = (family, weights[(len(weights) - 1) // 2])
            for i in indices:
                choice = common
                if (blocks[i].fontformat.font_weight >= FontWeight.Bold and ranks[i]
                        and ranks[i][0][0] == common[0]
                        and ranks[i][0][1] >= FontWeight.Bold
                        and len(_source_glyphs(image, mask, blocks[i])) >= 4):
                    choice = ranks[i][0]
                choices[i] = choice
        return choices
