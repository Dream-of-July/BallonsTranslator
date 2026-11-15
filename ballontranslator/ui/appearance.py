"""Resolve the saved appearance preference into the applied theme.

macOS follows the system appearance unless the user forces light or dark;
other platforms keep the plain ``pcfg.darkmode`` toggle.
"""

from typing import Tuple

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QGuiApplication, QPalette

from ballontranslator.utils import shared
from ballontranslator.utils.config import (
    APPEARANCE_DARK,
    APPEARANCE_LIGHT,
    APPEARANCE_SYSTEM,
    pcfg,
)

# The stylesheet's established accent, used whenever the system has none.
DEFAULT_ACCENT_RGB: Tuple[int, int, int] = (30, 147, 229)


def _color_scheme_enum():
    # Qt.ColorScheme and QStyleHints.colorScheme() need Qt >= 6.5.
    return getattr(Qt, 'ColorScheme', None)


def system_prefers_dark() -> bool:
    """Whether Qt reports a dark scheme; False when it cannot tell.

    >>> isinstance(system_prefers_dark(), bool)
    True
    """
    scheme = _color_scheme_enum()
    if scheme is None or QGuiApplication.instance() is None:
        return False
    hints = QGuiApplication.styleHints()
    if not hasattr(hints, 'colorScheme'):
        return False
    return hints.colorScheme() == scheme.Dark


def resolve_darkmode() -> bool:
    """Dark flag the UI should use for the current preference and platform."""
    if not shared.ON_MACOS:
        return pcfg.darkmode
    if pcfg.appearance == APPEARANCE_DARK:
        return True
    if pcfg.appearance == APPEARANCE_LIGHT:
        return False
    return system_prefers_dark()


def apply_appearance_override() -> None:
    """Make native chrome (title bar, scrollbars, sheets) match a forced choice.

    QStyleHints.setColorScheme() needs Qt >= 6.8; without it only the themed
    widgets follow a forced light/dark choice.
    """
    scheme = _color_scheme_enum()
    if not shared.ON_MACOS or scheme is None or QGuiApplication.instance() is None:
        return
    hints = QGuiApplication.styleHints()
    if not hasattr(hints, 'setColorScheme'):
        return
    if pcfg.appearance == APPEARANCE_DARK:
        hints.setColorScheme(scheme.Dark)
    elif pcfg.appearance == APPEARANCE_LIGHT:
        hints.setColorScheme(scheme.Light)
    else:
        hints.setColorScheme(scheme.Unknown)


def sync_darkmode() -> bool:
    """Store the resolved dark flag in ``pcfg.darkmode``; True if it changed."""
    darkmode = resolve_darkmode()
    changed = darkmode != pcfg.darkmode
    pcfg.darkmode = darkmode
    return changed


def current_theme() -> str:
    """Name of the themes.json entry for the platform and ``pcfg.darkmode``.

    >>> current_theme() in ('eva-light', 'eva-dark', 'mac-light', 'mac-dark')
    True
    """
    family = 'mac' if shared.ON_MACOS else 'eva'
    return f'{family}-dark' if pcfg.darkmode else f'{family}-light'


def accent_rgb() -> Tuple[int, int, int]:
    """System accent as an (r, g, b) tuple, or the established blue.

    Read through QPalette so no PyObjC bridge is involved; only the cocoa
    platform reports the real system accent.
    """
    accent_role = getattr(QPalette.ColorRole, 'Accent', None)
    if (
        not shared.ON_MACOS
        or accent_role is None
        or QGuiApplication.instance() is None
        or QGuiApplication.platformName() != 'cocoa'
    ):
        return DEFAULT_ACCENT_RGB
    color: QColor = QGuiApplication.palette().color(accent_role)
    if not color.isValid():
        return DEFAULT_ACCENT_RGB
    return (color.red(), color.green(), color.blue())
