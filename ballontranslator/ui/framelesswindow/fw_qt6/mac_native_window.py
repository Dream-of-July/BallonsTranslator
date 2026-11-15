from typing import Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget


class MacNativeWindow(QWidget):
    """Top-level window that keeps the system title bar on macOS.

    Other platforms draw their own frame; macOS already provides traffic
    lights, dragging and zooming, so this is a plain titled window.

    >>> issubclass(MacNativeWindow, QWidget)
    True
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        flags: Qt.WindowType = Qt.WindowType.Widget,
    ) -> None:
        super().__init__(parent, flags)
