import logging

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger("QuallyGUI")


class EmptyState(QWidget):
    """Centered placeholder shown when a list or table has no content."""

    def __init__(self, icon="", heading="Nothing here yet", subtitle="", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 48, 24, 48)

        if icon:
            icon_lbl = QLabel(icon)
            icon_lbl.setObjectName("emptyStateIcon")
            icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(icon_lbl)

        heading_lbl = QLabel(heading)
        heading_lbl.setObjectName("emptyStateHeading")
        heading_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(heading_lbl)

        if subtitle:
            sub_lbl = QLabel(subtitle)
            sub_lbl.setObjectName("emptyStateSubtitle")
            sub_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            sub_lbl.setWordWrap(True)
            layout.addWidget(sub_lbl)


class SectionHeader(QWidget):
    """Page section title with an optional one-line subtitle."""

    def __init__(self, title, subtitle="", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.setSpacing(2)

        title_lbl = QLabel(title)
        title_lbl.setObjectName("sectionTitle")
        layout.addWidget(title_lbl)

        if subtitle:
            sub_lbl = QLabel(subtitle)
            sub_lbl.setObjectName("sectionSubtitle")
            sub_lbl.setWordWrap(True)
            layout.addWidget(sub_lbl)


class StatusBadge(QLabel):
    """Colored pill label showing API key / provider connection status."""

    _LABELS = {
        "connected":      "Connected",
        "not_configured": "Not configured",
        "invalid":        "Invalid",
        "testing":        "Testing...",
        "unknown":        "—",
    }

    def __init__(self, state="unknown", parent=None):
        super().__init__(parent)
        self.setObjectName("statusBadge")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedWidth(110)
        self.set_state(state)

    def set_state(self, state: str):
        self.setText(self._LABELS.get(state, "—"))
        self.setProperty("badgeState", state)
        # Re-apply QSS so property-based rules take effect immediately.
        self.style().unpolish(self)
        self.style().polish(self)


class SectionCard(QWidget):
    """Standard section container: bold title above a bordered content area.

    Use card.content_layout (QVBoxLayout) to add child widgets or layouts.
    Pass collapsible=True to get a chevron toggle that hides/shows the content.
    """

    def __init__(self, title: str, collapsible: bool = False, collapsed: bool = False, parent=None):
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # Header row
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(2, 0, 2, 0)
        header_layout.setSpacing(6)

        title_lbl = QLabel(title)
        title_lbl.setObjectName("sectionCardTitle")
        header_layout.addWidget(title_lbl)
        header_layout.addStretch()

        self._toggle_btn = None
        if collapsible:
            self._toggle_btn = QPushButton("▼" if not collapsed else "▶")
            self._toggle_btn.setObjectName("sectionCardToggle")
            self._toggle_btn.setCheckable(True)
            self._toggle_btn.setChecked(not collapsed)
            self._toggle_btn.setFixedSize(QSize(24, 24))
            self._toggle_btn.clicked.connect(self._on_toggle)
            header_layout.addWidget(self._toggle_btn)

        outer.addWidget(header, 0)

        # Content area — stretch=1 so it fills the card when the card is stretched
        # by its parent, keeping items pinned to the top inside the border.
        self._content = QWidget()
        self._content.setObjectName("sectionCardContent")
        self.content_layout = QVBoxLayout(self._content)
        self.content_layout.setContentsMargins(12, 12, 12, 12)
        self.content_layout.setSpacing(8)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        outer.addWidget(self._content, 1)

        if collapsible:
            self._content.setVisible(not collapsed)

    def _on_toggle(self, checked: bool):
        self._content.setVisible(checked)
        self._toggle_btn.setText("▼" if checked else "▶")


class ParameterSlider(QWidget):
    """Horizontal slider linked bidirectionally to a QDoubleSpinBox."""

    value_changed = pyqtSignal(float)

    def __init__(self, label, min_val, max_val, default, decimals=2, step=0.01, parent=None):
        super().__init__(parent)
        self._factor = 10 ** decimals
        self._syncing = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        lbl = QLabel(label)
        lbl.setFixedWidth(140)
        layout.addWidget(lbl)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(int(min_val * self._factor))
        self.slider.setMaximum(int(max_val * self._factor))
        self.slider.setValue(int(default * self._factor))
        layout.addWidget(self.slider, 1)

        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setMinimum(min_val)
        self.spin.setMaximum(max_val)
        self.spin.setSingleStep(step)
        self.spin.setValue(default)
        self.spin.setFixedWidth(72)
        layout.addWidget(self.spin)

        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)

    def _on_slider(self, int_val: int):
        if self._syncing:
            return
        self._syncing = True
        fval = int_val / self._factor
        self.spin.setValue(fval)
        self._syncing = False
        self.value_changed.emit(fval)

    def _on_spin(self, fval: float):
        if self._syncing:
            return
        self._syncing = True
        self.slider.setValue(int(fval * self._factor))
        self._syncing = False
        self.value_changed.emit(fval)

    def value(self) -> float:
        return self.spin.value()

    def set_value(self, val: float):
        self.spin.setValue(val)
