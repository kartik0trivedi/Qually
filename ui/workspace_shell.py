import logging
import os

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ui.api_keys_page import ApiKeysPage
from ui.data_files_page import DataFilesPage
from ui.experiments_page import ExperimentsPage
from ui.prompts_page import PromptsPage
from ui.results_page import ResultsPage

logger = logging.getLogger("QuallyGUI")

_NAV_ITEMS = [
    ("Setup",        0),
    ("Data",         1),
    ("Prompts",      2),
    ("Experiments",  3),
    ("Results",      4),
]


class WorkspaceShell(QWidget):
    """Left-sidebar navigation shell containing all five workflow pages."""

    status_message = pyqtSignal(str, int)

    def __init__(self, experiment_manager, api_key_manager, prompt_manager,
                 data_manager, project_folder, parent=None):
        super().__init__(parent)
        self.experiment_manager = experiment_manager
        self.api_key_manager = api_key_manager
        self.prompt_manager = prompt_manager
        self.data_manager = data_manager
        self.project_folder = project_folder

        self._build_ui()
        QTimer.singleShot(100, self._check_api_keys_on_startup)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # Create all pages before wiring any cross-page signals.
        api_keys_page = ApiKeysPage(self.api_key_manager, self)
        self.data_files_page = DataFilesPage(self.data_manager, self.project_folder, self)
        self.prompts_page = PromptsPage(self.prompt_manager, self.data_manager, self.project_folder, self)
        self.experiments_page = ExperimentsPage(
            self.experiment_manager, self.api_key_manager,
            self.prompt_manager, self.data_manager, self,
        )
        self.results_page = ResultsPage(self.experiment_manager, self.project_folder, self)

        # Wire cross-page signals — single authoritative location.
        api_keys_page.keys_saved.connect(self.experiments_page.refresh_providers)
        self.data_files_page.data_changed.connect(self.prompts_page.refresh_data_status)
        self.data_files_page.data_changed.connect(self.prompts_page.load_prompts)
        self.experiments_page.experiment_run_complete.connect(self.results_page.refresh)
        self.experiments_page.status_message.connect(self.status_message)

        body_layout.addWidget(self._build_sidebar())

        # Pages stack — order must match _NAV_ITEMS indices.
        self._pages = QStackedWidget()
        self._pages.addWidget(api_keys_page)         # 0 – Setup
        self._pages.addWidget(self.data_files_page)  # 1 – Data
        self._pages.addWidget(self.prompts_page)     # 2 – Prompts
        self._pages.addWidget(self.experiments_page) # 3 – Experiments
        self._pages.addWidget(self.results_page)     # 4 – Results

        body_layout.addWidget(self._pages, 1)
        root.addWidget(body, 1)

        self._go_to(0)

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(180)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 20, 0, 20)
        layout.setSpacing(2)

        brand = QLabel("Qually")
        brand.setObjectName("sidebarBrand")
        brand.setContentsMargins(20, 0, 20, 16)
        layout.addWidget(brand)

        self._nav_btns: list[QPushButton] = []
        for label, idx in _NAV_ITEMS:
            btn = QPushButton(label)
            btn.setObjectName("navButton")
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.clicked.connect(lambda _, i=idx: self._go_to(i))
            layout.addWidget(btn)
            self._nav_btns.append(btn)

        layout.addStretch()

        project_name = os.path.basename(self.project_folder) if self.project_folder else ""
        if project_name:
            project_lbl = QLabel(project_name)
            project_lbl.setObjectName("sidebarProjectLabel")
            project_lbl.setWordWrap(True)
            layout.addWidget(project_lbl)

        return sidebar

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _go_to(self, index: int):
        self._pages.setCurrentIndex(index)
        for i, btn in enumerate(self._nav_btns):
            btn.setChecked(i == index)

    # ------------------------------------------------------------------
    # Startup check
    # ------------------------------------------------------------------

    def _check_api_keys_on_startup(self):
        if not self.api_key_manager.list_providers():
            logger.warning("No API keys found on startup.")
            reply = QMessageBox.information(
                self, "API Keys Needed",
                "No API keys found in your project.\n\n"
                "You need to add API keys to run experiments.\n"
                "Would you like to go to the API Keys tab now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._go_to(0)
