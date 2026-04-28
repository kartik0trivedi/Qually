import logging

from PyQt6.QtWidgets import QMessageBox, QTabWidget, QVBoxLayout, QWidget
from PyQt6.QtCore import QTimer, pyqtSignal

from ui.api_keys_page import ApiKeysPage
from ui.data_files_page import DataFilesPage
from ui.experiments_page import ExperimentsPage
from ui.prompts_page import PromptsPage
from ui.results_page import ResultsPage

logger = logging.getLogger("QuallyGUI")


class WorkspaceShell(QWidget):
    """Tab container and cross-page signal hub for an open project."""

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

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()

        # Create all page objects before wiring cross-page signals
        api_keys_page = ApiKeysPage(self.api_key_manager, self)
        self.data_files_page = DataFilesPage(self.data_manager, self.project_folder, self)
        self.prompts_page = PromptsPage(self.prompt_manager, self.data_manager, self.project_folder, self)
        self.experiments_page = ExperimentsPage(
            self.experiment_manager, self.api_key_manager,
            self.prompt_manager, self.data_manager, self,
        )
        self.results_page = ResultsPage(self.experiment_manager, self.project_folder, self)

        # Wire cross-page signals — this is the only place these connections are made
        api_keys_page.keys_saved.connect(self.experiments_page.refresh_providers)
        self.data_files_page.data_changed.connect(self.prompts_page.refresh_data_status)
        self.data_files_page.data_changed.connect(self.prompts_page.load_prompts)
        self.experiments_page.experiment_run_complete.connect(self.results_page.refresh)
        self.experiments_page.status_message.connect(self.status_message)

        self.tabs.addTab(api_keys_page, "API Keys")
        self.tabs.addTab(self.data_files_page, "Data Files")
        self.tabs.addTab(self.prompts_page, "Prompts")
        self.tabs.addTab(self.experiments_page, "Experiments")
        self.tabs.addTab(self.results_page, "Results")

        layout.addWidget(self.tabs)

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
                self.tabs.setCurrentIndex(0)
