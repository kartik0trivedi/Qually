import logging
import os
from pathlib import Path

from PyQt6.QtWidgets import (
    QLabel, QMainWindow, QMessageBox, QProgressBar, QVBoxLayout, QWidget,
)

import qually_tool
from ui.data_file_manager import DataFileManager
from ui.settings_manager import SettingsManager
from ui.welcome_screen import WelcomeScreen
from ui.workspace_shell import WorkspaceShell

log_folder = Path("logs")
log_folder.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_folder / "qually_gui.log"),
        logging.StreamHandler(),
    ],
)
gui_logger = logging.getLogger("QuallyGUI")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qually")
        self.setMinimumSize(1200, 800)

        self.project_folder = None
        self.data_manager = DataFileManager()
        self.settings_manager = None
        self.settings = {}
        self._workspace = None

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.show_welcome_screen()

    def show_welcome_screen(self):
        gui_logger.info("Displaying welcome screen.")
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        welcome_screen = WelcomeScreen(self)
        welcome_screen.project_selected.connect(self.open_project)
        self.layout.addWidget(welcome_screen)

    def open_project(self, project_folder):
        self.project_folder = project_folder
        self._setup_workspace()

    def _setup_workspace(self):
        if not self.project_folder:
            QMessageBox.warning(self, "Warning", "Please select a valid project folder first.")
            return

        gui_logger.info(f"Setting up workspace for project: {self.project_folder}")

        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        try:
            folder_path = Path(self.project_folder)
            (folder_path / "results").mkdir(parents=True, exist_ok=True)
            (folder_path / "logs").mkdir(parents=True, exist_ok=True)
        except Exception as mkdir_e:
            gui_logger.warning(f"Could not create subdirectories in {folder_path}: {mkdir_e}")
            QMessageBox.warning(
                self, "Warning",
                f"Could not create required subdirectories in project folder.\nError: {str(mkdir_e)}",
            )

        try:
            experiment_manager = qually_tool.ExperimentManager(self.project_folder)
            api_key_manager = experiment_manager.api_key_manager
            prompt_manager = experiment_manager.prompt_manager
            prompt_manager.data_manager = self.data_manager

            self.settings_manager = SettingsManager(Path(self.project_folder))
            self.settings = self.settings_manager.load_settings()

            self._workspace = WorkspaceShell(
                experiment_manager, api_key_manager, prompt_manager,
                self.data_manager, self.project_folder, self,
            )
            self._workspace.status_message.connect(self._show_status)
            self.layout.addWidget(self._workspace)

            self.statusBar().showMessage("Ready")

            self.progress_bar = QProgressBar()
            self.progress_bar.setMaximum(100)
            self.progress_bar.setValue(0)
            self.progress_bar.setTextVisible(False)
            self.progress_bar.setFixedWidth(200)
            self.progress_bar.hide()
            self.statusBar().addPermanentWidget(self.progress_bar)

            self.folder_label = QLabel(f"Project: {os.path.basename(self.project_folder)}")
            self.statusBar().addPermanentWidget(self.folder_label)

        except Exception as e:
            gui_logger.error(f"Failed to initialize workspace: {e}")
            QMessageBox.critical(self, "Error", f"Failed to initialize workspace:\n{str(e)}")

    def _show_status(self, message, timeout):
        self.statusBar().showMessage(message, timeout)

    def closeEvent(self, event):
        if self.settings_manager:
            self.settings_manager.save_settings(self.settings)
        if self._workspace:
            self._workspace.data_files_page.cleanup_temp_file()
        event.accept()
