import logging
import time
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.theme import resources_path


gui_logger = logging.getLogger("QuallyGUI")


class WelcomeScreen(QWidget):
    project_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project_folder = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)

        img_path = resources_path / "icon.png"
        logo_label = QLabel()
        logo_pixmap = QPixmap(str(img_path))
        if not logo_pixmap.isNull():
            scaled_pixmap = logo_pixmap.scaled(
                200,
                200,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            logo_label.setPixmap(scaled_pixmap)
        else:
            gui_logger.warning(f"Logo file not found or invalid: {img_path}")
            logo_label.setText("[Logo Not Found]")

        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo_label)

        welcome_label = QLabel("Welcome to Qually")
        welcome_label.setStyleSheet("font-size: 28px; font-weight: bold;")
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(welcome_label)

        description = QLabel(
            "Qually helps you audit and analyze LLM outputs for social science research.\n"
            "Please select a folder where your project files (prompts, experiments, results, keys) will be stored."
        )
        description.setStyleSheet("font-size: 14px; color: #555;")
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setWordWrap(True)
        layout.addWidget(description)

        folder_group = QFrame()
        folder_group.setFrameShape(QFrame.Shape.StyledPanel)
        folder_layout = QHBoxLayout(folder_group)

        self.folder_path_label = QLineEdit("No folder selected")
        self.folder_path_label.setReadOnly(True)
        self.folder_path_label.setStyleSheet(
            "padding: 5px; border: 1px solid #ccc; border-radius: 3px; background-color: #eee;"
        )
        folder_layout.addWidget(QLabel("Project Folder:"))
        folder_layout.addWidget(self.folder_path_label, 1)

        select_button = QPushButton("...")
        select_button.setToolTip("Select Project Folder")
        select_button.setFixedWidth(40)
        select_button.clicked.connect(self.select_project_folder)
        folder_layout.addWidget(select_button)

        layout.addWidget(folder_group)

        self.continue_button = QPushButton("Start Qually")
        self.continue_button.setStyleSheet("font-size: 16px; padding: 10px 20px;")
        self.continue_button.clicked.connect(self._emit_project_selected)
        self.continue_button.setEnabled(False)
        layout.addWidget(self.continue_button, 0, Qt.AlignmentFlag.AlignCenter)

    def select_project_folder(self):
        """Show folder selection dialog and validate writability."""
        gui_logger.info("Opening project folder selection dialog.")
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Project Folder",
            str(Path.home()),
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )

        if folder:
            folder_path = Path(folder)
            gui_logger.info(f"Folder selected: {folder_path}")
            try:
                test_file = folder_path / f"qually_write_test_{int(time.time())}.txt"
                with open(test_file, "w") as f:
                    f.write("test")
                test_file.unlink()

                self.project_folder = str(folder_path)
                self.folder_path_label.setText(self.project_folder)
                self.continue_button.setEnabled(True)
                gui_logger.info(f"Project folder set and validated: {self.project_folder}")

            except Exception as e:
                gui_logger.error(f"Selected folder '{folder}' is not writable: {e}")
                QMessageBox.critical(
                    self,
                    "Folder Error",
                    f"The selected folder is not writable or accessible:\n\n{folder}\n\n"
                    f"Error: {str(e)}\n\nPlease select a different folder.",
                )
                self.project_folder = None
                self.folder_path_label.setText("No folder selected")
                self.continue_button.setEnabled(False)

    def _emit_project_selected(self):
        if self.project_folder:
            self.project_selected.emit(self.project_folder)
