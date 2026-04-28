import logging
import time
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
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
        self.setObjectName("welcomeScreen")
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        card = QWidget()
        card.setObjectName("welcomeCard")
        card.setFixedWidth(520)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(16)

        # Logo
        logo_lbl = QLabel()
        img_path = resources_path / "icon.png"
        pixmap = QPixmap(str(img_path))
        if not pixmap.isNull():
            logo_lbl.setPixmap(
                pixmap.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
            )
        else:
            gui_logger.warning(f"Logo not found: {img_path}")
            logo_lbl.setText("Q")
        logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo_lbl)

        # Title / subtitle
        title_lbl = QLabel("Qually")
        title_lbl.setObjectName("welcomeTitle")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_lbl)

        subtitle_lbl = QLabel("LLM audit workspace for research projects")
        subtitle_lbl.setObjectName("welcomeSubtitle")
        subtitle_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle_lbl)

        layout.addSpacing(8)

        # Project folder field
        folder_field_lbl = QLabel("Project folder")
        folder_field_lbl.setObjectName("welcomeFieldLabel")
        layout.addWidget(folder_field_lbl)

        folder_row = QHBoxLayout()
        self.folder_path_input = QLineEdit()
        self.folder_path_input.setReadOnly(True)
        self.folder_path_input.setPlaceholderText("No folder selected")
        self.folder_path_input.setObjectName("welcomeFolderInput")

        browse_btn = QPushButton("Browse")
        browse_btn.setObjectName("welcomeBrowseButton")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self.select_project_folder)

        folder_row.addWidget(self.folder_path_input, 1)
        folder_row.addWidget(browse_btn)
        layout.addLayout(folder_row)

        helper_lbl = QLabel(
            "Your project folder stores encrypted API keys, prompts, experiments, and results."
        )
        helper_lbl.setObjectName("welcomeHelperText")
        helper_lbl.setWordWrap(True)
        layout.addWidget(helper_lbl)

        layout.addSpacing(8)

        self.open_btn = QPushButton("Open Project")
        self.open_btn.setObjectName("welcomeOpenButton")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._emit_project_selected)
        layout.addWidget(self.open_btn)

        outer.addWidget(card)

    def select_project_folder(self):
        gui_logger.info("Opening project folder selection dialog.")
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Project Folder",
            str(Path.home()),
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )
        if not folder:
            return

        folder_path = Path(folder)
        try:
            test_file = folder_path / f"qually_write_test_{int(time.time())}.txt"
            test_file.write_text("test")
            test_file.unlink()

            self.project_folder = str(folder_path)
            self.folder_path_input.setText(self.project_folder)
            self.open_btn.setEnabled(True)
            gui_logger.info(f"Project folder validated: {self.project_folder}")

        except Exception as e:
            gui_logger.error(f"Folder not writable: {e}")
            QMessageBox.critical(
                self,
                "Folder Error",
                f"The selected folder is not writable or accessible:\n\n{folder}\n\nError: {str(e)}",
            )
            self.project_folder = None
            self.folder_path_input.clear()
            self.open_btn.setEnabled(False)

    def _emit_project_selected(self):
        if self.project_folder:
            self.project_selected.emit(self.project_folder)
