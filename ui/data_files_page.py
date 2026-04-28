import logging
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QComboBox,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


gui_logger = logging.getLogger("QuallyGUI")


class DataFilesPage(QWidget):
    data_changed = pyqtSignal()

    def __init__(self, data_manager, project_folder, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self.project_folder = project_folder
        self.last_temp_csv_path = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        file_group = QGroupBox("Data File Selection")
        file_layout = QHBoxLayout(file_group)

        self.file_path_input = QLineEdit()
        self.file_path_input.setReadOnly(True)
        browse_button = QPushButton("Browse")
        browse_button.setToolTip("Select a data file (CSV, JSON, or Excel)")
        browse_button.clicked.connect(self.browse_data_file)

        data_label = QLabel("Data File:")
        data_label.setToolTip(
            "Supported formats: CSV (.csv), JSON (.json), Excel (.xls, .xlsx).\n"
            "For JSON, use an array of objects or a single object.\n"
            "For Excel, the first sheet will be used."
        )
        info_icon = QPushButton()
        info_icon.setText("ⓘ")
        info_icon.setFixedWidth(24)
        info_icon.setStyleSheet("border: none; background: transparent; color: #666; font-weight: bold;")
        info_icon.setToolTip(data_label.toolTip())

        import_txt_button = QPushButton("Import Multiple Text Files")
        import_txt_button.setToolTip("Import multiple .txt files and convert to a table with ID, file name, and content.")
        import_txt_button.clicked.connect(self.import_multiple_text_files)

        file_layout.addWidget(data_label)
        file_layout.addWidget(self.file_path_input)
        file_layout.addWidget(browse_button)
        file_layout.addWidget(info_icon)
        file_layout.addWidget(import_txt_button)

        layout.addWidget(file_group)

        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.preview_table = QTableWidget()
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.preview_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.preview_table.verticalHeader().setVisible(False)
        preview_layout.addWidget(self.preview_table)

        layout.addWidget(preview_group)

        column_group = QGroupBox("Column Selection")
        column_layout = QVBoxLayout(column_group)

        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("ID Column:"))
        self.id_column_combo = QComboBox()
        self.id_column_combo.currentTextChanged.connect(self.update_id_column)
        id_layout.addWidget(self.id_column_combo)
        column_layout.addLayout(id_layout)

        text_layout = QHBoxLayout()
        text_layout.addWidget(QLabel("Text Column:"))
        self.text_column_combo = QComboBox()
        self.text_column_combo.currentTextChanged.connect(self.update_text_column)
        text_layout.addWidget(self.text_column_combo)
        column_layout.addLayout(text_layout)

        layout.addWidget(column_group)

    def import_multiple_text_files(self):
        """Import multiple .txt files, convert them to a table, and load as data."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Text Files",
            str(self.project_folder) if self.project_folder else str(Path.home()),
            "Text Files (*.txt)",
        )
        if not file_paths:
            return
        try:
            rows = []
            for idx, file_path in enumerate(file_paths):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                rows.append(
                    {
                        "id": f"doc_{idx+1:03d}",
                        "file_name": Path(file_path).name,
                        "content": content,
                    }
                )

            df = pd.DataFrame(rows)
            temp_dir = tempfile.gettempdir()
            temp_csv_path = os.path.join(temp_dir, f"qually_imported_texts_{int(time.time())}.csv")
            df.to_csv(temp_csv_path, index=False)
            self.last_temp_csv_path = temp_csv_path

            reply = QMessageBox.question(
                self,
                "Save as CSV?",
                "Would you like to save this table as a CSV in your project folder?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                save_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save CSV File",
                    str(Path(self.project_folder) / "imported_texts.csv"),
                    "CSV Files (*.csv)",
                )
                if save_path:
                    df.to_csv(save_path, index=False)
                    QMessageBox.information(self, "Saved", f"CSV saved to: {save_path}")

            preview_data, columns = self.data_manager.import_csv(temp_csv_path)
            self.file_path_input.setText(temp_csv_path)
            self._populate_preview(preview_data, columns)
            self._populate_column_selectors(columns, id_column="id", text_column="content")
            self.data_manager.file_info["id_column"] = "id"
            self.data_manager.file_info["text_column"] = "content"
            gui_logger.info(f"Imported {len(file_paths)} text files as data table.")
            self.data_changed.emit()
        except Exception as e:
            gui_logger.error(f"Failed to import text files: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to import text files: {str(e)}")

    def browse_data_file(self):
        """Open file dialog to select a data file and load its preview."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            str(self.project_folder) if self.project_folder else str(Path.home()),
            "Data Files (*.csv *.json *.xls *.xlsx)",
        )

        if file_path:
            try:
                file_extension = Path(file_path).suffix.lower()
                if file_extension == ".csv":
                    preview_data, columns = self.data_manager.import_csv(file_path)
                elif file_extension == ".json":
                    preview_data, columns = self.data_manager.import_json(file_path)
                elif file_extension in [".xls", ".xlsx"]:
                    preview_data, columns = self.data_manager.import_excel(file_path)
                else:
                    raise ValueError(f"Unsupported file type: {file_extension}")

                self.file_path_input.setText(file_path)
                self._populate_preview(preview_data, columns)

                id_column = None
                possible_id_cols = [col for col in columns if col.lower() == "id"]
                if possible_id_cols:
                    id_column = possible_id_cols[0]
                    self.data_manager.file_info["id_column"] = id_column

                text_column = None
                possible_text_cols = ["text", "content", "body", "main_text", "abstract", "title", "combined"]
                for col in possible_text_cols:
                    if col in columns:
                        text_column = col
                        self.data_manager.file_info["text_column"] = col
                        break

                self._populate_column_selectors(columns, id_column=id_column, text_column=text_column)

                gui_logger.info(f"Data file loaded and previewed: {file_path}")
                self.data_changed.emit()
            except Exception as e:
                gui_logger.error(f"Failed to load data file: {e}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to load data file: {str(e)}")

    def update_text_column(self, column_name):
        """Update the text column in file_info when selection changes."""
        if self.data_manager and self.data_manager.file_info:
            self.data_manager.file_info["text_column"] = column_name
            gui_logger.info(f"Updated text column to: {column_name}")
            self.data_changed.emit()

    def update_id_column(self, column_name):
        """Update the ID column in file_info when selection changes."""
        if self.data_manager and self.data_manager.file_info:
            self.data_manager.file_info["id_column"] = column_name
            gui_logger.info(f"Updated ID column to: {column_name}")
            self.data_changed.emit()

    def cleanup_temp_file(self):
        if self.last_temp_csv_path:
            try:
                if os.path.exists(self.last_temp_csv_path):
                    os.remove(self.last_temp_csv_path)
                    gui_logger.info(f"Deleted temp CSV: {self.last_temp_csv_path}")
            except Exception as e:
                gui_logger.warning(f"Failed to delete temp CSV: {e}")

    def _populate_preview(self, preview_data, columns):
        self.preview_table.setRowCount(len(preview_data))
        self.preview_table.setColumnCount(len(columns))
        self.preview_table.setHorizontalHeaderLabels(columns)
        for row in range(len(preview_data)):
            for col in range(len(columns)):
                item = QTableWidgetItem(str(preview_data.iloc[row, col]))
                self.preview_table.setItem(row, col, item)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def _populate_column_selectors(self, columns, id_column=None, text_column=None):
        self.id_column_combo.blockSignals(True)
        self.text_column_combo.blockSignals(True)
        self.id_column_combo.clear()
        self.text_column_combo.clear()
        self.id_column_combo.addItems(columns)
        self.text_column_combo.addItems(columns)
        if id_column:
            self.id_column_combo.setCurrentText(id_column)
        if text_column:
            self.text_column_combo.setCurrentText(text_column)
        self.id_column_combo.blockSignals(False)
        self.text_column_combo.blockSignals(False)
