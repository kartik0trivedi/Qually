import logging
import time
from pathlib import Path

import pandas as pd
from PyQt6.QtCore import Qt
import qtawesome as qta
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.widgets import SectionHeader

gui_logger = logging.getLogger("QuallyGUI")


class PromptsPage(QWidget):
    def __init__(self, prompt_manager, data_manager, project_folder, parent=None):
        super().__init__(parent)
        self.prompt_manager = prompt_manager
        self.data_manager = data_manager
        self.project_folder = project_folder
        self._build_ui()
        self.refresh_data_status()
        self.load_prompts()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        layout.addWidget(SectionHeader(
            "Prompts",
            "Build prompts from your data file and manage the prompt library.",
        ))

        self.data_warning_label = QLabel()
        self.data_warning_label.setStyleSheet(
            "color: #856404; background-color: #fff3cd; padding: 10px; "
            "border: 1px solid #ffeeba; border-radius: 4px;"
        )
        self.data_warning_label.setWordWrap(True)
        layout.addWidget(self.data_warning_label)

        self.data_info_label = QLabel()
        self.data_info_label.setStyleSheet(
            "color: #155724; background-color: #d4edda; padding: 10px; "
            "border: 1px solid #c3e6cb; border-radius: 4px;"
        )
        self.data_info_label.setWordWrap(True)
        layout.addWidget(self.data_info_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        form_layout.setContentsMargins(10, 10, 10, 10)
        form_layout.setSpacing(10)

        help_text = QLabel(
            "When a prompt is used, the text will be combined in this order:\n"
            "1. System Prompt (optional) - Sets the behavior of the model\n"
            "2. Prepend Text + Data Text - Text added before each data row\n"
            "3. Prompt Text - The main instruction for the model\n\n"
            "Example: If your data contains 'The sky is blue', and you set:\n"
            "Prepend: 'Consider this text: '\n"
            "Prompt: 'What color is mentioned?'\n"
            "The model will see: 'Consider this text: The sky is blue. What color is mentioned?'"
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet(
            """
            QLabel {
                padding: 10px;
                background-color: #e9ecef;
                border: 1px solid #ced4da;
                border-radius: 5px;
                margin-bottom: 10px;
                font-size: 13px;
            }
        """
        )
        form_layout.addWidget(help_text)

        self.system_prompt_edit = QTextEdit()
        self.system_prompt_edit.setPlaceholderText(
            "Optional: Set the model's behavior. Example: 'You are a helpful assistant that answers questions about text.'"
        )
        self.system_prompt_edit.setMaximumHeight(100)
        form_layout.addWidget(QLabel("System Prompt:"))
        form_layout.addWidget(self.system_prompt_edit)

        self.prepend_edit = QTextEdit()
        self.prepend_edit.setPlaceholderText(
            "Optional: Text to add before each data row. Example: 'Consider this text: '"
        )
        self.prepend_edit.setMaximumHeight(100)
        form_layout.addWidget(QLabel("Prepend Text:"))
        form_layout.addWidget(self.prepend_edit)

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText(
            "Required: The main instruction for the model. Example: 'What is the main topic of this text?'"
        )
        self.prompt_edit.setMaximumHeight(100)
        form_layout.addWidget(QLabel("Prompt:"))
        form_layout.addWidget(self.prompt_edit)

        self.create_from_data_button = QPushButton("Create Prompts from Data File")
        self.create_from_data_button.setIcon(qta.icon("fa5s.plus", color="white"))
        self.create_from_data_button.clicked.connect(self.create_prompts_from_data)
        form_layout.addWidget(self.create_from_data_button, 0, Qt.AlignmentFlag.AlignRight)
        form_layout.addStretch()

        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setContentsMargins(10, 10, 10, 10)
        table_layout.setSpacing(10)

        self.prompts_table = QTableWidget()
        self.prompts_table.setAlternatingRowColors(True)
        self.prompts_table.setColumnCount(6)
        self.prompts_table.setHorizontalHeaderLabels(
            ["ID", "Data ID", "System Prompt", "Prepend Text", "Prompt Text", "Tags"]
        )
        self.prompts_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.prompts_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.prompts_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.prompts_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.prompts_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.prompts_table.verticalHeader().setVisible(False)
        table_layout.addWidget(self.prompts_table)

        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 5, 0, 0)

        delete_selected_button = QPushButton("Delete Selected")
        delete_selected_button.setIcon(qta.icon("fa5s.trash-alt", color="white"))
        delete_selected_button.clicked.connect(self.delete_selected_prompts)

        clear_all_button = QPushButton("Clear All")
        clear_all_button.setIcon(qta.icon("fa5s.times-circle", color="white"))
        clear_all_button.clicked.connect(self.clear_all_prompts)

        export_button = QPushButton("Export Prompts")
        export_button.setIcon(qta.icon("fa5s.save", color="white"))
        export_button.clicked.connect(self.export_prompts)

        buttons_layout.addWidget(delete_selected_button)
        buttons_layout.addWidget(clear_all_button)
        buttons_layout.addStretch()
        buttons_layout.addWidget(export_button)
        table_layout.addWidget(buttons_widget)

        splitter.addWidget(form_widget)
        splitter.addWidget(table_widget)
        splitter.setSizes([300, 600])

    def refresh_data_status(self):
        has_data = self.data_manager and self.data_manager.has_data()
        self.data_warning_label.setVisible(not has_data)
        self.data_info_label.setVisible(has_data)
        self.create_from_data_button.setEnabled(has_data)

        if not has_data:
            self.data_warning_label.setText(
                "⚠️ Warning: No data file loaded. Please import data in the Data Files tab before working with prompts."
            )
            return

        info = self.data_manager.file_info
        self.data_info_label.setText(
            f"📊 Data File Info:\n"
            f"File: {Path(info['file_path']).name}\n"
            f"Text Column: {info.get('text_column', 'Not selected')}\n"
            f"Total Rows: {info['total_rows']}\n\n"
            f"To create prompts for each row in your data file:\n"
            f"1. Set the system prompt (optional)\n"
            f"2. Set the prepend text (optional)\n"
            f"3. Set the prompt text\n"
            f"4. Click 'Create Prompts from Data File'"
        )

    def create_prompts_from_data(self):
        """Create prompts from the loaded data file."""
        if not self.prompt_manager:
            QMessageBox.critical(self, "Error", "Prompt manager not initialized.")
            return

        if not self.data_manager or not self.data_manager.has_data():
            QMessageBox.warning(self, "No Data", "Please load a data file first.")
            return

        if not self.data_manager.file_info.get("text_column"):
            QMessageBox.warning(
                self,
                "No Text Column Selected",
                "Please select a text column in the Data Files tab first.\n\n"
                "Available columns:\n" + "\n".join(f"- {col}" for col in self.data_manager.data.columns),
            )
            return

        if not self.data_manager.file_info.get("id_column"):
            QMessageBox.warning(self, "No ID Column Selected", "Please select an ID column in the Data Files tab first.")
            return

        try:
            system_prompt = self.system_prompt_edit.toPlainText().strip()
            prepend_text = self.prepend_edit.toPlainText().strip()
            prompt_text = self.prompt_edit.toPlainText().strip()

            if not prompt_text:
                QMessageBox.warning(self, "Error", "Prompt text cannot be empty.")
                return

            data = self.data_manager.data
            text_column = self.data_manager.file_info["text_column"]
            id_column = self.data_manager.file_info["id_column"]

            timestamp = int(time.time())
            for index, row in data.iterrows():
                text_content = str(row[text_column])
                data_id = str(row[id_column])
                prompt_id = f"prompt_{timestamp}_row_{index}"

                full_text = text_content
                if prepend_text:
                    full_text = f"{prepend_text} {text_content}"
                full_prompt = f"{full_text}\n{prompt_text}"

                self.prompt_manager.add_prompt(
                    prompt_id=prompt_id,
                    prompt_text=full_prompt,
                    system_prompt=system_prompt,
                    prepend_text=prepend_text,
                    tags=[f"data_id_{data_id}", f"row_{index}"],
                )

            if self.prompt_manager.export_prompts():
                self.system_prompt_edit.clear()
                self.prepend_edit.clear()
                self.prompt_edit.clear()
                self.load_prompts()

                QMessageBox.information(
                    self,
                    "Success",
                    f"Created prompts for all rows in the data file.\n"
                    f"Text column used: {text_column}\n"
                    f"ID column used: {id_column}\n"
                    f"Total prompts created: {len(data)}",
                )
                gui_logger.info("Created prompts from data file")
            else:
                QMessageBox.critical(self, "Error", "Failed to save prompts to file.")
                gui_logger.error("Failed to save prompts to file")

        except Exception as e:
            gui_logger.error(f"Error creating prompts from data: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to create prompts from data: {str(e)}")

    def delete_selected_prompts(self):
        """Delete the currently selected prompts after confirmation."""
        if not self.prompt_manager:
            return

        selected_rows = self.prompts_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select prompts to delete.")
            return

        prompt_ids = []
        for row in selected_rows:
            prompt_id = self.prompts_table.item(row.row(), 0).text()
            prompt_ids.append(prompt_id)

        if len(prompt_ids) == 1:
            message = f"Are you sure you want to delete prompt '{prompt_ids[0]}'?"
        else:
            message = f"Are you sure you want to delete {len(prompt_ids)} selected prompts?"

        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                for prompt_id in prompt_ids:
                    self.prompt_manager.delete_prompt(prompt_id)
                self.prompt_manager.export_prompts()
                self.load_prompts()
                QMessageBox.information(self, "Success", "Selected prompts deleted successfully.")
            except Exception as e:
                gui_logger.error(f"Failed to delete prompts: {e}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to delete prompts: {str(e)}")

    def clear_all_prompts(self):
        """Delete all prompts after confirmation."""
        if not self.prompt_manager:
            return

        reply = QMessageBox.question(
            self,
            "Confirm Clear All",
            "Are you sure you want to delete ALL prompts? This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.prompt_manager.clear_prompts()
                self.prompt_manager.export_prompts()
                self.load_prompts()
                QMessageBox.information(self, "Success", "All prompts deleted successfully.")
            except Exception as e:
                gui_logger.error(f"Failed to clear prompts: {e}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to clear prompts: {str(e)}")

    def export_prompts(self):
        """Export prompts to CSV file."""
        if not self.prompt_manager:
            QMessageBox.critical(self, "Error", "Prompt manager not initialized.")
            return

        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Prompts",
                str(Path(self.project_folder) / "prompts.csv"),
                "CSV Files (*.csv)",
            )

            if not file_path:
                return

            prompts = self.prompt_manager.list_prompts()
            if not prompts:
                QMessageBox.warning(self, "No Prompts", "There are no prompts to export.")
                return

            data = []
            for prompt in prompts:
                data.append(
                    {
                        "id": prompt["id"],
                        "system_prompt": prompt["system_prompt"],
                        "prompt_text": prompt["prompt_text"],
                    }
                )

            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)

            QMessageBox.information(self, "Success", f"Prompts exported successfully to:\n{file_path}")
            gui_logger.info(f"Prompts exported to CSV: {file_path}")

        except Exception as e:
            gui_logger.error(f"Failed to export prompts: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to export prompts: {str(e)}")

    def load_prompts(self):
        """Populate the prompts table from prompt_manager."""
        if not self.prompt_manager:
            return

        prompts = self.prompt_manager.list_prompts()
        self.prompts_table.setRowCount(len(prompts))

        for row, prompt in enumerate(prompts):
            prompt_id = prompt.get("id", "")
            self.prompts_table.setItem(row, 0, QTableWidgetItem(prompt_id))

            data_id = ""
            for tag in prompt.get("tags", []):
                if tag.startswith("data_id_"):
                    data_id = tag.replace("data_id_", "")
                    break

            self.prompts_table.setItem(row, 1, QTableWidgetItem(data_id))
            self.prompts_table.setItem(row, 2, QTableWidgetItem(prompt.get("system_prompt", "")))
            self.prompts_table.setItem(row, 3, QTableWidgetItem(prompt.get("prepend_text", "")))
            self.prompts_table.setItem(row, 4, QTableWidgetItem(prompt.get("prompt_text", "")))
            tags = ";".join(prompt.get("tags", []))
            self.prompts_table.setItem(row, 5, QTableWidgetItem(tags))
