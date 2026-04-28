import csv
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Dict

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


gui_logger = logging.getLogger("QuallyGUI")


class ResultsPage(QWidget):
    def __init__(self, experiment_manager, project_folder, parent=None):
        super().__init__(parent)
        self.experiment_manager = experiment_manager
        self.project_folder = project_folder
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        file_list_widget = QWidget()
        file_list_layout = QVBoxLayout(file_list_widget)
        file_list_layout.setContentsMargins(0, 0, 0, 0)

        file_list_label = QLabel("Result Files:")
        self.results_file_list = QListWidget()
        self.results_file_list.currentItemChanged.connect(self.display_selected_result_file)

        refresh_files_button = QPushButton("Refresh File List")
        refresh_files_button.setIcon(QIcon.fromTheme("view-refresh"))
        refresh_files_button.clicked.connect(self.refresh)

        file_list_layout.addWidget(file_list_label)
        file_list_layout.addWidget(self.results_file_list)
        file_list_layout.addWidget(refresh_files_button)

        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        details_layout.setContentsMargins(0, 0, 0, 0)

        exp_info_group = QFrame()
        exp_info_group.setFrameShape(QFrame.Shape.StyledPanel)
        exp_info_layout = QVBoxLayout(exp_info_group)
        self.result_exp_name_label = QLabel("Experiment: N/A")
        self.result_exp_name_label.setStyleSheet("font-weight: bold;")
        self.result_exp_id_label = QLabel("ID: N/A")
        self.result_timestamp_label = QLabel("Timestamp: N/A")
        exp_info_layout.addWidget(self.result_exp_name_label)
        exp_info_layout.addWidget(self.result_exp_id_label)
        exp_info_layout.addWidget(self.result_timestamp_label)
        details_layout.addWidget(exp_info_group)

        self.results_detail_table = QTableWidget()
        self.results_detail_table.setColumnCount(8)
        self.results_detail_table.setHorizontalHeaderLabels(
            ["Prompt ID", "Experiment", "Condition", "Provider", "Model", "Response", "Duration (s)", "Error"]
        )
        self.results_detail_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.results_detail_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_detail_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_detail_table.verticalHeader().setVisible(False)
        self.results_detail_table.setWordWrap(True)
        self.results_detail_table.resizeRowsToContents()
        details_layout.addWidget(self.results_detail_table)

        export_button = QPushButton("Export Selected Result to CSV")
        export_button.setIcon(QIcon.fromTheme("document-save-as"))
        export_button.clicked.connect(self.export_selected_result)
        details_layout.addWidget(export_button, 0, Qt.AlignmentFlag.AlignRight)

        splitter.addWidget(file_list_widget)
        splitter.addWidget(details_widget)
        splitter.setSizes([300, 900])
        layout.addWidget(splitter)

    def refresh(self):
        """Refresh the list of JSON result files."""
        if not self.experiment_manager:
            return

        gui_logger.info("Refreshing result file list.")
        self.results_file_list.clear()
        self.results_detail_table.setRowCount(0)
        self.result_exp_name_label.setText("Experiment: N/A")
        self.result_exp_id_label.setText("ID: N/A")
        self.result_timestamp_label.setText("Timestamp: N/A")

        try:
            results_folder = self.experiment_manager.results_folder
            result_files_paths = sorted(
                results_folder.glob("results_*.json"),
                key=os.path.getmtime,
                reverse=True,
            )
            if not result_files_paths:
                self.results_file_list.addItem("No result files found.")
                self.results_file_list.setEnabled(False)
                return

            self.results_file_list.setEnabled(True)
            for path in result_files_paths:
                mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
                item = QListWidgetItem(f"{path.name} ({mod_time})")
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                self.results_file_list.addItem(item)
            gui_logger.info(f"Found {len(result_files_paths)} result files.")

        except Exception as e:
            gui_logger.error(f"Error listing result files: {e}", exc_info=True)
            self.results_file_list.addItem("Error listing files.")
            self.results_file_list.setEnabled(False)
            QMessageBox.critical(self, "File Error", f"Could not list result files.\nError: {e}")

    def display_selected_result_file(self, current_item: QListWidgetItem, previous_item: QListWidgetItem = None):
        """Load and display the details of the selected result JSON file."""
        if not current_item:
            self.results_detail_table.setRowCount(0)
            return

        file_path_str = current_item.data(Qt.ItemDataRole.UserRole)
        if not file_path_str:
            return

        file_path = Path(file_path_str)
        gui_logger.info(f"Displaying results from: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                results_data = json.load(f)

            experiment_name = results_data.get("experiment_name", "N/A")
            self.result_exp_name_label.setText(f"Experiment: {experiment_name}")
            self.result_exp_id_label.setText(f"ID: {results_data.get('experiment_id', 'N/A')}")
            ts = results_data.get("timestamp", "N/A")
            try:
                ts_dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
                ts_display = ts_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            except Exception:
                ts_display = ts
            self.result_timestamp_label.setText(f"Timestamp: {ts_display}")

            self.results_detail_table.setColumnCount(8)
            self.results_detail_table.setHorizontalHeaderLabels(
                ["Prompt ID", "Experiment", "Condition", "Provider", "Model", "Response", "Duration (s)", "Error"]
            )
            self.results_detail_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)

            self.results_detail_table.setRowCount(0)
            all_rows_data = []
            conditions_info = results_data.get("conditions", {})
            prompt_results = results_data.get("prompt_results", {})

            for prompt_id, prompt_data in prompt_results.items():
                for condition_id, resp_data in prompt_data.get("condition_results", {}).items():
                    condition_details = conditions_info.get(condition_id, {})
                    row_data = {
                        "prompt_id": prompt_id,
                        "experiment": experiment_name,
                        "condition": condition_details.get("name", "N/A"),
                        "provider": condition_details.get("provider", "N/A").capitalize(),
                        "model": condition_details.get("model", "N/A"),
                        "response": resp_data.get("text", "N/A"),
                        "duration": (
                            f"{resp_data.get('duration_seconds', 'N/A'):.2f}"
                            if resp_data.get("duration_seconds") is not None
                            else "N/A"
                        ),
                        "error": resp_data.get("error", ""),
                    }
                    all_rows_data.append(row_data)

            self.results_detail_table.setRowCount(len(all_rows_data))
            for i, row in enumerate(all_rows_data):
                self.results_detail_table.setItem(i, 0, QTableWidgetItem(row["prompt_id"]))
                self.results_detail_table.setItem(i, 1, QTableWidgetItem(row["experiment"]))
                self.results_detail_table.setItem(i, 2, QTableWidgetItem(row["condition"]))
                self.results_detail_table.setItem(i, 3, QTableWidgetItem(row["provider"]))
                self.results_detail_table.setItem(i, 4, QTableWidgetItem(row["model"]))
                self.results_detail_table.setItem(i, 5, QTableWidgetItem(row["response"]))
                self.results_detail_table.setItem(i, 6, QTableWidgetItem(row["duration"]))
                error_item = QTableWidgetItem(row["error"])
                if row["error"]:
                    error_item.setForeground(Qt.GlobalColor.red)
                self.results_detail_table.setItem(i, 7, error_item)

            self.results_detail_table.resizeRowsToContents()

        except FileNotFoundError:
            gui_logger.error(f"Result file selected but not found: {file_path}")
            QMessageBox.critical(self, "File Error", f"Could not find the selected result file:\n{file_path}")
            self.results_detail_table.setRowCount(0)
        except json.JSONDecodeError:
            gui_logger.error(f"Error decoding JSON from result file: {file_path}")
            QMessageBox.critical(self, "File Error", f"The selected result file is not valid JSON:\n{file_path}")
            self.results_detail_table.setRowCount(0)
        except Exception as e:
            gui_logger.error(f"Error loading results from {file_path}: {e}", exc_info=True)
            QMessageBox.critical(self, "Load Error", f"An error occurred loading results:\n{e}")
            self.results_detail_table.setRowCount(0)

    def export_selected_result(self):
        """Export the selected result to CSV."""
        selected_items = self.results_file_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a result file to export.")
            return

        selected_item = selected_items[0]
        file_path = selected_item.data(Qt.ItemDataRole.UserRole)

        reply = QMessageBox.question(
            self,
            "Export to CSV",
            "Would you like to export this result to CSV format?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    results = json.load(f)

                csv_path = self.experiment_manager.export_results_csv(results)

                if csv_path:
                    QMessageBox.information(self, "Export Successful", f"Results exported to CSV:\n{csv_path}")
                else:
                    QMessageBox.warning(self, "Export Failed", "Failed to export results to CSV.")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error exporting results: {str(e)}")

    def export_blinded_results(self, results: Dict) -> str:
        """Export results to blinded CSV file for analysis."""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Blinded Results",
                str(Path(self.project_folder) / "blinded_results.csv"),
                "CSV Files (*.csv)",
            )

            if not file_path:
                return ""

            condition_mapping = {}
            for cond_id, cond_data in results["conditions"].items():
                condition_mapping[cond_id] = f"Condition_{len(condition_mapping) + 1}"

            with open(file_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Prompt_ID", "Response_ID", "Condition", "Response"])

                for prompt_result in results["prompt_results"]:
                    for response in prompt_result["responses"]:
                        writer.writerow(
                            [
                                prompt_result["prompt_id"],
                                response["response_id"],
                                condition_mapping[response["condition_id"]],
                                response["text"],
                            ]
                        )

            mapping_file = str(Path(file_path).with_suffix(".mapping.json"))
            with open(mapping_file, "w") as f:
                json.dump(condition_mapping, f, indent=2)

            QMessageBox.information(self, "Success", f"Blinded results exported successfully to:\n{file_path}")
            gui_logger.info(f"Blinded results exported to CSV: {file_path}")
            return file_path

        except Exception as e:
            gui_logger.error(f"Failed to export blinded results: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to export blinded results: {str(e)}")
            return ""
