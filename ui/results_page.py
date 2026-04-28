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
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.widgets import EmptyState, SectionHeader

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
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        layout.addWidget(SectionHeader(
            "Results",
            "Browse and export experiment result files.",
        ))

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---- Left: file list ----
        file_list_widget = QWidget()
        file_list_layout = QVBoxLayout(file_list_widget)
        file_list_layout.setContentsMargins(0, 0, 0, 0)
        file_list_layout.setSpacing(8)

        file_list_layout.addWidget(QLabel("Result runs:"))

        # Stacked widget: empty state or file list
        self._list_stack = QStackedWidget()

        self._empty_list = EmptyState(
            "📂",
            "No results yet",
            "Run an experiment to generate result files.",
        )
        self._list_stack.addWidget(self._empty_list)  # index 0

        self.results_file_list = QListWidget()
        self.results_file_list.currentItemChanged.connect(self.display_selected_result_file)
        self._list_stack.addWidget(self.results_file_list)  # index 1

        file_list_layout.addWidget(self._list_stack, 1)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setIcon(QIcon.fromTheme("view-refresh"))
        refresh_btn.clicked.connect(self.refresh)
        file_list_layout.addWidget(refresh_btn)

        # ---- Right: result details ----
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(8)

        # Experiment info panel
        info_frame = QFrame()
        info_frame.setObjectName("resultInfoFrame")
        info_frame.setFrameShape(QFrame.Shape.StyledPanel)
        info_layout = QVBoxLayout(info_frame)
        info_layout.setSpacing(4)

        self.result_exp_name_label = QLabel("Experiment: —")
        self.result_exp_name_label.setObjectName("resultExpName")
        info_layout.addWidget(self.result_exp_name_label)

        self.result_exp_id_label = QLabel("ID: —")
        info_layout.addWidget(self.result_exp_id_label)

        self.result_timestamp_label = QLabel("Timestamp: —")
        info_layout.addWidget(self.result_timestamp_label)

        # Summary counters row
        counters_row = QHBoxLayout()
        self._counter_labels: dict[str, QLabel] = {}
        for key, display in [("prompts", "Prompts"), ("conditions", "Conditions"),
                              ("responses", "Responses"), ("errors", "Errors")]:
            lbl = QLabel(f"{display}: —")
            lbl.setObjectName("resultCounter")
            if key == "errors":
                lbl.setObjectName("resultCounterErrors")
            counters_row.addWidget(lbl)
            self._counter_labels[key] = lbl
        counters_row.addStretch()
        info_layout.addLayout(counters_row)

        details_layout.addWidget(info_frame)

        # Detail table
        self.results_detail_table = QTableWidget()
        self.results_detail_table.setColumnCount(8)
        self.results_detail_table.setHorizontalHeaderLabels(
            ["Prompt ID", "Experiment", "Condition", "Provider", "Model",
             "Response", "Duration (s)", "Error"]
        )
        self.results_detail_table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeMode.Stretch
        )
        self.results_detail_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_detail_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_detail_table.verticalHeader().setVisible(False)
        self.results_detail_table.setWordWrap(True)
        details_layout.addWidget(self.results_detail_table, 1)

        export_btn = QPushButton("Export to CSV")
        export_btn.setIcon(QIcon.fromTheme("document-save-as"))
        export_btn.clicked.connect(self.export_selected_result)
        details_layout.addWidget(export_btn, 0, Qt.AlignmentFlag.AlignRight)

        splitter.addWidget(file_list_widget)
        splitter.addWidget(details_widget)
        splitter.setSizes([280, 920])
        layout.addWidget(splitter, 1)

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self):
        if not self.experiment_manager:
            return

        gui_logger.info("Refreshing result file list.")
        self.results_file_list.clear()
        self._reset_detail_panel()

        try:
            results_folder = self.experiment_manager.results_folder
            paths = sorted(
                results_folder.glob("results_*.json"),
                key=os.path.getmtime,
                reverse=True,
            )

            if not paths:
                self._list_stack.setCurrentIndex(0)  # empty state
                return

            self._list_stack.setCurrentIndex(1)  # file list
            for path in paths:
                mod_time = datetime.datetime.fromtimestamp(
                    os.path.getmtime(path)
                ).strftime("%Y-%m-%d %H:%M")
                item = QListWidgetItem(f"{path.name}\n{mod_time}")
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                self.results_file_list.addItem(item)

            gui_logger.info(f"Found {len(paths)} result files.")

        except Exception as e:
            gui_logger.error(f"Error listing result files: {e}", exc_info=True)
            self._list_stack.setCurrentIndex(0)
            QMessageBox.critical(self, "File Error", f"Could not list result files.\nError: {e}")

    def _reset_detail_panel(self):
        self.result_exp_name_label.setText("Experiment: —")
        self.result_exp_id_label.setText("ID: —")
        self.result_timestamp_label.setText("Timestamp: —")
        for lbl in self._counter_labels.values():
            text = lbl.text().split(":")[0]
            lbl.setText(f"{text}: —")
        self.results_detail_table.setRowCount(0)

    # ------------------------------------------------------------------
    # Display selected result
    # ------------------------------------------------------------------

    def display_selected_result_file(self, current_item: QListWidgetItem, _=None):
        if not current_item:
            self._reset_detail_panel()
            return

        file_path_str = current_item.data(Qt.ItemDataRole.UserRole)
        if not file_path_str:
            return

        file_path = Path(file_path_str)
        gui_logger.info(f"Displaying results from: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                results_data = json.load(f)

            exp_name = results_data.get("experiment_name", "—")
            self.result_exp_name_label.setText(f"Experiment: {exp_name}")
            self.result_exp_id_label.setText(f"ID: {results_data.get('experiment_id', '—')}")

            ts = results_data.get("timestamp", "—")
            try:
                ts_dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
                ts_display = ts_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            except Exception:
                ts_display = ts
            self.result_timestamp_label.setText(f"Timestamp: {ts_display}")

            # Build row data and compute counters
            conditions_info = results_data.get("conditions", {})
            prompt_results = results_data.get("prompt_results", {})

            all_rows = []
            error_count = 0
            for prompt_id, prompt_data in prompt_results.items():
                for cond_id, resp_data in prompt_data.get("condition_results", {}).items():
                    cond = conditions_info.get(cond_id, {})
                    error_text = resp_data.get("error", "")
                    if error_text:
                        error_count += 1
                    all_rows.append({
                        "prompt_id": prompt_id,
                        "experiment": exp_name,
                        "condition": cond.get("name", "—"),
                        "provider": cond.get("provider", "—").capitalize(),
                        "model": cond.get("model", "—"),
                        "response": resp_data.get("text", "—"),
                        "duration": (
                            f"{resp_data['duration_seconds']:.2f}"
                            if resp_data.get("duration_seconds") is not None else "—"
                        ),
                        "error": error_text,
                    })

            n_prompts = len(prompt_results)
            n_conditions = len(conditions_info)
            n_responses = len(all_rows)

            self._counter_labels["prompts"].setText(f"Prompts: {n_prompts}")
            self._counter_labels["conditions"].setText(f"Conditions: {n_conditions}")
            self._counter_labels["responses"].setText(f"Responses: {n_responses}")
            self._counter_labels["errors"].setText(f"Errors: {error_count}")

            # Populate table
            self.results_detail_table.setRowCount(len(all_rows))
            for i, row in enumerate(all_rows):
                self.results_detail_table.setItem(i, 0, QTableWidgetItem(row["prompt_id"]))
                self.results_detail_table.setItem(i, 1, QTableWidgetItem(row["experiment"]))
                self.results_detail_table.setItem(i, 2, QTableWidgetItem(row["condition"]))
                self.results_detail_table.setItem(i, 3, QTableWidgetItem(row["provider"]))
                self.results_detail_table.setItem(i, 4, QTableWidgetItem(row["model"]))
                self.results_detail_table.setItem(i, 5, QTableWidgetItem(row["response"]))
                self.results_detail_table.setItem(i, 6, QTableWidgetItem(row["duration"]))
                err_item = QTableWidgetItem(row["error"])
                if row["error"]:
                    err_item.setForeground(Qt.GlobalColor.red)
                self.results_detail_table.setItem(i, 7, err_item)

            self.results_detail_table.resizeRowsToContents()

        except FileNotFoundError:
            gui_logger.error(f"Result file not found: {file_path}")
            QMessageBox.critical(self, "File Error", f"Result file not found:\n{file_path}")
            self._reset_detail_panel()
        except json.JSONDecodeError:
            gui_logger.error(f"Invalid JSON in: {file_path}")
            QMessageBox.critical(self, "File Error", f"Invalid JSON in result file:\n{file_path}")
            self._reset_detail_panel()
        except Exception as e:
            gui_logger.error(f"Error loading results from {file_path}: {e}", exc_info=True)
            QMessageBox.critical(self, "Load Error", f"An error occurred loading results:\n{e}")
            self._reset_detail_panel()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_selected_result(self):
        selected = self.results_file_list.selectedItems()
        if not selected:
            QMessageBox.warning(self, "No Selection", "Please select a result file to export.")
            return

        file_path = selected[0].data(Qt.ItemDataRole.UserRole)
        reply = QMessageBox.question(
            self, "Export to CSV",
            "Export this result to CSV format?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            csv_path = self.experiment_manager.export_results_csv(results)
            if csv_path:
                QMessageBox.information(self, "Exported", f"Results exported to:\n{csv_path}")
            else:
                QMessageBox.warning(self, "Export Failed", "Failed to export results to CSV.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error exporting results:\n{str(e)}")

    def export_blinded_results(self, results: Dict) -> str:
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
            for cond_id in results["conditions"]:
                condition_mapping[cond_id] = f"Condition_{len(condition_mapping) + 1}"

            with open(file_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Prompt_ID", "Response_ID", "Condition", "Response"])
                for prompt_result in results["prompt_results"]:
                    for response in prompt_result["responses"]:
                        writer.writerow([
                            prompt_result["prompt_id"],
                            response["response_id"],
                            condition_mapping[response["condition_id"]],
                            response["text"],
                        ])

            mapping_file = str(Path(file_path).with_suffix(".mapping.json"))
            with open(mapping_file, "w") as f:
                json.dump(condition_mapping, f, indent=2)

            QMessageBox.information(self, "Exported", f"Blinded results exported to:\n{file_path}")
            gui_logger.info(f"Blinded results exported: {file_path}")
            return file_path

        except Exception as e:
            gui_logger.error(f"Failed to export blinded results: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to export blinded results:\n{str(e)}")
            return ""
