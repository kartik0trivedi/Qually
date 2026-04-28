import json
import logging
import os

from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QPushButton, QProgressBar, QSizePolicy, QSlider, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QIcon, QDoubleValidator, QIntValidator

import qually_tool
from ui.workers import ExperimentRunner

logger = logging.getLogger("QuallyGUI")


class ExperimentsPage(QWidget):
    experiment_run_complete = pyqtSignal()
    status_message = pyqtSignal(str, int)

    def __init__(self, experiment_manager, api_key_manager, prompt_manager, data_manager, parent=None):
        super().__init__(parent)
        self.experiment_manager = experiment_manager
        self.api_key_manager = api_key_manager
        self.prompt_manager = prompt_manager
        self.data_manager = data_manager
        self._build_ui()
        self._init_data()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        main_splitter.addWidget(self._build_left_panel())
        main_splitter.addWidget(self._build_right_panel())
        main_splitter.setSizes([500, 500])
        layout.addWidget(main_splitter)

    def _build_left_panel(self):
        left_panel = QWidget()
        left_panel.setObjectName("experimentPane")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        left_layout.addWidget(self._build_experiment_details_group())
        left_layout.addWidget(self._build_conditions_group())
        return left_panel

    def _build_experiment_details_group(self):
        exp_group = QGroupBox("Create New Experiment")
        exp_layout = QFormLayout(exp_group)
        exp_layout.setContentsMargins(10, 10, 10, 10)
        exp_layout.setSpacing(5)

        name_desc_layout = QHBoxLayout()
        name_desc_layout.setSpacing(10)

        name_form = QFormLayout()
        name_form.setSpacing(5)
        self.experiment_name_edit = QLineEdit()
        self.experiment_name_edit.setPlaceholderText("Enter experiment name")
        name_form.addRow("Name:", self.experiment_name_edit)
        name_desc_layout.addLayout(name_form)

        desc_form = QFormLayout()
        desc_form.setSpacing(5)
        self.experiment_desc_edit = QLineEdit()
        self.experiment_desc_edit.setPlaceholderText("Enter a brief description (optional)")
        desc_form.addRow("Description:", self.experiment_desc_edit)
        name_desc_layout.addLayout(desc_form)

        exp_layout.addRow(name_desc_layout)
        return exp_group

    def _build_conditions_group(self):
        conditions_group = QGroupBox("Add Conditions")
        conditions_layout = QVBoxLayout(conditions_group)
        conditions_layout.setSpacing(10)

        basic_params_form = QFormLayout()
        basic_params_form.setSpacing(10)
        basic_params_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.condition_name_input = QLineEdit()
        self.condition_name_input.setPlaceholderText("Enter a descriptive name for this condition")
        basic_params_form.addRow("Condition Name:", self.condition_name_input)

        provider_model_layout = QHBoxLayout()
        self.provider_combo = QComboBox()
        provider_model_layout.addWidget(QLabel("Provider:"))
        provider_model_layout.addWidget(self.provider_combo)

        self.model_combo = QComboBox()
        provider_model_layout.addWidget(QLabel("Model:"))
        provider_model_layout.addWidget(self.model_combo)

        refresh_button = QPushButton("Refresh Models")
        refresh_button.setIcon(QIcon.fromTheme("view-refresh"))
        refresh_button.clicked.connect(self.refresh_models_list)
        provider_model_layout.addWidget(refresh_button)
        basic_params_form.addRow(provider_model_layout)

        tokens_layout = QHBoxLayout()
        self.tokens_input = QLineEdit("500")
        self.tokens_input.setValidator(QIntValidator(1, 100000))
        self.tokens_input.setMaximumWidth(100)

        tokens_info = QPushButton()
        info_icon = QIcon.fromTheme("help-about")
        if info_icon.isNull():
            tokens_info.setText("ⓘ")
            tokens_info.setStyleSheet(
                "QPushButton { color: #666; font-weight: bold; border: none; padding: 0px; background: transparent; }"
                "QPushButton:hover { color: #000; }"
            )
        else:
            tokens_info.setIcon(info_icon)
            tokens_info.setIconSize(QSize(16, 16))
            tokens_info.setStyleSheet(
                "QPushButton { border: none; padding: 0px; background: transparent; }"
            )
        tokens_info.setToolTip(
            "Tokens are pieces of text that the model processes. As a rough guide:\n\n"
            "• 1 token ≈ 4 characters or 3/4 of a word\n"
            "• A typical paragraph (100 words) ≈ 133 tokens\n"
            "• Consider your prompt length and desired response length\n"
            "• Higher values allow longer responses but may increase costs\n\n"
            "Tip: Start with 500-1000 tokens for most tasks"
        )
        tokens_layout.addWidget(self.tokens_input)
        tokens_layout.addWidget(tokens_info)
        tokens_layout.addStretch()
        basic_params_form.addRow("Max Tokens:", tokens_layout)
        conditions_layout.addLayout(basic_params_form)

        conditions_layout.addWidget(self._build_generation_params_group())

        add_condition_button = QPushButton("Add Condition")
        add_condition_button.setIcon(QIcon.fromTheme("list-add"))
        add_condition_button.clicked.connect(self.add_condition_from_form)
        conditions_layout.addWidget(add_condition_button, 0, Qt.AlignmentFlag.AlignRight)

        return conditions_group

    def _build_generation_params_group(self):
        params_group = QGroupBox("Generation Parameters")
        params_group.setProperty("advanced", True)
        params_group.setCheckable(True)
        params_group.setChecked(False)
        params_layout = QFormLayout(params_group)
        params_layout.setSpacing(10)
        params_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Temperature
        self.temp_slider, self.temp_input = self._make_slider(0, 200, 70, 0.0, 2.0, 2, "0.7", 20)
        self.temp_slider.valueChanged.connect(lambda v: self.temp_input.setText(f"{v/100:.2f}"))
        self.temp_input.textChanged.connect(self._safe_update_temp_slider)
        temp_layout = self._make_param_row(
            self.temp_slider, self.temp_input,
            "Controls randomness in the output (0.0 to 2.0):\n\n"
            "• 0.0: Most deterministic, focused output\n"
            "• 0.7: Balanced creativity and coherence\n"
            "• 1.0: More creative, varied responses\n"
            "• 2.0: Maximum creativity, less predictable",
        )
        params_layout.addRow("Temperature:", temp_layout)

        # Top P
        self.top_p_slider, self.top_p_input = self._make_slider(0, 100, 100, 0.0, 1.0, 2, "1.0", 20)
        self.top_p_slider.valueChanged.connect(lambda v: self.top_p_input.setText(f"{v/100:.2f}"))
        self.top_p_input.textChanged.connect(self._safe_update_top_p_slider)
        top_p_layout = self._make_param_row(
            self.top_p_slider, self.top_p_input,
            "Controls diversity via nucleus sampling (0.0 to 1.0):\n\n"
            "• 1.0: Consider all possible tokens\n"
            "• 0.9: Consider top 90% of tokens\n"
            "• 0.5: Consider top 50% of tokens\n"
            "• Lower values make output more focused",
        )
        params_layout.addRow("Top P:", top_p_layout)

        # Frequency Penalty
        self.freq_penalty_slider, self.freq_penalty_input = self._make_slider(-200, 200, 0, -2.0, 2.0, 2, "0.0", 100)
        self.freq_penalty_slider.valueChanged.connect(lambda v: self.freq_penalty_input.setText(f"{v/100:.2f}"))
        self.freq_penalty_input.textChanged.connect(self._safe_update_freq_penalty_slider)
        freq_layout = self._make_param_row(
            self.freq_penalty_slider, self.freq_penalty_input,
            "Controls repetition in the output (-2.0 to 2.0):\n\n"
            "• 0.0: No penalty for repetition\n"
            "• Positive: Reduces repetition of frequent tokens\n"
            "• Negative: Allows more repetition\n"
            "• Higher values encourage more diverse vocabulary",
        )
        params_layout.addRow("Frequency Penalty:", freq_layout)

        return params_group

    def _make_slider(self, min_val, max_val, default, dbl_min, dbl_max, decimals, default_text, tick_interval):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.setTickInterval(tick_interval)
        slider.setFixedHeight(30)
        slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        line_edit = QLineEdit(default_text)
        line_edit.setValidator(QDoubleValidator(dbl_min, dbl_max, decimals))
        line_edit.setFixedWidth(60)
        line_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        line_edit.setFixedHeight(30)

        return slider, line_edit

    def _make_param_row(self, slider, line_edit, tooltip):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(slider, 1)
        layout.addWidget(line_edit)
        layout.addWidget(self._make_info_button(tooltip))
        return layout

    def _make_info_button(self, tooltip_text):
        btn = QPushButton("ⓘ")
        btn.setStyleSheet(
            "QPushButton { color: #666; font-weight: bold; border: none; padding: 0px; background: transparent; }"
            "QPushButton:hover { color: #000; }"
        )
        btn.setToolTip(tooltip_text)
        return btn

    def _build_right_panel(self):
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # Current conditions
        conditions_list_group = QGroupBox("Current Conditions")
        conditions_list_layout = QVBoxLayout(conditions_list_group)

        self.conditions_table = QTableWidget()
        self.conditions_table.setAlternatingRowColors(True)
        self.conditions_table.setColumnCount(4)
        self.conditions_table.setHorizontalHeaderLabels(["Name", "Provider", "Model", "Parameters"])
        self.conditions_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.conditions_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.conditions_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.conditions_table.verticalHeader().setVisible(False)
        self.conditions_table.setMinimumHeight(100)
        conditions_list_layout.addWidget(self.conditions_table)

        condition_buttons_layout = QHBoxLayout()
        delete_cond_btn = QPushButton("Delete Selected")
        delete_cond_btn.setIcon(QIcon.fromTheme("edit-delete"))
        delete_cond_btn.clicked.connect(self.delete_selected_conditions)

        clear_cond_btn = QPushButton("Clear All")
        clear_cond_btn.setIcon(QIcon.fromTheme("edit-clear"))
        clear_cond_btn.clicked.connect(self.clear_all_conditions)

        save_exp_btn = QPushButton("Save Experiment")
        save_exp_btn.setProperty("class", "primary")
        save_exp_btn.setIcon(QIcon.fromTheme("document-save"))
        save_exp_btn.clicked.connect(self.add_experiment_from_form)

        condition_buttons_layout.addWidget(delete_cond_btn)
        condition_buttons_layout.addWidget(clear_cond_btn)
        condition_buttons_layout.addStretch()
        condition_buttons_layout.addWidget(save_exp_btn)
        conditions_list_layout.addLayout(condition_buttons_layout)
        right_layout.addWidget(conditions_list_group)

        # Existing experiments
        experiments_label = QLabel("Existing Experiments")
        experiments_label.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(experiments_label)

        self.experiments_table = QTableWidget()
        self.experiments_table.setAlternatingRowColors(True)
        self.experiments_table.setColumnCount(4)
        self.experiments_table.setHorizontalHeaderLabels(["ID", "Name", "Description", "Conditions"])
        self.experiments_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.experiments_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.experiments_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.experiments_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.experiments_table.verticalHeader().setVisible(False)
        self.experiments_table.doubleClicked.connect(self.show_experiment_details_for_selected)
        right_layout.addWidget(self.experiments_table)

        experiment_buttons_layout = QHBoxLayout()
        view_details_btn = QPushButton("View Details")
        view_details_btn.setIcon(QIcon.fromTheme("document-properties"))
        view_details_btn.clicked.connect(self.show_experiment_details_for_selected)

        delete_exp_btn = QPushButton("Delete Selected")
        delete_exp_btn.setIcon(QIcon.fromTheme("edit-delete"))
        delete_exp_btn.clicked.connect(self.delete_selected_experiments)

        run_btn = QPushButton("Run Selected")
        run_btn.setProperty("class", "primary")
        run_btn.setIcon(QIcon.fromTheme("system-run"))
        run_btn.clicked.connect(self.show_run_experiment_dialog_for_selected)

        experiment_buttons_layout.addWidget(view_details_btn)
        experiment_buttons_layout.addWidget(delete_exp_btn)
        experiment_buttons_layout.addStretch()
        experiment_buttons_layout.addWidget(run_btn)
        right_layout.addLayout(experiment_buttons_layout)

        return right_panel

    # ------------------------------------------------------------------
    # Initialization and provider/model refresh
    # ------------------------------------------------------------------

    def _init_data(self):
        providers = self.api_key_manager.list_providers()
        self.provider_combo.addItems(providers)
        self.provider_combo.currentTextChanged.connect(self.update_models_for_provider)
        if providers:
            self.update_models_for_provider(self.provider_combo.currentText())
        self.load_experiments()

    def refresh_providers(self):
        """Update provider dropdown when API keys change. Replaces tab destroy-and-recreate."""
        providers = self.api_key_manager.list_providers()
        self.provider_combo.blockSignals(True)
        self.provider_combo.clear()
        self.provider_combo.addItems(providers)
        self.provider_combo.blockSignals(False)
        if providers:
            self.update_models_for_provider(self.provider_combo.currentText())

    def update_models_for_provider(self, provider):
        if not provider:
            return
        self.model_combo.clear()
        try:
            api_key = self.api_key_manager.get_key(provider)
            if not api_key:
                logger.warning(f"No API key found for provider: {provider}")
                return
            from qually_tool import ProviderFactory
            provider_instance = ProviderFactory.create_provider(provider, api_key)
            if not provider_instance:
                logger.error(f"Failed to create provider instance for {provider}")
                return
            self.status_message.emit(f"Loading models for {provider}...", 0)
            QApplication.processEvents()
            models = provider_instance.get_available_models()
            self.model_combo.addItems(models)
            self.status_message.emit(f"Loaded {len(models)} models for {provider}", 3000)
        except Exception as e:
            logger.error(f"Error fetching models for {provider}: {e}", exc_info=True)
            self.status_message.emit(f"Error loading models for {provider}", 3000)

    def refresh_models_list(self):
        provider = self.provider_combo.currentText()
        if provider:
            self.update_models_for_provider(provider)

    # ------------------------------------------------------------------
    # Slider helpers
    # ------------------------------------------------------------------

    def _safe_update_temp_slider(self, text):
        try:
            if text and text.strip():
                self.temp_slider.setValue(int(max(0.0, min(2.0, float(text))) * 100))
        except (ValueError, AttributeError):
            pass

    def _safe_update_top_p_slider(self, text):
        try:
            if text and text.strip():
                self.top_p_slider.setValue(int(max(0.0, min(1.0, float(text))) * 100))
        except (ValueError, AttributeError):
            pass

    def _safe_update_freq_penalty_slider(self, text):
        try:
            if text and text.strip():
                self.freq_penalty_slider.setValue(int(max(-2.0, min(2.0, float(text))) * 100))
        except (ValueError, AttributeError):
            pass

    # ------------------------------------------------------------------
    # Condition management
    # ------------------------------------------------------------------

    def add_condition_from_form(self):
        condition_name = self.condition_name_input.text().strip()
        provider = self.provider_combo.currentText()
        model = self.model_combo.currentText()

        if not condition_name:
            QMessageBox.warning(self, "Validation Error", "Please enter a condition name.")
            return
        if not provider:
            QMessageBox.warning(self, "Validation Error", "Please select a provider.")
            return
        if not model:
            QMessageBox.warning(self, "Validation Error", "Please select a model.")
            return

        try:
            parameters = {
                "temperature": float(self.temp_input.text()),
                "max_tokens": int(self.tokens_input.text()),
                "top_p": float(self.top_p_input.text()),
                "frequency_penalty": float(self.freq_penalty_input.text()),
                # The current form has one penalty field. Keep presence aligned until
                # a dedicated presence-penalty control is added.
                "presence_penalty": float(self.freq_penalty_input.text()),
            }
            row = self.conditions_table.rowCount()
            self.conditions_table.insertRow(row)
            self.conditions_table.setItem(row, 0, QTableWidgetItem(condition_name))
            self.conditions_table.setItem(row, 1, QTableWidgetItem(provider))
            self.conditions_table.setItem(row, 2, QTableWidgetItem(model))
            self.conditions_table.setItem(row, 3, QTableWidgetItem(json.dumps(parameters)))
            self.condition_name_input.clear()
            logger.info(f"Added condition: {condition_name} ({provider}/{model})")
            self.status_message.emit(f"Added condition: {condition_name}", 3000)
        except Exception as e:
            logger.error(f"Error adding condition: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to add condition: {str(e)}")

    def delete_selected_conditions(self):
        selected_rows = self.conditions_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select conditions to delete.")
            return
        for row in sorted([i.row() for i in selected_rows], reverse=True):
            self.conditions_table.removeRow(row)

    def clear_all_conditions(self):
        reply = QMessageBox.question(
            self, "Confirm Clear All",
            "Are you sure you want to delete ALL conditions? This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.conditions_table.setRowCount(0)

    # ------------------------------------------------------------------
    # Experiment management
    # ------------------------------------------------------------------

    def add_experiment_from_form(self):
        try:
            name = self.experiment_name_edit.text().strip()
            description = self.experiment_desc_edit.text().strip()
            if not name:
                QMessageBox.warning(self, "Error", "Experiment name cannot be empty.")
                return
            conditions = []
            for row in range(self.conditions_table.rowCount()):
                conditions.append({
                    "name": self.conditions_table.item(row, 0).text(),
                    "provider": self.conditions_table.item(row, 1).text(),
                    "model": self.conditions_table.item(row, 2).text(),
                    "parameters": json.loads(self.conditions_table.item(row, 3).text()),
                })
            experiment_id = self.experiment_manager.create_experiment_with_conditions(
                name=name, description=description, conditions=conditions
            )
            if experiment_id:
                self.experiment_name_edit.clear()
                self.experiment_desc_edit.clear()
                self.conditions_table.setRowCount(0)
                self.load_experiments()
                QMessageBox.information(self, "Success", f"Experiment created successfully with ID: {experiment_id}")
                logger.info(f"Created new experiment with ID: {experiment_id}")
            else:
                QMessageBox.critical(self, "Error", "Failed to create experiment.")
                logger.error("Failed to create experiment")
        except Exception as e:
            logger.error(f"Error creating experiment: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to create experiment: {str(e)}")

    def load_experiments(self):
        self.experiments_table.setRowCount(0)
        for experiment in self.experiment_manager.list_experiments():
            row = self.experiments_table.rowCount()
            self.experiments_table.insertRow(row)
            for col, value in enumerate([
                experiment["id"],
                experiment["name"],
                experiment.get("description", ""),
                str(len(experiment.get("conditions", []))),
            ]):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.experiments_table.setItem(row, col, item)

    def delete_selected_experiments(self):
        selected_rows = self.experiments_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select experiments to delete.")
            return

        experiment_ids = [self.experiments_table.item(r.row(), 0).text() for r in selected_rows]
        if len(experiment_ids) == 1:
            message = f"Are you sure you want to delete experiment '{experiment_ids[0]}'?"
        else:
            message = f"Are you sure you want to delete {len(experiment_ids)} selected experiments?"

        reply = QMessageBox.question(
            self, "Confirm Deletion", message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                for experiment_id in experiment_ids:
                    self.experiment_manager.delete_experiment(experiment_id)
                self.load_experiments()
                QMessageBox.information(self, "Success", "Selected experiments deleted successfully.")
                logger.info(f"Deleted experiments: {experiment_ids}")
            except Exception as e:
                logger.error(f"Failed to delete experiments: {e}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to delete experiments: {str(e)}")

    # ------------------------------------------------------------------
    # Experiment details dialog
    # ------------------------------------------------------------------

    def show_experiment_details_for_selected(self):
        selected_rows = self.experiments_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "Selection Required", "Please select an experiment from the table first.")
            return
        if len(selected_rows) > 1:
            QMessageBox.warning(self, "Selection Error", "Please select only one experiment.")
            return
        experiment_id = self.experiments_table.item(selected_rows[0].row(), 0).text()
        self.show_experiment_details(experiment_id)

    def show_experiment_details(self, experiment_id: str):
        experiment = self.experiment_manager.get_experiment(experiment_id)
        if not experiment:
            QMessageBox.warning(self, "Error", f"Experiment with ID {experiment_id} not found.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Experiment Details: {experiment['name']}")
        dialog.setMinimumWidth(600)
        layout = QVBoxLayout(dialog)

        info_group = QGroupBox("Basic Information")
        info_layout = QFormLayout(info_group)
        info_layout.addRow("ID:", QLabel(experiment["id"]))
        info_layout.addRow("Name:", QLabel(experiment["name"]))
        info_layout.addRow("Description:", QLabel(experiment.get("description", "N/A")))
        info_layout.addRow("Created At:", QLabel(experiment.get("created_at", "N/A")))
        layout.addWidget(info_group)

        conditions_group = QGroupBox("Conditions")
        conditions_layout = QVBoxLayout(conditions_group)
        conditions_table = QTableWidget()
        conditions_table.setColumnCount(4)
        conditions_table.setHorizontalHeaderLabels(["Name", "Provider", "Model", "Parameters"])
        conditions_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        conditions_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        conditions_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        conditions_table.verticalHeader().setVisible(False)
        conditions = experiment.get("conditions", [])
        conditions_table.setRowCount(len(conditions))
        for row, cond in enumerate(conditions):
            conditions_table.setItem(row, 0, QTableWidgetItem(cond["name"]))
            conditions_table.setItem(row, 1, QTableWidgetItem(cond["provider"]))
            conditions_table.setItem(row, 2, QTableWidgetItem(cond["model"]))
            conditions_table.setItem(row, 3, QTableWidgetItem(json.dumps(cond.get("parameters", {}), indent=2)))
        conditions_layout.addWidget(conditions_table)
        layout.addWidget(conditions_group)

        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        results_table = QTableWidget()
        results_table.setColumnCount(3)
        results_table.setHorizontalHeaderLabels(["Timestamp", "Status", "File"])
        results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        results_table.verticalHeader().setVisible(False)

        results_files = [
            f for f in os.listdir(self.experiment_manager.results_folder)
            if f.startswith(f"results_{experiment_id}_") and f.endswith(".json")
        ]
        results_table.setRowCount(len(results_files))
        for row, file in enumerate(results_files):
            try:
                with open(os.path.join(self.experiment_manager.results_folder, file)) as f:
                    result = json.load(f)
                results_table.setItem(row, 0, QTableWidgetItem(result.get("timestamp", "N/A")))
                results_table.setItem(row, 1, QTableWidgetItem("Completed" if not result.get("error") else "Failed"))
                results_table.setItem(row, 2, QTableWidgetItem(file))
            except Exception as e:
                logger.error(f"Error loading result file {file}: {e}")
                results_table.setItem(row, 0, QTableWidgetItem("N/A"))
                results_table.setItem(row, 1, QTableWidgetItem("Error"))
                results_table.setItem(row, 2, QTableWidgetItem(file))
        results_layout.addWidget(results_table)
        layout.addWidget(results_group)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        dialog.exec()

    # ------------------------------------------------------------------
    # Run experiment
    # ------------------------------------------------------------------

    def show_run_experiment_dialog_for_selected(self):
        selected_items = self.experiments_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select an experiment to run.")
            return

        row = selected_items[0].row()
        experiment_id = self.experiments_table.item(row, 0).text()
        experiment = self.experiment_manager.get_experiment(experiment_id)
        if not experiment:
            QMessageBox.critical(self, "Error", "Selected experiment not found.")
            return

        prompts = self.prompt_manager.list_prompts()
        if not prompts:
            QMessageBox.warning(self, "No Prompts", "No prompts available to run.")
            return

        data_prompts = []
        data_ids = set()
        if self.data_manager and self.data_manager.has_data() and self.data_manager.file_info.get("id_column"):
            data_ids = set(self.data_manager.data[self.data_manager.file_info["id_column"]].astype(str))
            for prompt in prompts:
                data_id = None
                for tag in prompt.get("tags", []):
                    if tag.startswith("data_id_"):
                        data_id = tag.replace("data_id_", "")
                        break
                if data_id in data_ids:
                    data_prompts.append(prompt)
                    logger.info(f"Found matching prompt: ID={prompt['id']}, data_id={data_id}")
                else:
                    logger.debug(f"No match for prompt: ID={prompt['id']}, data_id={data_id}")

        if not data_prompts:
            logger.warning("No matching data prompts found. Available IDs in data: " +
                           ", ".join(str(i) for i in data_ids))
            reply = QMessageBox.warning(
                self, "Warning",
                "No prompts matching your data file IDs were found. This means the experiment will only run "
                "with a single prompt instead of processing all rows in your data file.\n\n"
                "Would you like to continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        dialog = QDialog(self)
        dialog.setWindowTitle("Running Experiment")
        dlg_layout = QVBoxLayout(dialog)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, 0)
        progress_bar.setTextVisible(False)
        dlg_layout.addWidget(progress_bar)

        status_label = QLabel("Preparing to run experiment...")
        dlg_layout.addWidget(status_label)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        dlg_layout.addWidget(cancel_button)

        dialog.show()
        QApplication.processEvents()

        prompt_ids = [p["id"] for p in (data_prompts if data_prompts else prompts)]
        self._experiment_runner = ExperimentRunner(self.experiment_manager, experiment_id, prompt_ids)

        def update_progress(current, total):
            status_label.setText(f"Processing prompt {current}/{total}")
            QApplication.processEvents()

        self._experiment_runner.progress.connect(update_progress)
        self._experiment_runner.finished.connect(lambda results: self._on_experiment_finished(results, dialog))
        self._experiment_runner.error.connect(lambda error: self._on_experiment_error(error, dialog))
        self._experiment_runner.start()

    def _on_experiment_finished(self, results, dialog):
        dialog.accept()
        self.experiment_run_complete.emit()
        if results:
            QMessageBox.information(
                self, "Experiment Complete",
                f"Experiment completed successfully.\nResults saved to: {results.get('results_file', 'results folder')}",
            )
        else:
            QMessageBox.warning(self, "Experiment Complete", "Experiment completed but no results were generated.")

    def _on_experiment_error(self, error, dialog):
        dialog.reject()
        QMessageBox.critical(self, "Error", f"Experiment failed: {error}")
