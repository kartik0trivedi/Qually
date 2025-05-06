import sys
import logging
import time
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox, QTableWidget,
    QTableWidgetItem, QMessageBox, QFileDialog, QListWidget, QProgressBar,
    QFrame, QHeaderView, QScrollArea, QGroupBox, QListWidgetItem, QDialog, QDialogButtonBox,
    QProgressDialog, QFormLayout, QSplitter, QGridLayout, QSlider, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QPixmap, QDoubleValidator, QIntValidator
import pandas as pd
import json
import os
from pathlib import Path
from qually_tool import APIKeyManager, PromptManager, ExperimentManager, ProviderFactory
from typing import Dict
import csv
import datetime
import tempfile

# Configure logging for the GUI (optional, but helpful)
log_folder = Path("logs")
log_folder.mkdir(exist_ok=True)
gui_log_file = log_folder / "qually_gui.log"

logging.basicConfig(
    level=logging.INFO, # Adjust level as needed
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(gui_log_file),
        logging.StreamHandler() # Also show logs in console
    ]
)
gui_logger = logging.getLogger("QuallyGUI")

# Determine if running as a bundled app
if getattr(sys, 'frozen', False):
    # If running as a bundled app
    application_path = Path(sys.executable).parent.parent
    resources_path = application_path / 'Resources'
    gui_logger.info(f"Running as bundled app from: {application_path}")
    gui_logger.info(f"Resources path: {resources_path}")
    
    # If resources not found in expected location, try alternate paths
    if not resources_path.exists():
        alternate_paths = [
            application_path.parent / 'Resources',
            application_path / 'Contents' / 'Resources',
            Path(sys.executable).parent / 'resources'
        ]
        for alt_path in alternate_paths:
            if alt_path.exists():
                resources_path = alt_path
                gui_logger.info(f"Found resources in alternate path: {resources_path}")
                break
        else:
            gui_logger.warning("Could not find resources directory in any expected location")
else:
    # Running in development mode
    application_path = Path(__file__).parent
    resources_path = application_path / 'resources'
    # Change working directory to the script's directory for relative paths
    os.chdir(application_path)
    gui_logger.info(f"Running as script from: {application_path}")
    gui_logger.info(f"Resources path: {resources_path}")

# Log the actual icon path that will be used
icon_path = resources_path / 'icon.png'
gui_logger.info(f"Icon path: {icon_path} (exists: {icon_path.exists()})")


class ExperimentRunner(QThread):
    """Runs an experiment in a separate thread to avoid freezing the GUI."""
    progress = pyqtSignal(int, int) # current_step, total_steps
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    log_message = pyqtSignal(str) # Signal to emit log messages

    def __init__(self, experiment_manager, experiment_id, prompt_ids):
        super().__init__()
        self.experiment_manager = experiment_manager
        self.experiment_id = experiment_id
        self.prompt_ids = prompt_ids
        self._is_running = True

    def run(self):
        """Executes the experiment run."""
        self.log_message.emit(f"Starting experiment run for ID: {self.experiment_id}")
        try:
            # Validate experiment and prompts before running
            if not self.experiment_manager:
                raise ValueError("Experiment manager not initialized")
            
            experiment = self.experiment_manager.get_experiment(self.experiment_id)
            if not experiment:
                raise ValueError(f"Experiment with ID {self.experiment_id} not found")
            
            if not experiment.get('conditions'):
                raise ValueError("Experiment has no conditions defined")
            
            # --- Direct call to backend run_experiment ---
            results = self.experiment_manager.run_experiment(self.experiment_id, self.prompt_ids)

            if results:
                self.log_message.emit(f"Experiment run completed successfully for ID: {self.experiment_id}")
                self.finished.emit(results)
            else:
                self.log_message.emit(f"Experiment run for ID {self.experiment_id} returned no results or failed.")
                self.error.emit("Experiment run finished but returned no results. Check backend logs.")

        except Exception as e:
            gui_logger.error(f"Error during experiment run thread: {e}", exc_info=True)
            self.log_message.emit(f"Error during experiment run: {e}")
            self.error.emit(f"An error occurred during the experiment: {str(e)}")

    def stop(self):
        """Stop the experiment run."""
        self._is_running = False
        self.log_message.emit("Experiment run cancellation requested.")
        # Note: The backend doesn't currently support cancellation mid-run.
        # This only prevents the thread from emitting signals after stopping.

class DataFileManager:
    def __init__(self):
        self.data = None
        self.file_info = None

    def import_csv(self, file_path):
        """Import a CSV file and store both the data and file info."""
        try:
            self.data = pd.read_csv(file_path, dtype=str)
            self.data.fillna("", inplace=True)
            
            # Auto-detect text column
            possible_text_cols = ["text", "content", "body", "main_text"]
            text_column = None
            for col in possible_text_cols:
                if col in self.data.columns:
                    text_column = col
                    break
            
            self.file_info = {
                "file_path": file_path,
                "text_column": text_column,
                "total_rows": len(self.data),
                "file_type": "csv"
            }
            
            return self.data.head(5), list(self.data.columns)
        except Exception as e:
            gui_logger.error(f"Failed to import CSV: {e}", exc_info=True)
            raise

    def import_json(self, file_path):
        """Import a JSON file and store both the data and file info."""
        try:
            # Read JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Convert JSON to DataFrame
            # Handle both array of objects and single object cases
            if isinstance(json_data, list):
                self.data = pd.DataFrame(json_data)
            elif isinstance(json_data, dict):
                # If it's a single object, convert to DataFrame with one row
                self.data = pd.DataFrame([json_data])
            else:
                raise ValueError("JSON file must contain either an array of objects or a single object")
            
            self.data.fillna("", inplace=True)
            
            # Auto-detect text column
            possible_text_cols = ["text", "content", "body", "main_text"]
            text_column = None
            for col in possible_text_cols:
                if col in self.data.columns:
                    text_column = col
                    break
            
            self.file_info = {
                "file_path": file_path,
                "text_column": text_column,
                "total_rows": len(self.data),
                "file_type": "json"
            }
            
            return self.data.head(5), list(self.data.columns)
        except Exception as e:
            gui_logger.error(f"Failed to import JSON: {e}", exc_info=True)
            raise

    def has_data(self):
        """Check if data is loaded."""
        return self.data is not None and self.file_info is not None

    def import_excel(self, file_path):
        """Import an Excel file and store both the data and file info."""
        try:
            self.data = pd.read_excel(file_path, dtype=str)
            self.data.fillna("", inplace=True)

            # Auto-detect text column
            possible_text_cols = ["text", "content", "body", "main_text"]
            text_column = None
            for col in possible_text_cols:
                if col in self.data.columns:
                    text_column = col
                    break

            self.file_info = {
                "file_path": file_path,
                "text_column": text_column,
                "total_rows": len(self.data),
                "file_type": "excel"
            }

            return self.data.head(5), list(self.data.columns)
        except Exception as e:
            gui_logger.error(f"Failed to import Excel file: {e}", exc_info=True)
            raise

class SettingsManager:
    SETTINGS_PATH = "settings.json"

    @staticmethod
    def load_settings():
        try:
            with open(SettingsManager.SETTINGS_PATH, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    @staticmethod
    def save_settings(settings):
        with open(SettingsManager.SETTINGS_PATH, 'w') as f:
            json.dump(settings, f, indent=4)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM Audit Tool")
        self.setMinimumSize(1200, 800)
        
        # Initialize log panel
        self.log_panel = QTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setMaximumHeight(150)
        
        # Initialize other attributes
        self.project_folder = None
        self.experiment_manager = None
        self.prompt_manager = None
        self.data_manager = DataFileManager()

        self.settings = SettingsManager.load_settings()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Initialize project folder and managers to None initially
        self.project_folder = None
        self.api_key_manager = None
        self.prompt_manager = None
        self.experiment_manager = None

        # Show welcome screen
        self.show_welcome_screen()

        self.last_temp_csv_path = None  # Track last temp CSV for cleanup

    def show_welcome_screen(self):
        """Show the welcome screen with folder selection."""
        gui_logger.info("Displaying welcome screen.")

        # Clear existing widgets
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Create welcome widget and layout
        welcome_widget = QWidget()
        welcome_layout = QVBoxLayout(welcome_widget)
        welcome_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_layout.setSpacing(20)

        # Logo section
        img_path = resources_path / 'icon.png'
        logo_label = QLabel()
        logo_pixmap = QPixmap(str(img_path))
        if not logo_pixmap.isNull():
            scaled_pixmap = logo_pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
        else:
            gui_logger.warning(f"Logo file not found or invalid: {img_path}")
            logo_label.setText("[Logo Not Found]")

        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_layout.addWidget(logo_label)

        # Welcome message
        welcome_label = QLabel("Welcome to Qually")
        welcome_label.setStyleSheet("font-size: 28px; font-weight: bold;")
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_layout.addWidget(welcome_label)

        # Description
        description = QLabel(
            "Qually helps you audit and analyze LLM outputs for social science research.\n"
            "Please select a folder where your project files (prompts, experiments, results, keys) will be stored."
        )
        description.setStyleSheet("font-size: 14px; color: #555;")
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setWordWrap(True)
        welcome_layout.addWidget(description)

        # Folder selection group
        folder_group = QFrame()
        folder_group.setFrameShape(QFrame.Shape.StyledPanel)
        folder_layout = QHBoxLayout(folder_group)

        self.folder_path_label = QLineEdit("No folder selected")
        self.folder_path_label.setReadOnly(True)
        self.folder_path_label.setStyleSheet("padding: 5px; border: 1px solid #ccc; border-radius: 3px; background-color: #eee;")
        folder_layout.addWidget(QLabel("Project Folder:"))
        folder_layout.addWidget(self.folder_path_label, 1)

        select_button = QPushButton("...")
        select_button.setToolTip("Select Project Folder")
        select_button.setFixedWidth(40)
        select_button.clicked.connect(self.select_project_folder)
        folder_layout.addWidget(select_button)

        welcome_layout.addWidget(folder_group)

        # Continue button (initially disabled)
        self.continue_button = QPushButton("Start Qually")
        self.continue_button.setStyleSheet("font-size: 16px; padding: 10px 20px;")
        self.continue_button.clicked.connect(self.setup_main_interface)
        self.continue_button.setEnabled(False)
        welcome_layout.addWidget(self.continue_button, 0, Qt.AlignmentFlag.AlignCenter)

        self.layout.addWidget(welcome_widget)

    def select_project_folder(self):
        """Show folder selection dialog and validate writability."""
        gui_logger.info("Opening project folder selection dialog.")
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Project Folder",
            str(Path.home()),
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks
        )

        if folder:
            folder_path = Path(folder)
            gui_logger.info(f"Folder selected: {folder_path}")
            try:
                # Test folder writability
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
                    f"The selected folder is not writable or accessible:\n\n{folder}\n\nError: {str(e)}\n\nPlease select a different folder."
                )
                self.project_folder = None
                self.folder_path_label.setText("No folder selected")
                self.continue_button.setEnabled(False)

    def setup_main_interface(self):
        """Set up the main tabbed interface after folder selection."""
        if not self.project_folder:
            QMessageBox.warning(
                self,
                "Warning",
                "Please select a valid project folder first."
            )
            return

        gui_logger.info(f"Setting up main interface for project: {self.project_folder}")
        
        # Clear existing widgets from the welcome screen
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Create necessary subdirectories
        try:
            folder_path = Path(self.project_folder)
            (folder_path / "results").mkdir(parents=True, exist_ok=True)
            (folder_path / "logs").mkdir(parents=True, exist_ok=True)
        except Exception as mkdir_e:
            gui_logger.warning(f"Could not create subdirectories in {folder_path}: {mkdir_e}")
            QMessageBox.warning(
                self,
                "Warning",
                f"Could not create required subdirectories in project folder.\nError: {str(mkdir_e)}"
            )

        # Initialize managers
        try:
            self.experiment_manager = ExperimentManager(self.project_folder)
            self.api_key_manager = self.experiment_manager.api_key_manager
            self.prompt_manager = self.experiment_manager.prompt_manager

            # Create main tab widget
            self.tabs = QTabWidget()
            self.layout.addWidget(self.tabs)

            # Create tabs
            self.tabs.addTab(self.create_api_keys_tab(), "API Keys")
            self.tabs.addTab(self.create_data_files_tab(), "Data Files")
            self.tabs.addTab(self.create_prompts_tab(), "Prompts")
            self.tabs.addTab(self.create_experiments_tab(), "Experiments")
            self.tabs.addTab(self.create_results_tab(), "Results")

            # Status Bar Setup
            self.statusBar().showMessage("Ready")

            # Progress bar in status bar (initially hidden)
            self.progress_bar = QProgressBar()
            self.progress_bar.setMaximum(100)
            self.progress_bar.setValue(0)
            self.progress_bar.setTextVisible(False)
            self.progress_bar.setFixedWidth(200)
            self.progress_bar.hide()
            self.statusBar().addPermanentWidget(self.progress_bar)

            # Add project folder label to status bar
            self.folder_label = QLabel(f"Project: {os.path.basename(self.project_folder)}")
            self.statusBar().addPermanentWidget(self.folder_label)

            # Check if any API keys are missing and prompt user if needed
            QTimer.singleShot(100, self.check_api_keys_on_startup)

        except Exception as e:
            gui_logger.error(f"Failed to initialize managers: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to initialize managers:\n{str(e)}"
            )
            return

    def check_api_keys_on_startup(self):
        """Checks if API keys exist and prompts the user if none are set."""
        if self.api_key_manager and not self.api_key_manager.list_providers():
            gui_logger.warning("No API keys found on startup.")
            reply = QMessageBox.information(
                self,
                "API Keys Needed",
                "No API keys found in your project.\n\n"
                "You need to add API keys to run experiments.\n"
                "Would you like to go to the API Keys tab now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.tabs.setCurrentIndex(0) # Switch to API Keys tab

    def create_api_keys_tab(self):
        """Creates the API Keys configuration tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)

        # Add help text
        help_text = QLabel(
            "Add your API keys for different LLM services below. These keys are required to run experiments.\n\n"
            "<b>Supported services:</b>\n"
            " • <b>OpenAI:</b> For GPT models (e.g., gpt-4, gpt-3.5-turbo)\n"
            " • <b>Anthropic:</b> For Claude models (e.g., claude-3-opus, claude-3.5-sonnet)\n"
            " • <b>Google:</b> For Gemini models (e.g., gemini-1.5-pro, gemini-1.5-flash)\n"
            " • <b>Mistral:</b> For Mistral models (e.g., mistral-large-latest, open-mixtral-8x7b)\n"
            " • <b>Grok (xAI):</b> For Grok models (e.g., grok-1) <i>(Requires API access from xAI)</i>\n"
            " • <b>DeepSeek:</b> For DeepSeek models (e.g., deepseek-chat, deepseek-coder)\n\n"
            "<i>Your API keys are stored encrypted in the 'api_keys.json' file within your project folder.</i>"
        )
        help_text.setWordWrap(True)
        help_text.setTextFormat(Qt.TextFormat.RichText)
        help_text.setStyleSheet("""
            QLabel {
                padding: 15px;
                background-color: #e9ecef;
                border: 1px solid #ced4da;
                border-radius: 5px;
                margin-bottom: 20px;
                font-size: 13px;
            }
        """)
        layout.addWidget(help_text)

        # API Keys input fields in a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        api_keys_group = QWidget()
        api_keys_layout = QVBoxLayout(api_keys_group)
        api_keys_layout.setSpacing(10)

        # Input Fields
        self.api_key_inputs = {}
        self.api_key_status_icons = {}
        self.api_key_test_buttons = {}
        providers = ["OpenAI", "Anthropic", "Google", "Mistral", "Grok", "DeepSeek"]
        for provider in providers:
            key_layout = QHBoxLayout()
            key_label = QLabel(f"{provider} API Key:")
            key_label.setFixedWidth(120)
            key_input = QLineEdit()
            key_input.setEchoMode(QLineEdit.EchoMode.Password)
            key_input.setPlaceholderText(f"Enter {provider} key here")
            key_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            # Status icon
            status_icon = QLabel()
            status_icon.setFixedWidth(24)
            status_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
            status_icon.setText("⏺️")  # Not tested (gray dot)
            status_icon.setToolTip("Not tested")
            # Test button
            test_button = QPushButton("Test")
            test_button.setFixedWidth(50)
            test_button.setToolTip(f"Test {provider} API key")
            # Add widgets to layout
            key_layout.addWidget(key_label)
            key_layout.addWidget(key_input, 1)  # Expanding
            key_layout.addWidget(status_icon, 0)
            key_layout.addWidget(test_button, 0)
            api_keys_layout.addLayout(key_layout)
            # Store references
            self.api_key_inputs[provider.lower()] = key_input
            self.api_key_status_icons[provider.lower()] = status_icon
            self.api_key_test_buttons[provider.lower()] = test_button

        scroll_area.setWidget(api_keys_group)
        layout.addWidget(scroll_area)

        # Save button
        save_button = QPushButton("Save All API Keys")
        save_button.setIcon(QIcon.fromTheme("document-save"))
        save_button.clicked.connect(self.save_api_keys)
        layout.addWidget(save_button, 0, Qt.AlignmentFlag.AlignRight)

        layout.addStretch()

        # Load existing keys after creating the inputs
        self.load_api_keys_to_ui()

        # Connect test buttons to test logic
        for provider_lower, test_button in self.api_key_test_buttons.items():
            test_button.clicked.connect(lambda checked, p=provider_lower: self.test_api_key(p))

        return tab

    def load_api_keys_to_ui(self):
        """Loads existing keys (just indicators) into the UI fields."""
        if not self.api_key_manager:
            gui_logger.warning("API Key Manager not initialized, cannot load keys to UI.")
            return

        gui_logger.info("Loading API key presence indicators into UI.")
        for provider_lower, key_input in self.api_key_inputs.items():
            if self.api_key_manager.get_key(provider_lower):
                key_input.setPlaceholderText("Key is set (enter new key to replace)")
            else:
                key_input.setPlaceholderText(f"Enter {provider_lower.capitalize()} key here")
            key_input.clear()

    def save_api_keys(self):
        """Saves the API keys entered in the UI."""
        if not self.api_key_manager:
            QMessageBox.critical(self, "Error", "API Key Manager not initialized.")
            return

        gui_logger.info("Attempting to save API keys.")
        keys_saved_count = 0
        try:
            for provider_lower, key_input in self.api_key_inputs.items():
                key_text = key_input.text().strip()
                if key_text:
                    self.api_key_manager.save_key(provider_lower, key_text)
                    keys_saved_count += 1
                    key_input.clear()
                    key_input.setPlaceholderText("Key is set (enter new key to replace)")
                    gui_logger.info(f"Saved API key for provider: {provider_lower}")

            if keys_saved_count > 0:
                QMessageBox.information(self, "Success", f"{keys_saved_count} API key(s) saved successfully!")
                # Refresh the experiments tab after saving new API keys
                self.refresh_experiments_tab()
            else:
                QMessageBox.information(self, "No Changes", "No new API keys were entered to save.")

            self.load_api_keys_to_ui()

        except Exception as e:
            gui_logger.error(f"Failed to save API keys: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to save API keys: {str(e)}")

    def refresh_experiments_tab(self):
        """Refreshes the experiments tab to update provider and model lists."""
        # Get the current tab index
        current_index = self.tabs.currentIndex()
        
        # Remove the old experiments tab
        self.tabs.removeTab(self.tabs.indexOf(self.tabs.widget(3)))
        
        # Add the new experiments tab
        self.tabs.insertTab(3, self.create_experiments_tab(), "Experiments")
        
        # Restore the current tab
        self.tabs.setCurrentIndex(current_index)

    def create_data_files_tab(self):
        """Creates the Data Files management tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # File selection group
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
        info_icon.setToolTip(
            "Supported formats: CSV (.csv), JSON (.json), Excel (.xls, .xlsx).\n"
            "For JSON, use an array of objects or a single object.\n"
            "For Excel, the first sheet will be used."
        )

        # New: Import multiple text files button
        import_txt_button = QPushButton("Import Multiple Text Files")
        import_txt_button.setToolTip("Import multiple .txt files and convert to a table with ID, file name, and content.")
        import_txt_button.clicked.connect(self.import_multiple_text_files)

        file_layout.addWidget(data_label)
        file_layout.addWidget(self.file_path_input)
        file_layout.addWidget(browse_button)
        file_layout.addWidget(info_icon)
        file_layout.addWidget(import_txt_button)

        layout.addWidget(file_group)

        # Preview group
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_group)

        self.preview_table = QTableWidget()
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.preview_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.preview_table.verticalHeader().setVisible(False)
        preview_layout.addWidget(self.preview_table)

        layout.addWidget(preview_group)

        # Column selection group
        column_group = QGroupBox("Column Selection")
        column_layout = QVBoxLayout(column_group)

        # ID column selection
        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("ID Column:"))
        self.id_column_combo = QComboBox()
        id_layout.addWidget(self.id_column_combo)
        column_layout.addLayout(id_layout)

        # Text column selection
        text_layout = QHBoxLayout()
        text_layout.addWidget(QLabel("Text Column:"))
        self.text_column_combo = QComboBox()
        text_layout.addWidget(self.text_column_combo)
        column_layout.addLayout(text_layout)

        layout.addWidget(column_group)

        return tab

    def import_multiple_text_files(self):
        """Import multiple .txt files, convert to a table, and load as data."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Text Files",
            str(self.project_folder) if self.project_folder else str(Path.home()),
            "Text Files (*.txt)"
        )
        if not file_paths:
            return
        try:
            rows = []
            for idx, file_path in enumerate(file_paths):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                rows.append({
                    'id': f'doc_{idx+1:03d}',
                    'file_name': Path(file_path).name,
                    'content': content
                })
            # Save to a temporary CSV file
            df = pd.DataFrame(rows)
            temp_dir = tempfile.gettempdir()
            temp_csv_path = os.path.join(temp_dir, f"qually_imported_texts_{int(time.time())}.csv")
            df.to_csv(temp_csv_path, index=False)
            self.last_temp_csv_path = temp_csv_path  # Track for cleanup
            # Offer to save in project folder
            reply = QMessageBox.question(
                self,
                "Save as CSV?",
                "Would you like to save this table as a CSV in your project folder?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                save_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save CSV File",
                    str(Path(self.project_folder) / "imported_texts.csv"),
                    "CSV Files (*.csv)"
                )
                if save_path:
                    df.to_csv(save_path, index=False)
                    QMessageBox.information(self, "Saved", f"CSV saved to: {save_path}")
            # Load as data file
            preview_data, columns = self.data_manager.import_csv(temp_csv_path)
            self.file_path_input.setText(temp_csv_path)
            # Update preview table
            self.preview_table.setRowCount(len(preview_data))
            self.preview_table.setColumnCount(len(columns))
            self.preview_table.setHorizontalHeaderLabels(columns)
            for row in range(len(preview_data)):
                for col in range(len(columns)):
                    item = QTableWidgetItem(str(preview_data.iloc[row, col]))
                    self.preview_table.setItem(row, col, item)
            self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            # Update column selection combos
            self.id_column_combo.clear()
            self.text_column_combo.clear()
            self.id_column_combo.addItems(columns)
            self.text_column_combo.addItems(columns)
            # Auto-select columns
            self.id_column_combo.setCurrentText('id')
            self.text_column_combo.setCurrentText('content')
            # Update file_info
            self.data_manager.file_info['id_column'] = 'id'
            self.data_manager.file_info['text_column'] = 'content'
            # Connect combos
            self.text_column_combo.currentTextChanged.connect(self.update_text_column)
            self.id_column_combo.currentTextChanged.connect(self.update_id_column)
            gui_logger.info(f"Imported {len(file_paths)} text files as data table.")
            self.refresh_prompts_tab()
        except Exception as e:
            gui_logger.error(f"Failed to import text files: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to import text files: {str(e)}")

    def browse_data_file(self):
        """Open file dialog to select a data file and load its preview."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            str(self.project_folder) if self.project_folder else str(Path.home()),
            "Data Files (*.csv *.json *.xls *.xlsx)"
        )

        if file_path:
            try:
                # Load and preview data based on file extension
                file_extension = Path(file_path).suffix.lower()
                if file_extension == '.csv':
                    preview_data, columns = self.data_manager.import_csv(file_path)
                elif file_extension == '.json':
                    preview_data, columns = self.data_manager.import_json(file_path)
                elif file_extension in ['.xls', '.xlsx']:
                    preview_data, columns = self.data_manager.import_excel(file_path)
                else:
                    raise ValueError(f"Unsupported file type: {file_extension}")

                # Update file path
                self.file_path_input.setText(file_path)

                # Update preview table
                self.preview_table.setRowCount(len(preview_data))
                self.preview_table.setColumnCount(len(columns))
                self.preview_table.setHorizontalHeaderLabels(columns)

                for row in range(len(preview_data)):
                    for col in range(len(columns)):
                        item = QTableWidgetItem(str(preview_data.iloc[row, col]))
                        self.preview_table.setItem(row, col, item)

                self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

                # Update column selection combos
                self.id_column_combo.clear()
                self.text_column_combo.clear()
                self.id_column_combo.addItems(columns)
                self.text_column_combo.addItems(columns)

                # Auto-select columns if possible
                possible_id_cols = [col for col in columns if col.lower() == 'id']
                if possible_id_cols:
                    self.id_column_combo.setCurrentText(possible_id_cols[0])
                    # Update the file_info with the selected ID column
                    self.data_manager.file_info['id_column'] = possible_id_cols[0]

                possible_text_cols = ["text", "content", "body", "main_text", "abstract", "title", "combined"]
                for col in possible_text_cols:
                    if col in columns:
                        self.text_column_combo.setCurrentText(col)
                        # Update the file_info with the selected text column
                        self.data_manager.file_info['text_column'] = col
                        break

                # Connect the column combo boxes to update file_info when selection changes
                self.text_column_combo.currentTextChanged.connect(self.update_text_column)
                self.id_column_combo.currentTextChanged.connect(self.update_id_column)

                gui_logger.info(f"Data file loaded and previewed: {file_path}")
                
                # Refresh the prompts tab to update data file status
                self.refresh_prompts_tab()
            except Exception as e:
                gui_logger.error(f"Failed to load data file: {e}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to load data file: {str(e)}")

    def update_text_column(self, column_name):
        """Update the text column in file_info when selection changes."""
        if self.data_manager and self.data_manager.file_info:
            self.data_manager.file_info['text_column'] = column_name
            gui_logger.info(f"Updated text column to: {column_name}")
            # Refresh the prompts tab to show the updated text column
            self.refresh_prompts_tab()

    def update_id_column(self, column_name):
        """Update the ID column in file_info when selection changes."""
        if self.data_manager and self.data_manager.file_info:
            self.data_manager.file_info['id_column'] = column_name
            gui_logger.info(f"Updated ID column to: {column_name}")

    def create_log_panel(self):
        self.log_panel = QTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setMaximumHeight(100)
        self.layout.addWidget(self.log_panel)

    def log_message(self, message):
        self.log_panel.append(f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

    def closeEvent(self, event):
        SettingsManager.save_settings(self.settings)
        # Cleanup temp CSV if it exists
        if hasattr(self, 'last_temp_csv_path') and self.last_temp_csv_path:
            try:
                if os.path.exists(self.last_temp_csv_path):
                    os.remove(self.last_temp_csv_path)
                    gui_logger.info(f"Deleted temp CSV: {self.last_temp_csv_path}")
            except Exception as e:
                gui_logger.warning(f"Failed to delete temp CSV: {e}")
        event.accept()

    def create_experiments_tab(self):
        """Creates the Experiments management tab with integrated condition management."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # ----- TOP SECTION: HELP TEXT -----
        help_text = QLabel(
            "Create experiments to test different LLM models and configurations. "
            "Add one or more conditions to each experiment to compare performance."
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: #e9ecef;
                border: 1px solid #ced4da;
                border-radius: 5px;
                margin-bottom: 10px;
                font-size: 13px;
            }
        """)
        layout.addWidget(help_text)

        # Create a horizontal splitter for the main content
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # ----- LEFT PANEL: EXPERIMENT CREATION AND CONDITIONS -----
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        
        # ----- EXPERIMENT DETAILS FORM -----
        exp_group = QGroupBox("Create New Experiment")
        exp_layout = QFormLayout(exp_group)
        exp_layout.setContentsMargins(10, 15, 10, 15)
        exp_layout.setSpacing(10)

        # Experiment Name
        self.experiment_name_edit = QLineEdit()
        self.experiment_name_edit.setPlaceholderText("Enter experiment name")
        exp_layout.addRow("Name:", self.experiment_name_edit)

        # Experiment Description
        self.experiment_desc_edit = QLineEdit()
        self.experiment_desc_edit.setPlaceholderText("Enter a brief description (optional)")
        exp_layout.addRow("Description:", self.experiment_desc_edit)
        
        left_layout.addWidget(exp_group)
        
        # ----- CONDITION CREATION SECTION -----
        conditions_group = QGroupBox("Add Conditions")
        conditions_layout = QVBoxLayout(conditions_group)
        
        # Condition Name (moved to top)
        name_layout = QHBoxLayout()
        name_label = QLabel("Condition Name:")
        self.condition_name_input = QLineEdit()
        self.condition_name_input.setPlaceholderText("Enter a descriptive name for this condition")
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.condition_name_input)
        conditions_layout.addLayout(name_layout)
        
        # Provider and Model Selection
        selection_layout = QHBoxLayout()
        
        # Provider Selection
        provider_form = QFormLayout()
        self.provider_combo = QComboBox()
        provider_form.addRow("Provider:", self.provider_combo)
        selection_layout.addLayout(provider_form)
        
        # Model Selection
        model_form = QFormLayout()
        self.model_combo = QComboBox()
        model_form.addRow("Model:", self.model_combo)
        selection_layout.addLayout(model_form)
        
        # Refresh Models Button
        refresh_button = QPushButton("Refresh Models")
        refresh_button.setIcon(QIcon.fromTheme("view-refresh"))
        refresh_button.clicked.connect(self.refresh_models_list)
        selection_layout.addWidget(refresh_button, 0, Qt.AlignmentFlag.AlignBottom)
        
        conditions_layout.addLayout(selection_layout)
        
        # Max Tokens with info icon
        tokens_layout = QHBoxLayout()
        tokens_label = QLabel("Max Tokens:")
        self.tokens_input = QLineEdit("500")
        self.tokens_input.setValidator(QIntValidator(1, 100000))
        self.tokens_input.setMaximumWidth(100)
        
        # Add clickable info icon for max tokens
        tokens_info = QPushButton()
        info_icon = QIcon.fromTheme("help-about")
        if info_icon.isNull():
            # Fallback to a text-based info icon if theme icon is not available
            tokens_info.setText("ⓘ")
            tokens_info.setStyleSheet("""
                QPushButton {
                    color: #666;
                    font-weight: bold;
                    border: none;
                    padding: 0px;
                    background: transparent;
                }
                QPushButton:hover {
                    color: #000;
                }
            """)
        else:
            tokens_info.setIcon(info_icon)
            tokens_info.setIconSize(QSize(16, 16))
            tokens_info.setStyleSheet("""
                QPushButton {
                    border: none;
                    padding: 0px;
                    background: transparent;
                }
                QPushButton:hover {
                    background: transparent;
                }
            """)
        tokens_info.setToolTip(
            "Tokens are pieces of text that the model processes. As a rough guide:\n\n"
            "• 1 token ≈ 4 characters or 3/4 of a word\n"
            "• A typical paragraph (100 words) ≈ 133 tokens\n"
            "• Consider your prompt length and desired response length\n"
            "• Higher values allow longer responses but may increase costs\n\n"
            "Tip: Start with 500-1000 tokens for most tasks"
        )
        
        tokens_layout.addWidget(tokens_label)
        tokens_layout.addWidget(self.tokens_input)
        tokens_layout.addWidget(tokens_info)
        tokens_layout.addStretch()
        conditions_layout.addLayout(tokens_layout)
        
        # Parameters with sliders
        params_group = QGroupBox("Generation Parameters")
        params_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                margin-top: 1em;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QLineEdit {
                padding: 2px;
                margin: 2px;
            }
            QPushButton {
                margin: 2px;
            }
        """)
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(10)  # Add spacing between parameter rows
        
        # Temperature
        temp_layout = QHBoxLayout()
        temp_layout.setSpacing(5)  # Add spacing between elements
        temp_label = QLabel("Temperature:")
        temp_label.setMinimumWidth(120)  # Ensure consistent label width
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(0, 200)  # 0.0 to 2.0 with 0.01 steps
        self.temp_slider.setValue(70)  # Default 0.7
        self.temp_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.temp_slider.setTickInterval(20)
        self.temp_slider.setFixedHeight(32)  # Match text box height
        self.temp_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.temp_input = QLineEdit("0.7")
        self.temp_input.setValidator(QDoubleValidator(0.0, 2.0, 2))
        self.temp_input.setFixedWidth(60)
        self.temp_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.temp_input.setFixedHeight(32)  # Increased height
        self.temp_input.setStyleSheet("QLineEdit { font-size: 15px; }")
        # Synchronize slider and text box (after both are created)
        def update_temp_input():
            value = self.temp_slider.value() / 100
            self.temp_input.setText(f"{value:.2f}")
        def update_temp_slider():
            try:
                value = float(self.temp_input.text())
                self.temp_slider.setValue(int(value * 100))
            except ValueError:
                pass
        self.temp_slider.valueChanged.connect(update_temp_input)
        self.temp_input.textChanged.connect(update_temp_slider)
        
        # Add info icon for temperature
        temp_info = QPushButton()
        if info_icon.isNull():
            temp_info.setText("ⓘ")
            temp_info.setStyleSheet("""
                QPushButton {
                    color: #666;
                    font-weight: bold;
                    border: none;
                    padding: 2px;
                    background: transparent;
                    min-width: 20px;
                    min-height: 20px;
                }
                QPushButton:hover {
                    color: #000;
                }
            """)
        else:
            temp_info.setIcon(info_icon)
            temp_info.setIconSize(QSize(16, 16))
            temp_info.setStyleSheet("""
                QPushButton {
                    border: none;
                    padding: 2px;
                    background: transparent;
                    min-width: 20px;
                    min-height: 20px;
                }
                QPushButton:hover {
                    background: transparent;
                }
            """)
        temp_info.setToolTip(
            "Controls randomness in the output:\n\n"
            "• 0.0: Most deterministic, focused output\n"
            "• 0.7: Balanced creativity (default)\n"
            "• 1.0: More creative, diverse output\n"
            "• 2.0: Maximum creativity\n\n"
            "Tip: Use lower values (0.0-0.5) for factual responses,\n"
            "higher values (0.7-1.0) for creative tasks"
        )
        
        temp_layout.addWidget(temp_label)
        temp_layout.addWidget(self.temp_slider, 1)  # Give slider stretch
        temp_layout.addWidget(self.temp_input, 0)
        temp_layout.addWidget(temp_info, 0)
        params_layout.addLayout(temp_layout)
        
        # Top P
        top_p_layout = QHBoxLayout()
        top_p_layout.setSpacing(5)  # Add spacing between elements
        top_p_label = QLabel("Top P:")
        top_p_label.setMinimumWidth(120)  # Ensure consistent label width
        self.top_p_slider = QSlider(Qt.Orientation.Horizontal)
        self.top_p_slider.setRange(0, 100)  # 0.0 to 1.0 with 0.01 steps
        self.top_p_slider.setValue(100)  # Default 1.0
        self.top_p_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.top_p_slider.setTickInterval(20)
        self.top_p_slider.setFixedHeight(32)  # Match text box height
        self.top_p_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.top_p_input = QLineEdit("1.0")
        self.top_p_input.setValidator(QDoubleValidator(0.0, 1.0, 2))
        self.top_p_input.setFixedWidth(60)
        self.top_p_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.top_p_input.setFixedHeight(32)  # Increased height
        self.top_p_input.setStyleSheet("QLineEdit { font-size: 15px; }")
        # Synchronize slider and text box (after both are created)
        def update_top_p_input():
            value = self.top_p_slider.value() / 100
            self.top_p_input.setText(f"{value:.2f}")
        def update_top_p_slider():
            try:
                value = float(self.top_p_input.text())
                self.top_p_slider.setValue(int(value * 100))
            except ValueError:
                pass
        self.top_p_slider.valueChanged.connect(update_top_p_input)
        self.top_p_input.textChanged.connect(update_top_p_slider)
        
        # Add info icon for top P
        top_p_info = QPushButton()
        if info_icon.isNull():
            top_p_info.setText("ⓘ")
            top_p_info.setStyleSheet("""
                QPushButton {
                    color: #666;
                    font-weight: bold;
                    border: none;
                    padding: 2px;
                    background: transparent;
                    min-width: 20px;
                    min-height: 20px;
                }
                QPushButton:hover {
                    color: #000;
                }
            """)
        else:
            top_p_info.setIcon(info_icon)
            top_p_info.setIconSize(QSize(16, 16))
            top_p_info.setStyleSheet("""
                QPushButton {
                    border: none;
                    padding: 2px;
                    background: transparent;
                    min-width: 20px;
                    min-height: 20px;
                }
                QPushButton:hover {
                    background: transparent;
                }
            """)
        top_p_info.setToolTip(
            "Controls diversity via nucleus sampling:\n\n"
            "• 0.1: Very focused, deterministic output\n"
            "• 0.5: Balanced diversity\n"
            "• 1.0: Maximum diversity (default)\n\n"
            "Tip: Use lower values (0.1-0.5) for focused responses,\n"
            "higher values (0.7-1.0) for diverse outputs"
        )
        
        top_p_layout.addWidget(top_p_label)
        top_p_layout.addWidget(self.top_p_slider, 1)  # Give slider stretch
        top_p_layout.addWidget(self.top_p_input, 0)
        top_p_layout.addWidget(top_p_info, 0)
        params_layout.addLayout(top_p_layout)
        
        # Frequency Penalty
        freq_penalty_layout = QHBoxLayout()
        freq_penalty_layout.setSpacing(5)  # Add spacing between elements
        freq_penalty_label = QLabel("Frequency Penalty:")
        freq_penalty_label.setMinimumWidth(120)  # Ensure consistent label width
        self.freq_penalty_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_penalty_slider.setRange(-200, 200)  # -2.0 to 2.0 with 0.01 steps
        self.freq_penalty_slider.setValue(0)  # Default 0.0
        self.freq_penalty_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.freq_penalty_slider.setTickInterval(100)
        self.freq_penalty_slider.setFixedHeight(32)  # Match text box height
        self.freq_penalty_slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.freq_penalty_input = QLineEdit("0.0")
        self.freq_penalty_input.setValidator(QDoubleValidator(-2.0, 2.0, 2))
        self.freq_penalty_input.setFixedWidth(60)
        self.freq_penalty_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.freq_penalty_input.setFixedHeight(32)  # Increased height
        self.freq_penalty_input.setStyleSheet("QLineEdit { font-size: 15px; }")
        # Synchronize slider and text box (after both are created)
        def update_freq_penalty_input():
            value = self.freq_penalty_slider.value() / 100
            self.freq_penalty_input.setText(f"{value:.2f}")
        def update_freq_penalty_slider():
            try:
                value = float(self.freq_penalty_input.text())
                self.freq_penalty_slider.setValue(int(value * 100))
            except ValueError:
                pass
        self.freq_penalty_slider.valueChanged.connect(update_freq_penalty_input)
        self.freq_penalty_input.textChanged.connect(update_freq_penalty_slider)
        
        # Add info icon for frequency penalty
        freq_penalty_info = QPushButton()
        if info_icon.isNull():
            freq_penalty_info.setText("ⓘ")
            freq_penalty_info.setStyleSheet("""
                QPushButton {
                    color: #666;
                    font-weight: bold;
                    border: none;
                    padding: 2px;
                    background: transparent;
                    min-width: 20px;
                    min-height: 20px;
                }
                QPushButton:hover {
                    color: #000;
                }
            """)
        else:
            freq_penalty_info.setIcon(info_icon)
            freq_penalty_info.setIconSize(QSize(16, 16))
            freq_penalty_info.setStyleSheet("""
                QPushButton {
                    border: none;
                    padding: 2px;
                    background: transparent;
                    min-width: 20px;
                    min-height: 20px;
                }
                QPushButton:hover {
                    background: transparent;
                }
            """)
        freq_penalty_info.setToolTip(
            "Controls repetition in the output:\n\n"
            "• -2.0: More likely to repeat\n"
            "• 0.0: No effect on repetition (default)\n"
            "• 2.0: Less likely to repeat\n\n"
            "Tip: Use positive values (0.5-2.0) to reduce repetition,\n"
            "negative values (-0.5 to -2.0) to encourage repetition"
        )
        
        freq_penalty_layout.addWidget(freq_penalty_label)
        freq_penalty_layout.addWidget(self.freq_penalty_slider, 1)  # Give slider stretch
        freq_penalty_layout.addWidget(self.freq_penalty_input, 0)
        freq_penalty_layout.addWidget(freq_penalty_info, 0)
        params_layout.addLayout(freq_penalty_layout)
        
        conditions_layout.addWidget(params_group)
        
        # Add Condition Button
        add_condition_button = QPushButton("Add Condition")
        add_condition_button.setIcon(QIcon.fromTheme("list-add"))
        add_condition_button.clicked.connect(self.add_condition_from_form)
        conditions_layout.addWidget(add_condition_button, 0, Qt.AlignmentFlag.AlignRight)
        
        left_layout.addWidget(conditions_group)
        
        # ----- CONDITIONS LIST SECTION -----
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
        
        # Buttons for conditions table
        condition_buttons_layout = QHBoxLayout()
        
        delete_condition_button = QPushButton("Delete Selected")
        delete_condition_button.setIcon(QIcon.fromTheme("edit-delete"))
        delete_condition_button.clicked.connect(self.delete_selected_conditions)
        
        clear_conditions_button = QPushButton("Clear All")
        clear_conditions_button.setIcon(QIcon.fromTheme("edit-clear"))
        clear_conditions_button.clicked.connect(self.clear_all_conditions)
        
        condition_buttons_layout.addWidget(delete_condition_button)
        condition_buttons_layout.addWidget(clear_conditions_button)
        condition_buttons_layout.addStretch()
        
        save_experiment_button = QPushButton("Save Experiment")
        save_experiment_button.setIcon(QIcon.fromTheme("document-save"))
        save_experiment_button.clicked.connect(self.add_experiment_from_form)
        condition_buttons_layout.addWidget(save_experiment_button)
        
        conditions_list_layout.addLayout(condition_buttons_layout)
        
        left_layout.addWidget(conditions_list_group)
        
        # ----- RIGHT PANEL: EXPERIMENTS TABLE -----
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        # Table Label
        experiments_label = QLabel("Existing Experiments")
        experiments_label.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(experiments_label)
        
        # Experiments table
        self.experiments_table = QTableWidget()
        self.experiments_table.setAlternatingRowColors(True)
        self.experiments_table.setColumnCount(4)
        self.experiments_table.setHorizontalHeaderLabels(["ID", "Name", "Description", "Conditions"])
        self.experiments_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.experiments_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.experiments_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.experiments_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.experiments_table.verticalHeader().setVisible(False)
        # Connect double-click to view details
        self.experiments_table.doubleClicked.connect(self.show_experiment_details_for_selected)
        right_layout.addWidget(self.experiments_table)
        
        # Buttons for experiments table
        experiment_buttons_layout = QHBoxLayout()
        
        view_details_button = QPushButton("View Details")
        view_details_button.setIcon(QIcon.fromTheme("document-properties"))
        view_details_button.clicked.connect(self.show_experiment_details_for_selected)
        
        delete_button = QPushButton("Delete Selected")
        delete_button.setIcon(QIcon.fromTheme("edit-delete"))
        delete_button.clicked.connect(self.delete_selected_experiments)
        
        run_button = QPushButton("Run Selected")
        run_button.setIcon(QIcon.fromTheme("system-run"))
        run_button.clicked.connect(self.show_run_experiment_dialog_for_selected)
        
        experiment_buttons_layout.addWidget(view_details_button)
        experiment_buttons_layout.addWidget(delete_button)
        experiment_buttons_layout.addStretch()
        experiment_buttons_layout.addWidget(run_button)
        
        right_layout.addLayout(experiment_buttons_layout)
        
        # Add panels to the main splitter
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        
        # Set initial sizes
        main_splitter.setSizes([500, 500])
        
        # Add the splitter to the main layout
        layout.addWidget(main_splitter)
        
        # Initialize the panel and load data
        self.initialize_experiments_tab()
        
        return tab

    def initialize_experiments_tab(self):
        """Initialize the experiments tab with data."""
        # Populate provider dropdown
        if hasattr(self, 'api_key_manager') and self.api_key_manager:
            providers = self.api_key_manager.list_providers()
            self.provider_combo.clear()
            self.provider_combo.addItems(providers)
            
            # Connect signal for provider change
            self.provider_combo.currentTextChanged.connect(self.update_models_for_provider)
            
            # Initial update of models list if a provider is selected
            if providers and self.provider_combo.currentText():
                self.update_models_for_provider(self.provider_combo.currentText())
        
        # Load existing experiments
        self.load_experiments()

    def update_models_for_provider(self, provider):
        """Update the model dropdown based on selected provider."""
        if not provider:
            return
            
        self.model_combo.clear()
        
        try:
            api_key = self.experiment_manager.api_key_manager.get_key(provider)
            if not api_key:
                gui_logger.warning(f"No API key found for provider: {provider}")
                return
                
            # Import ProviderFactory if needed
            from qually_tool import ProviderFactory
            
            # Create provider instance and get models
            provider_instance = ProviderFactory.create_provider(provider, api_key)
            if not provider_instance:
                gui_logger.error(f"Failed to create provider instance for {provider}")
                return
                
            # Show a small indicator that we're loading
            self.statusBar().showMessage(f"Loading models for {provider}...")
            QApplication.processEvents()  # Ensure UI updates
            
            # Get models
            models = provider_instance.get_available_models()
            self.model_combo.addItems(models)
            
            self.statusBar().showMessage(f"Loaded {len(models)} models for {provider}", 3000)
        except Exception as e:
            gui_logger.error(f"Error fetching models for {provider}: {e}", exc_info=True)
            self.statusBar().showMessage(f"Error loading models for {provider}", 3000)

    def refresh_models_list(self):
        """Refresh the models list for the current provider."""
        provider = self.provider_combo.currentText()
        if provider:
            self.update_models_for_provider(provider)

    def add_condition_from_form(self):
        """Add a condition using the values from the form fields."""
        # Validate inputs
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
            # Get parameter values
            parameters = {
                "temperature": float(self.temp_input.text()),
                "max_tokens": int(self.tokens_input.text()),
                "top_p": float(self.top_p_input.text()),
                "frequency_penalty": float(self.freq_penalty_input.text()),
                "presence_penalty": float(self.freq_penalty_input.text())
            }
            
            # Add to conditions table
            row = self.conditions_table.rowCount()
            self.conditions_table.insertRow(row)
            self.conditions_table.setItem(row, 0, QTableWidgetItem(condition_name))
            self.conditions_table.setItem(row, 1, QTableWidgetItem(provider))
            self.conditions_table.setItem(row, 2, QTableWidgetItem(model))
            self.conditions_table.setItem(row, 3, QTableWidgetItem(json.dumps(parameters)))
            
            # Clear inputs (except provider/model for easy addition of multiple conditions)
            self.condition_name_input.clear()
            
            gui_logger.info(f"Added condition: {condition_name} ({provider}/{model})")
            self.statusBar().showMessage(f"Added condition: {condition_name}", 3000)
            
        except Exception as e:
            gui_logger.error(f"Error adding condition: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to add condition: {str(e)}")
        
    def add_experiment_from_form(self):
        """Add a new experiment using the form fields."""
        if not self.experiment_manager:
            QMessageBox.critical(self, "Error", "Experiment manager not initialized.")
            return

        try:
            name = self.experiment_name_edit.text().strip()
            description = self.experiment_desc_edit.text().strip()

            if not name:
                QMessageBox.warning(self, "Error", "Experiment name cannot be empty.")
                return

            # Get conditions from the conditions table
            conditions = []
            for row in range(self.conditions_table.rowCount()):
                condition = {
                    "name": self.conditions_table.item(row, 0).text(),
                    "provider": self.conditions_table.item(row, 1).text(),
                    "model": self.conditions_table.item(row, 2).text(),
                    "parameters": json.loads(self.conditions_table.item(row, 3).text())
                }
                conditions.append(condition)

            # Create the experiment with all conditions
            experiment_id = self.experiment_manager.create_experiment_with_conditions(
                name=name,
                description=description,
                conditions=conditions
            )
            
            if experiment_id:
                # Clear the form fields
                self.experiment_name_edit.clear()
                self.experiment_desc_edit.clear()
                self.conditions_table.setRowCount(0)
                
                # Refresh the table
                self.load_experiments()
                
                QMessageBox.information(self, "Success", f"Experiment created successfully with ID: {experiment_id}")
                gui_logger.info(f"Created new experiment with ID: {experiment_id}")
            else:
                QMessageBox.critical(self, "Error", "Failed to create experiment.")
                gui_logger.error("Failed to create experiment")

        except Exception as e:
            gui_logger.error(f"Error creating experiment: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to create experiment: {str(e)}")

    def load_experiments(self):
        """Load experiments from experiment manager into the UI table."""
        if not hasattr(self, 'experiment_manager'):
            gui_logger.warning("Experiment manager not initialized")
            return
            
        self.experiments_table.setRowCount(0)
        experiments = self.experiment_manager.list_experiments()
        
        for experiment in experiments:
            row = self.experiments_table.rowCount()
            self.experiments_table.insertRow(row)
            
            # Add experiment ID
            id_item = QTableWidgetItem(experiment['id'])
            id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.experiments_table.setItem(row, 0, id_item)
            
            # Add experiment name
            name_item = QTableWidgetItem(experiment['name'])
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.experiments_table.setItem(row, 1, name_item)
            
            # Add experiment description
            desc_item = QTableWidgetItem(experiment.get('description', ''))
            desc_item.setFlags(desc_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.experiments_table.setItem(row, 2, desc_item)
            
            # Add conditions count
            conditions_count = len(experiment.get('conditions', []))
            conditions_item = QTableWidgetItem(str(conditions_count))
            conditions_item.setFlags(conditions_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.experiments_table.setItem(row, 3, conditions_item)

    def show_add_condition_dialog(self):
        """Show dialog to add a condition to the current experiment."""
        if not self.experiment_manager:
            QMessageBox.critical(self, "Error", "Experiment manager not initialized.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Add Condition")
        layout = QVBoxLayout(dialog)
        
        # Provider selection
        provider_layout = QHBoxLayout()
        provider_label = QLabel("Provider*:")
        provider_label.setToolTip("Required: Select the LLM provider")
        provider_combo = QComboBox()
        available_providers = self.experiment_manager.api_key_manager.list_providers()
        if not available_providers:
            QMessageBox.critical(self, "Error", "No API keys configured. Please set up API keys first.")
            return
        provider_combo.addItems(available_providers)
        provider_layout.addWidget(provider_label)
        provider_layout.addWidget(provider_combo)
        layout.addLayout(provider_layout)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model*:")
        model_label.setToolTip("Required: Select the model to use")
        model_combo = QComboBox()
        model_layout.addWidget(model_label)
        model_layout.addWidget(model_combo)
        layout.addLayout(model_layout)
        
        # Parameters
        parameters_group = QGroupBox("Parameters")
        parameters_layout = QVBoxLayout(parameters_group)
        
        # Temperature
        temp_layout = QHBoxLayout()
        temp_label = QLabel("Temperature:")
        temp_label.setToolTip("Controls randomness (0.0 to 2.0)")
        temp_input = QLineEdit("0.7")
        temp_input.setValidator(QDoubleValidator(0.0, 2.0, 2))
        temp_layout.addWidget(temp_label)
        temp_layout.addWidget(temp_input)
        parameters_layout.addLayout(temp_layout)
        
        # Max Tokens
        tokens_layout = QHBoxLayout()
        tokens_label = QLabel("Max Tokens:")
        tokens_label.setToolTip("Maximum number of tokens to generate")
        tokens_input = QLineEdit("500")
        tokens_input.setValidator(QIntValidator(1, 100000))
        tokens_layout.addWidget(tokens_label)
        tokens_layout.addWidget(tokens_input)
        parameters_layout.addLayout(tokens_layout)
        
        # Top P
        top_p_layout = QHBoxLayout()
        top_p_label = QLabel("Top P:")
        top_p_label.setToolTip("Controls diversity (0.0 to 1.0)")
        top_p_input = QLineEdit("1.0")
        top_p_input.setValidator(QDoubleValidator(0.0, 1.0, 2))
        top_p_layout.addWidget(top_p_label)
        top_p_layout.addWidget(top_p_input)
        parameters_layout.addLayout(top_p_layout)
        
        # Frequency Penalty
        freq_penalty_layout = QHBoxLayout()
        freq_penalty_label = QLabel("Frequency Penalty:")
        freq_penalty_label.setToolTip("Controls repetition (-2.0 to 2.0)")
        freq_penalty_input = QLineEdit("0.0")
        freq_penalty_input.setValidator(QDoubleValidator(-2.0, 2.0, 2))
        freq_penalty_layout.addWidget(freq_penalty_label)
        freq_penalty_layout.addWidget(freq_penalty_input)
        parameters_layout.addLayout(freq_penalty_layout)
        
        # Presence Penalty
        pres_penalty_layout = QHBoxLayout()
        pres_penalty_label = QLabel("Presence Penalty:")
        pres_penalty_label.setToolTip("Controls topic diversity (-2.0 to 2.0)")
        pres_penalty_input = QLineEdit("0.0")
        pres_penalty_input.setValidator(QDoubleValidator(-2.0, 2.0, 2))
        pres_penalty_layout.addWidget(pres_penalty_label)
        pres_penalty_layout.addWidget(pres_penalty_input)
        parameters_layout.addLayout(pres_penalty_layout)
        
        layout.addWidget(parameters_group)
        
        # Condition name
        name_layout = QHBoxLayout()
        name_label = QLabel("Condition Name*:")
        name_label.setToolTip("Required: Name for this condition")
        name_input = QLineEdit()
        name_layout.addWidget(name_label)
        name_layout.addWidget(name_input)
        layout.addLayout(name_layout)
        
        # Add required field note
        required_note = QLabel("* Required fields")
        required_note.setStyleSheet("color: #dc3545; font-style: italic;")
        layout.addWidget(required_note)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        add_button = QPushButton("Add")
        cancel_button = QPushButton("Cancel")
        buttons_layout.addWidget(add_button)
        buttons_layout.addWidget(cancel_button)
        layout.addLayout(buttons_layout)
        
        def update_models():
            provider = provider_combo.currentText()
            model_combo.clear()
            try:
                api_key = self.experiment_manager.api_key_manager.get_key(provider)
                if api_key:
                    provider_instance = ProviderFactory.create_provider(provider, api_key)
                    models = provider_instance.get_available_models()
                    for model in models:
                        model_combo.addItem(model)
            except Exception as e:
                QMessageBox.warning(dialog, "Warning", f"Could not fetch models: {str(e)}")
        
        provider_combo.currentTextChanged.connect(update_models)
        
        def add_condition():
            try:
                name = name_input.text().strip()
                if not name:
                    QMessageBox.warning(dialog, "Warning", "Condition name is required")
                    return
                
                if not model_combo.currentText():
                    QMessageBox.warning(dialog, "Warning", "Model selection is required")
                    return
                
                parameters = {
                    "temperature": float(temp_input.text()),
                    "max_tokens": int(tokens_input.text()),
                    "top_p": float(top_p_input.text()),
                    "frequency_penalty": float(freq_penalty_input.text()),
                    "presence_penalty": float(pres_penalty_input.text())
                }
                
                # Add the condition to the conditions table
                row = self.conditions_table.rowCount()
                self.conditions_table.insertRow(row)
                self.conditions_table.setItem(row, 0, QTableWidgetItem(name))
                self.conditions_table.setItem(row, 1, QTableWidgetItem(provider_combo.currentText()))
                self.conditions_table.setItem(row, 2, QTableWidgetItem(model_combo.currentText()))
                self.conditions_table.setItem(row, 3, QTableWidgetItem(json.dumps(parameters)))
                
                dialog.accept()
                gui_logger.info(f"Added condition '{name}'")
            except Exception as e:
                QMessageBox.critical(dialog, "Error", f"Failed to add condition: {str(e)}")
        
        add_button.clicked.connect(add_condition)
        cancel_button.clicked.connect(dialog.reject)
        
        # Initial model update
        update_models()
        
        dialog.exec()

    def delete_selected_conditions(self):
        """Delete the currently selected conditions."""
        selected_rows = self.conditions_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select conditions to delete.")
            return

        # Remove rows in reverse order to maintain correct indices
        for row in sorted([index.row() for index in selected_rows], reverse=True):
            self.conditions_table.removeRow(row)

    def clear_all_conditions(self):
        """Clear all conditions from the table."""
        reply = QMessageBox.question(
            self,
            "Confirm Clear All",
            "Are you sure you want to delete ALL conditions? This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.conditions_table.setRowCount(0)

    def show_experiment_details_for_selected(self):
        """Show details of the selected experiment in a message box."""
        selected_rows = self.experiments_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "Selection Required", "Please select an experiment from the table first.")
            return
        if len(selected_rows) > 1:
            QMessageBox.warning(self, "Selection Error", "Please select only one experiment.")
            return
            
        selected_row = selected_rows[0].row()
        experiment_id = self.experiments_table.item(selected_row, 0).text()
        self.show_experiment_details(experiment_id)

    def show_experiment_details(self, experiment_id: str):
        """Show detailed view of an experiment."""
        experiment = self.experiment_manager.get_experiment(experiment_id)
        if not experiment:
            QMessageBox.warning(self, "Error", f"Experiment with ID {experiment_id} not found.")
            return

        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Experiment Details: {experiment['name']}")
        dialog.setMinimumWidth(600)
        layout = QVBoxLayout(dialog)

        # Basic info
        info_group = QGroupBox("Basic Information")
        info_layout = QFormLayout(info_group)
        info_layout.addRow("ID:", QLabel(experiment['id']))
        info_layout.addRow("Name:", QLabel(experiment['name']))
        info_layout.addRow("Description:", QLabel(experiment.get('description', 'N/A')))
        info_layout.addRow("Created At:", QLabel(experiment.get('created_at', 'N/A')))
        layout.addWidget(info_group)

        # Conditions
        conditions_group = QGroupBox("Conditions")
        conditions_layout = QVBoxLayout(conditions_group)
        
        conditions_table = QTableWidget()
        conditions_table.setColumnCount(4)
        conditions_table.setHorizontalHeaderLabels(["Name", "Provider", "Model", "Parameters"])
        conditions_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        conditions_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        conditions_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        conditions_table.verticalHeader().setVisible(False)

        conditions = experiment.get('conditions', [])
        conditions_table.setRowCount(len(conditions))
        for row, condition in enumerate(conditions):
            conditions_table.setItem(row, 0, QTableWidgetItem(condition['name']))
            conditions_table.setItem(row, 1, QTableWidgetItem(condition['provider']))
            conditions_table.setItem(row, 2, QTableWidgetItem(condition['model']))
            conditions_table.setItem(row, 3, QTableWidgetItem(json.dumps(condition.get('parameters', {}), indent=2)))

        conditions_layout.addWidget(conditions_table)
        layout.addWidget(conditions_group)

        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)
        
        results_table = QTableWidget()
        results_table.setColumnCount(3)
        results_table.setHorizontalHeaderLabels(["Timestamp", "Status", "File"])
        results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        results_table.verticalHeader().setVisible(False)

        # Get results files
        results_files = [f for f in os.listdir(self.experiment_manager.results_folder) 
                        if f.startswith(f"results_{experiment_id}_") and f.endswith(".json")]
        
        results_table.setRowCount(len(results_files))
        for row, file in enumerate(results_files):
            try:
                with open(os.path.join(self.experiment_manager.results_folder, file), 'r') as f:
                    result = json.load(f)
                    results_table.setItem(row, 0, QTableWidgetItem(result.get('timestamp', 'N/A')))
                    results_table.setItem(row, 1, QTableWidgetItem("Completed" if not result.get('error') else "Failed"))
                    results_table.setItem(row, 2, QTableWidgetItem(file))
            except Exception as e:
                gui_logger.error(f"Error loading result file {file}: {e}")
                results_table.setItem(row, 0, QTableWidgetItem("N/A"))
                results_table.setItem(row, 1, QTableWidgetItem("Error"))
                results_table.setItem(row, 2, QTableWidgetItem(file))

        results_layout.addWidget(results_table)
        layout.addWidget(results_group)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.exec()

    def show_run_experiment_dialog_for_selected(self):
        """Show dialog to run the selected experiment."""
        selected_items = self.experiments_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select an experiment to run.")
            return

        # Get the experiment ID from the selected row
        row = selected_items[0].row()
        experiment_id = self.experiments_table.item(row, 0).text()

        # Get the experiment details
        experiment = self.experiment_manager.get_experiment(experiment_id)
        if not experiment:
            QMessageBox.critical(self, "Error", "Selected experiment not found.")
            return

        # Get all prompts
        prompts = self.prompt_manager.list_prompts()
        if not prompts:
            QMessageBox.warning(self, "No Prompts", "No prompts available to run.")
            return

        # Check if we have data file info and an ID column selected
        data_prompts = []
        if self.data_manager and self.data_manager.has_data() and self.data_manager.file_info.get('id_column'):
            # Get the list of IDs from the data file
            data_ids = set(self.data_manager.data[self.data_manager.file_info['id_column']].astype(str))
            
            # Find all prompts that have a data ID tag
            for prompt in prompts:
                # Get data ID from tags
                data_id = None
                for tag in prompt.get('tags', []):
                    if tag.startswith('data_id_'):
                        data_id = tag.replace('data_id_', '')
                        break
                
                # If this prompt has a data ID that matches our data file
                if data_id in data_ids:
                    data_prompts.append(prompt)
                    gui_logger.info(f"Found matching prompt: ID={prompt['id']}, data_id={data_id}")
                else:
                    gui_logger.debug(f"No match for prompt: ID={prompt['id']}, data_id={data_id}")

        if not data_prompts:
            gui_logger.warning("No matching data prompts found. Available IDs in data: " + 
                             ", ".join(str(id) for id in data_ids))
            reply = QMessageBox.warning(
                self,
                "Warning",
                "No prompts matching your data file IDs were found. This means the experiment will only run with a single prompt instead of processing all rows in your data file.\n\n"
                "Would you like to continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        # Create a simple dialog just to show progress
        dialog = QDialog(self)
        dialog.setWindowTitle("Running Experiment")
        layout = QVBoxLayout(dialog)
        
        # Add progress bar (set to shuttle mode)
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 0)  # Set to 0 to create shuttle effect
        progress_bar.setTextVisible(False)  # Hide percentage text
        layout.addWidget(progress_bar)
        
        # Add status label
        status_label = QLabel("Preparing to run experiment...")
        layout.addWidget(status_label)
        
        # Add cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        layout.addWidget(cancel_button)
        
        # Show the dialog
        dialog.show()
        QApplication.processEvents()

        # Use data prompts if available, otherwise use all prompts
        prompt_ids = [p['id'] for p in (data_prompts if data_prompts else prompts)]

        # Create and start the experiment runner
        self.experiment_runner = ExperimentRunner(self.experiment_manager, experiment_id, prompt_ids)
        
        # Connect progress signal to update status label
        def update_progress(current, total):
            status_label.setText(f"Processing prompt {current}/{total}")
            QApplication.processEvents()
            
        self.experiment_runner.progress.connect(update_progress)
        self.experiment_runner.finished.connect(lambda results: self.handle_experiment_results(results, dialog))
        self.experiment_runner.error.connect(lambda error: self.handle_experiment_error(error, dialog))
        self.experiment_runner.log_message.connect(self.log_message)
        
        # Start the experiment
        self.experiment_runner.start()

    def handle_experiment_results(self, results, dialog):
        """Handle experiment completion."""
        dialog.accept()
        if results:
            QMessageBox.information(
                self,
                "Experiment Complete",
                f"Experiment completed successfully.\nResults saved to: {results.get('results_file', 'results folder')}"
            )
        else:
            QMessageBox.warning(self, "Experiment Complete", "Experiment completed but no results were generated.")

    def handle_experiment_error(self, error, dialog):
        """Handle experiment error."""
        dialog.reject()
        QMessageBox.critical(self, "Error", f"Experiment failed: {error}")

    def create_prompts_tab(self):
        """Creates the Prompts management tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Warning message if no data is loaded
        if not self.data_manager or not self.data_manager.has_data():
            warning_label = QLabel("⚠️ Warning: No data file loaded. Please import data in the Data Files tab before working with prompts.")
            warning_label.setStyleSheet("color: #856404; background-color: #fff3cd; padding: 10px; border: 1px solid #ffeeba; border-radius: 4px;")
            warning_label.setWordWrap(True)
            layout.addWidget(warning_label)
        else:
            # Show data file info
            info_label = QLabel(
                f"📊 Data File Info:\n"
                f"File: {Path(self.data_manager.file_info['file_path']).name}\n"
                f"Text Column: {self.data_manager.file_info.get('text_column', 'Not selected')}\n"
                f"Total Rows: {self.data_manager.file_info['total_rows']}\n\n"
                f"To create prompts for each row in your data file:\n"
                f"1. Set the system prompt (optional)\n"
                f"2. Set the prepend text (optional)\n"
                f"3. Set the prompt text\n"
                f"4. Click 'Create Prompts from Data File'"
            )
            info_label.setStyleSheet("color: #155724; background-color: #d4edda; padding: 10px; border: 1px solid #c3e6cb; border-radius: 4px;")
            info_label.setWordWrap(True)
            layout.addWidget(info_label)

        # Create horizontal splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # Left side: Form for creating prompts
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        form_layout.setContentsMargins(10, 10, 10, 10)
        form_layout.setSpacing(10)

        # Help text explaining the format
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
        help_text.setStyleSheet("""
            QLabel {
                padding: 10px;
                background-color: #e9ecef;
                border: 1px solid #ced4da;
                border-radius: 5px;
                margin-bottom: 10px;
                font-size: 13px;
            }
        """)
        form_layout.addWidget(help_text)

        # System Prompt
        system_prompt_label = QLabel("System Prompt:")
        self.system_prompt_edit = QTextEdit()
        self.system_prompt_edit.setPlaceholderText(
            "Optional: Set the model's behavior. Example: 'You are a helpful assistant that answers questions about text.'"
        )
        self.system_prompt_edit.setMaximumHeight(100)
        form_layout.addWidget(system_prompt_label)
        form_layout.addWidget(self.system_prompt_edit)

        # Prepend Text
        prepend_label = QLabel("Prepend Text:")
        self.prepend_edit = QTextEdit()
        self.prepend_edit.setPlaceholderText(
            "Optional: Text to add before each data row. Example: 'Consider this text: '"
        )
        self.prepend_edit.setMaximumHeight(100)
        form_layout.addWidget(prepend_label)
        form_layout.addWidget(self.prepend_edit)

        # Prompt Text
        prompt_label = QLabel("Prompt:")
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText(
            "Required: The main instruction for the model. Example: 'What is the main topic of this text?'"
        )
        self.prompt_edit.setMaximumHeight(100)
        form_layout.addWidget(prompt_label)
        form_layout.addWidget(self.prompt_edit)

        # Create prompts from data file button
        create_from_data_button = QPushButton("Create Prompts from Data File")
        create_from_data_button.setIcon(QIcon.fromTheme("document-new"))
        create_from_data_button.clicked.connect(self.create_prompts_from_data)
        form_layout.addWidget(create_from_data_button, 0, Qt.AlignmentFlag.AlignRight)

        # Add some stretch to push everything up
        form_layout.addStretch()

        # Right side: Table and buttons
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.setContentsMargins(10, 10, 10, 10)
        table_layout.setSpacing(10)

        # Prompts table
        self.prompts_table = QTableWidget()
        self.prompts_table.setAlternatingRowColors(True)
        self.prompts_table.setColumnCount(6)  # Changed from 5 to 6
        self.prompts_table.setHorizontalHeaderLabels(["ID", "Data ID", "System Prompt", "Prepend Text", "Prompt Text", "Tags"])
        self.prompts_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.prompts_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.prompts_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.prompts_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.prompts_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.prompts_table.verticalHeader().setVisible(False)
        table_layout.addWidget(self.prompts_table)

        # Buttons layout
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 5, 0, 0)

        delete_selected_button = QPushButton("Delete Selected")
        delete_selected_button.setIcon(QIcon.fromTheme("edit-delete"))
        delete_selected_button.clicked.connect(self.delete_selected_prompts)

        clear_all_button = QPushButton("Clear All")
        clear_all_button.setIcon(QIcon.fromTheme("edit-clear"))
        clear_all_button.clicked.connect(self.clear_all_prompts)

        export_button = QPushButton("Export Prompts")
        export_button.setIcon(QIcon.fromTheme("document-save"))
        export_button.clicked.connect(self.export_prompts)

        buttons_layout.addWidget(delete_selected_button)
        buttons_layout.addWidget(clear_all_button)
        buttons_layout.addStretch()
        buttons_layout.addWidget(export_button)
        table_layout.addWidget(buttons_widget)

        # Add widgets to splitter
        splitter.addWidget(form_widget)
        splitter.addWidget(table_widget)
        
        # Set initial sizes (form takes 1/3, table takes 2/3)
        splitter.setSizes([300, 600])

        # Load existing prompts
        self.load_prompts()

        return tab

    def create_prompts_from_data(self):
        """Create prompts from the loaded data file."""
        if not self.prompt_manager:
            QMessageBox.critical(self, "Error", "Prompt manager not initialized.")
            return

        if not self.data_manager or not self.data_manager.has_data():
            QMessageBox.warning(self, "No Data", "Please load a data file first.")
            return

        # Check if text column is selected
        if not self.data_manager.file_info.get('text_column'):
            QMessageBox.warning(
                self,
                "No Text Column Selected",
                "Please select a text column in the Data Files tab first.\n\n"
                "Available columns:\n" + 
                "\n".join(f"- {col}" for col in self.data_manager.data.columns)
            )
            return

        # Check if ID column is selected
        if not self.data_manager.file_info.get('id_column'):
            QMessageBox.warning(
                self,
                "No ID Column Selected",
                "Please select an ID column in the Data Files tab first."
            )
            return

        try:
            system_prompt = self.system_prompt_edit.toPlainText().strip()
            prepend_text = self.prepend_edit.toPlainText().strip()
            prompt_text = self.prompt_edit.toPlainText().strip()

            if not prompt_text:
                QMessageBox.warning(self, "Error", "Prompt text cannot be empty.")
                return

            # Load data based on file type
            file_path = self.data_manager.file_info['file_path']
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.csv':
                data = pd.read_csv(file_path)
            elif file_extension == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                if isinstance(json_data, list):
                    data = pd.DataFrame(json_data)
                else:
                    data = pd.DataFrame([json_data])
            elif file_extension in ['.xls', '.xlsx']:
                data = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            text_column = self.data_manager.file_info['text_column']
            id_column = self.data_manager.file_info['id_column']

            for index, row in data.iterrows():
                text_content = str(row[text_column])
                data_id = str(row[id_column])
                
                # Generate a unique prompt ID with timestamp only
                timestamp = int(time.time())
                prompt_id = f"prompt_{timestamp}_row_{index}"  # Use row index in ID
                
                # Create the full prompt text with prepend text included
                full_text = text_content
                if prepend_text:
                    full_text = f"{prepend_text} {text_content}"
                full_prompt = f"{full_text}\n{prompt_text}"
                
                # Create the prompt with data ID in tags
                self.prompt_manager.add_prompt(
                    prompt_id=prompt_id,
                    prompt_text=full_prompt,  # Include prepend text in the prompt_text
                    system_prompt=system_prompt,
                    prepend_text=prepend_text,  # Keep prepend text separate for display
                    tags=[f"data_id_{data_id}", f"row_{index}"]  # Store both data ID and row index in tags
                )

            # Save prompts to file
            if self.prompt_manager.export_prompts():
                # Clear the form fields
                self.system_prompt_edit.clear()
                self.prepend_edit.clear()
                self.prompt_edit.clear()
                
                # Refresh the table
                self.load_prompts()
                
                QMessageBox.information(
                    self,
                    "Success",
                    f"Created prompts for all rows in the data file.\n"
                    f"Text column used: {text_column}\n"
                    f"ID column used: {id_column}\n"
                    f"Total prompts created: {len(data)}"
                )
                gui_logger.info("Created prompts from data file")
            else:
                QMessageBox.critical(self, "Error", "Failed to save prompts to file.")
                gui_logger.error("Failed to save prompts to file")

        except Exception as e:
            gui_logger.error(f"Error creating prompts from data: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to create prompts from data: {str(e)}")

    def refresh_prompts_tab(self):
        """Refreshes the prompts tab to update data file status."""
        # Get the current tab index
        current_index = self.tabs.currentIndex()
        
        # Remove the old prompts tab
        self.tabs.removeTab(self.tabs.indexOf(self.tabs.widget(2)))
        
        # Add the new prompts tab
        self.tabs.insertTab(2, self.create_prompts_tab(), "Prompts")
        
        # Restore the current tab
        self.tabs.setCurrentIndex(current_index)

    def add_prompt_from_form(self):
        """Add a new prompt using form fields."""
        try:
            # Get form values
            prompt_text = self.prompt_edit.toPlainText().strip()
            system_prompt = self.system_prompt_edit.toPlainText().strip()
            tags = [t.strip() for t in self.tags_input.text().split(";") if t.strip()]
            prepend_text = self.prepend_edit.toPlainText().strip()

            if not prompt_text:
                QMessageBox.warning(self, "Error", "Prompt text cannot be empty")
                return

            # Find the next task number
            existing_prompts = self.prompt_manager.list_prompts()
            task_numbers = []
            for prompt in existing_prompts:
                prompt_tags = prompt.get("tags", [])
                for tag in prompt_tags:
                    if tag.startswith("Task"):
                        try:
                            task_num = int(tag[4:])  # Extract number after "Task"
                            task_numbers.append(task_num)
                        except ValueError:
                            continue
            next_task_num = max(task_numbers) + 1 if task_numbers else 1

            # Generate a unique prompt ID based on timestamp and task number
            timestamp = int(time.time())
            prompt_id = f"prompt_{timestamp}_task{next_task_num}"

            # Add the prompt with the task number as a tag
            self.prompt_manager.add_prompt(
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                system_prompt=system_prompt,
                tags=[f"Task{next_task_num}"] + tags,  # Add task number as first tag
                prepend_text=prepend_text
            )

            # Refresh the prompts table
            self.refresh_prompts_tab()

            # Clear the form
            self.prompt_edit.clear()
            self.system_prompt_edit.clear()
            self.tags_input.clear()
            self.prepend_edit.clear()

            QMessageBox.information(self, "Success", "Prompt added successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add prompt: {str(e)}")

    def delete_selected_prompts(self):
        """Delete the currently selected prompts after confirmation."""
        if not self.prompt_manager:
            return

        selected_rows = self.prompts_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select prompts to delete.")
            return

        # Get prompt IDs to delete
        prompt_ids = []
        for row in selected_rows:
            prompt_id = self.prompts_table.item(row.row(), 0).text()
            prompt_ids.append(prompt_id)

        # Show confirmation dialog
        if len(prompt_ids) == 1:
            message = f"Are you sure you want to delete prompt '{prompt_ids[0]}'?"
        else:
            message = f"Are you sure you want to delete {len(prompt_ids)} selected prompts?"

        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                for prompt_id in prompt_ids:
                    self.prompt_manager.delete_prompt(prompt_id)
                self.prompt_manager.export_prompts()  # Save changes
                self.load_prompts()  # Refresh the table
                QMessageBox.information(self, "Success", "Selected prompts deleted successfully.")
            except Exception as e:
                gui_logger.error(f"Failed to delete prompts: {e}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to delete prompts: {str(e)}")

    def clear_all_prompts(self):
        """Delete all prompts after confirmation."""
        if not self.prompt_manager:
            return

        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Confirm Clear All",
            "Are you sure you want to delete ALL prompts? This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.prompt_manager.clear_prompts()
                self.prompt_manager.export_prompts()  # Save changes
                self.load_prompts()  # Refresh the table
                QMessageBox.information(self, "Success", "All prompts deleted successfully.")
            except Exception as e:
                gui_logger.error(f"Failed to clear prompts: {e}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to clear prompts: {str(e)}")

    def export_prompts(self):
        """Export prompts to CSV file."""
        if not hasattr(self, 'prompt_manager'):
            QMessageBox.critical(self, "Error", "Prompt manager not initialized.")
            return

        try:
            # Get file path from user
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Prompts",
                str(Path(self.project_folder) / "prompts.csv"),
                "CSV Files (*.csv)"
            )

            if not file_path:
                return

            # Get all prompts
            prompts = self.prompt_manager.list_prompts()
            if not prompts:
                QMessageBox.warning(self, "No Prompts", "There are no prompts to export.")
                return

            # Convert prompts to DataFrame
            data = []
            for prompt in prompts:
                data.append({
                    'id': prompt['id'],
                    'system_prompt': prompt['system_prompt'],
                    'prompt_text': prompt['prompt_text']
                })
            
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            
            QMessageBox.information(self, "Success", f"Prompts exported successfully to:\n{file_path}")
            gui_logger.info(f"Prompts exported to CSV: {file_path}")

        except Exception as e:
            gui_logger.error(f"Failed to export prompts: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to export prompts: {str(e)}")

    def export_blinded_results(self, results: Dict) -> str:
        """Export results to blinded CSV file for analysis."""
        try:
            # Get file path from user
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Blinded Results",
                str(Path(self.project_folder) / "blinded_results.csv"),
                "CSV Files (*.csv)"
            )

            if not file_path:
                return ""

            # Create blinded mapping
            condition_mapping = {}
            for cond_id, cond_data in results["conditions"].items():
                condition_mapping[cond_id] = f"Condition_{len(condition_mapping) + 1}"

            # Create blinded CSV
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(['Prompt_ID', 'Response_ID', 'Condition', 'Response'])
                
                # Write data
                for prompt_result in results['prompt_results']:
                    for response in prompt_result['responses']:
                        writer.writerow([
                            prompt_result['prompt_id'],
                            response['response_id'],
                            condition_mapping[response['condition_id']],
                            response['text']
                        ])

            # Save mapping file
            mapping_file = str(Path(file_path).with_suffix('.mapping.json'))
            with open(mapping_file, 'w') as f:
                json.dump(condition_mapping, f, indent=2)

            QMessageBox.information(self, "Success", f"Blinded results exported successfully to:\n{file_path}")
            gui_logger.info(f"Blinded results exported to CSV: {file_path}")
            return file_path

        except Exception as e:
            gui_logger.error(f"Failed to export blinded results: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to export blinded results: {str(e)}")
            return ""

    def show_create_prompt_dialog(self):
        """Show dialog to create a new prompt."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Create New Prompt")
        dialog.setMinimumWidth(600)

        layout = QVBoxLayout()

        # Prompt text
        prompt_label = QLabel("Prompt Text:")
        prompt_edit = QTextEdit()
        prompt_edit.setMinimumHeight(100)
        layout.addWidget(prompt_label)
        layout.addWidget(prompt_edit)

        # System prompt
        system_prompt_label = QLabel("System Prompt (Optional):")
        system_prompt_edit = QTextEdit()
        system_prompt_edit.setMinimumHeight(50)
        layout.addWidget(system_prompt_label)
        layout.addWidget(system_prompt_edit)

        # Tags
        tags_layout = QHBoxLayout()
        tags_label = QLabel("Tags (Optional):")
        tags_label.setToolTip("Semicolon-separated list of tags")
        tags_input = QLineEdit()
        tags_layout.addWidget(tags_label)
        tags_layout.addWidget(tags_input)
        layout.addLayout(tags_layout)

        # Prepend text
        prepend_label = QLabel("Prepend Text (Optional):")
        prepend_edit = QTextEdit()
        prepend_edit.setMinimumHeight(50)
        layout.addWidget(prepend_label)
        layout.addWidget(prepend_edit)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            try:
                prompt_text = prompt_edit.toPlainText().strip()
                system_prompt = system_prompt_edit.toPlainText().strip()
                tags = [t.strip() for t in tags_input.text().split(";") if t.strip()]
                prepend_text = prepend_edit.toPlainText().strip()

                if not prompt_text:
                    QMessageBox.warning(self, "Error", "Prompt text cannot be empty")
                    return

                # Generate a unique prompt ID
                prompt_id = f"prompt_{int(time.time())}"

                # Add the prompt
                self.prompt_manager.add_prompt(
                    prompt_id=prompt_id,
                    prompt_text=prompt_text,
                    system_prompt=system_prompt,
                    tags=tags,
                    prepend_text=prepend_text
                )

                # Refresh the prompts table
                self.refresh_prompts_tab()

                QMessageBox.information(self, "Success", "Prompt created successfully")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to create prompt: {str(e)}")

    def create_results_tab(self):
        """Creates the Results viewing tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Splitter for file list and details view
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Left side: File List ---
        file_list_widget = QWidget()
        file_list_layout = QVBoxLayout(file_list_widget)
        file_list_layout.setContentsMargins(0,0,0,0)

        file_list_label = QLabel("Result Files:")
        self.results_file_list = QListWidget()
        self.results_file_list.currentItemChanged.connect(self.display_selected_result_file)

        refresh_files_button = QPushButton("Refresh File List")
        refresh_files_button.setIcon(QIcon.fromTheme("view-refresh"))
        refresh_files_button.clicked.connect(self.refresh_result_file_list)

        file_list_layout.addWidget(file_list_label)
        file_list_layout.addWidget(self.results_file_list)
        file_list_layout.addWidget(refresh_files_button)

        # --- Right side: Details View ---
        details_widget = QWidget()
        details_layout = QVBoxLayout(details_widget)
        details_layout.setContentsMargins(0,0,0,0)

        # Experiment Info Area
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

        # Results Table
        self.results_detail_table = QTableWidget()
        self.results_detail_table.setColumnCount(8)  # Increased from 7 to 8 for new column
        self.results_detail_table.setHorizontalHeaderLabels([
            "Prompt ID", "Experiment", "Condition", "Provider", "Model", "Response", "Duration (s)", "Error"
        ])
        # Set stretch for response column
        self.results_detail_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.results_detail_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_detail_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_detail_table.verticalHeader().setVisible(False)
        self.results_detail_table.setWordWrap(True)
        self.results_detail_table.resizeRowsToContents()
        details_layout.addWidget(self.results_detail_table)

        # Export Button
        export_button = QPushButton("Export Selected Result to CSV")
        export_button.setIcon(QIcon.fromTheme("document-save-as"))
        export_button.clicked.connect(self.export_selected_result)
        details_layout.addWidget(export_button, 0, Qt.AlignmentFlag.AlignRight)

        # Add widgets to splitter
        splitter.addWidget(file_list_widget)
        splitter.addWidget(details_widget)
        splitter.setSizes([300, 900])

        layout.addWidget(splitter)

        # Load initial results file list
        self.refresh_result_file_list()

        return tab

    def refresh_result_file_list(self):
        """Refreshes the list of JSON result files."""
        if not self.experiment_manager:
            return

        gui_logger.info("Refreshing result file list.")
        self.results_file_list.clear()
        self.results_detail_table.setRowCount(0)
        self.result_exp_name_label.setText("Experiment: N/A")
        self.result_exp_id_label.setText("ID: N/A")
        self.result_timestamp_label.setText("Timestamp: N/A")

        try:
            # Get the selected experiment ID from the experiments table
            selected_row = self.experiments_table.currentRow()
            if selected_row < 0:
                self.results_file_list.addItem("No experiment selected.")
                self.results_file_list.setEnabled(False)
                return

            experiment_id = self.experiments_table.item(selected_row, 0).text()
            result_files_paths = sorted(
                self.experiment_manager.results_folder.glob(f"results_{experiment_id}_*.json"),
                key=os.path.getmtime,
                reverse=True
            )
            if not result_files_paths:
                self.results_file_list.addItem("No result files found for this experiment.")
                self.results_file_list.setEnabled(False)
            else:
                self.results_file_list.setEnabled(True)
                for path in result_files_paths:
                    mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M:%S')
                    item = QListWidgetItem(f"{path.name} ({mod_time})")
                    item.setData(Qt.ItemDataRole.UserRole, str(path))
                    self.results_file_list.addItem(item)
                gui_logger.info(f"Found {len(result_files_paths)} result files for experiment {experiment_id}.")

        except Exception as e:
            gui_logger.error(f"Error listing result files: {e}", exc_info=True)
            self.results_file_list.addItem("Error listing files.")
            self.results_file_list.setEnabled(False)
            QMessageBox.critical(self, "File Error", f"Could not list result files.\nError: {e}")

    def display_selected_result_file(self, current_item: QListWidgetItem, previous_item: QListWidgetItem = None):
        """Loads and displays the details of the selected result JSON file."""
        if not current_item:
            self.results_detail_table.setRowCount(0)
            return

        file_path_str = current_item.data(Qt.ItemDataRole.UserRole)
        if not file_path_str:
            return

        file_path = Path(file_path_str)
        gui_logger.info(f"Displaying results from: {file_path}")

        try:
            with open(file_path, "r", encoding='utf-8') as f:
                results_data = json.load(f)

            # Update header info
            experiment_name = results_data.get('experiment_name', 'N/A')
            self.result_exp_name_label.setText(f"Experiment: {experiment_name}")
            self.result_exp_id_label.setText(f"ID: {results_data.get('experiment_id', 'N/A')}")
            ts = results_data.get('timestamp', 'N/A')
            try:
                ts_dt = datetime.datetime.fromisoformat(ts.replace('Z', '+00:00'))
                ts_display = ts_dt.strftime('%Y-%m-%d %H:%M:%S %Z')
            except:
                ts_display = ts
            self.result_timestamp_label.setText(f"Timestamp: {ts_display}")

            # Set up table columns
            self.results_detail_table.setColumnCount(8)
            self.results_detail_table.setHorizontalHeaderLabels([
                "Prompt ID", "Experiment", "Condition", "Provider", "Model", "Response", "Duration (s)", "Error"
            ])
            self.results_detail_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)

            # Populate results table
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
                        "duration": f"{resp_data.get('duration_seconds', 'N/A'):.2f}" if resp_data.get('duration_seconds') is not None else "N/A",
                        "error": resp_data.get("error", "")
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
        
        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Export to CSV",
            "Would you like to export this result to CSV format?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Load the JSON file
                with open(file_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                # Get the experiment manager
                experiment_manager = self.experiment_manager
                
                # Export to CSV
                csv_path = experiment_manager.export_results_csv(results)
                
                if csv_path:
                    QMessageBox.information(
                        self,
                        "Export Successful",
                        f"Results exported to CSV:\n{csv_path}"
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Export Failed",
                        "Failed to export results to CSV."
                    )
                    
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Error exporting results: {str(e)}"
                )

    def load_prompts(self):
        """Populate the Prompts table from prompt_manager."""
        if not self.prompt_manager:
            return

        prompts = self.prompt_manager.list_prompts()
        self.prompts_table.setRowCount(len(prompts))

        for row, p in enumerate(prompts):
            # ID
            prompt_id = p.get("id", "")
            self.prompts_table.setItem(row, 0, QTableWidgetItem(prompt_id))
            
            # Data ID - Get from tags
            data_id = ""
            if "tags" in p:
                for tag in p["tags"]:
                    if tag.startswith("data_id_"):
                        data_id = tag.replace("data_id_", "")
                        break
            
            self.prompts_table.setItem(row, 1, QTableWidgetItem(data_id))
            
            # System Prompt
            self.prompts_table.setItem(row, 2, QTableWidgetItem(p.get("system_prompt", "")))
            # Prepend Text
            self.prompts_table.setItem(row, 3, QTableWidgetItem(p.get("prepend_text", "")))
            # Prompt Text
            self.prompts_table.setItem(row, 4, QTableWidgetItem(p.get("prompt_text", "")))
            # Tags
            tags = ";".join(p.get("tags", []))
            self.prompts_table.setItem(row, 5, QTableWidgetItem(tags))

    def import_data_file(self):
        """Import a data file and set up for prompt generation."""
        if not self.prompt_manager:
            return

        file_name, _ = QFileDialog.getOpenFileName(
            self, "Import Data File", self.project_folder, "CSV Files (*.csv)"
        )
        if file_name:
            try:
                # Use the data manager to import the file
                self.data_manager.import_csv(file_name)
                
                # Refresh the prompts tab to update the data file status
                self.refresh_prompts_tab()
                
                QMessageBox.information(self, "Success", 
                    f"Data file imported successfully!\n"
                    f"Text column: {self.data_manager.file_info['text_column']}\n"
                    f"Total rows: {self.data_manager.file_info['total_rows']}\n\n"
                    f"Next steps:\n"
                    f"1. Set the system prompt\n"
                    f"2. Add prepend-prompt pairs\n"
                    f"3. Generate prompts")

            except Exception as e:
                gui_logger.error(f"Failed to import data file: {e}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to import data file: {str(e)}")

    def delete_selected_experiments(self):
        """Delete the currently selected experiments after confirmation."""
        if not self.experiment_manager:
            QMessageBox.critical(self, "Error", "Experiment manager not initialized.")
            return

        selected_rows = self.experiments_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select experiments to delete.")
            return

        # Get experiment IDs to delete
        experiment_ids = []
        for row in selected_rows:
            experiment_id = self.experiments_table.item(row.row(), 0).text()
            experiment_ids.append(experiment_id)

        # Show confirmation dialog
        if len(experiment_ids) == 1:
            message = f"Are you sure you want to delete experiment '{experiment_ids[0]}'?"
        else:
            message = f"Are you sure you want to delete {len(experiment_ids)} selected experiments?"

        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                for experiment_id in experiment_ids:
                    self.experiment_manager.delete_experiment(experiment_id)
                self.load_experiments()  # Refresh the table
                QMessageBox.information(self, "Success", "Selected experiments deleted successfully.")
                gui_logger.info(f"Deleted experiments: {experiment_ids}")
            except Exception as e:
                gui_logger.error(f"Failed to delete experiments: {e}", exc_info=True)
                QMessageBox.critical(self, "Error", f"Failed to delete experiments: {str(e)}")

    def test_api_key(self, provider_lower):
        """Test the API key for a given provider and update the status icon."""
        key_input = self.api_key_inputs[provider_lower]
        status_icon = self.api_key_status_icons[provider_lower]
        api_key = key_input.text().strip() or self.api_key_manager.get_key(provider_lower)
        if not api_key:
            status_icon.setText("⏺️")
            status_icon.setStyleSheet("color: gray;")
            status_icon.setToolTip("No key entered")
            return
        # Show loading indicator
        status_icon.setText("⏳")
        status_icon.setStyleSheet("")
        status_icon.setToolTip("Testing...")
        QApplication.processEvents()
        # Test logic for each provider
        from qually_tool import ProviderFactory
        try:
            provider_instance = ProviderFactory.create_provider(provider_lower, api_key)
            if not provider_instance:
                raise Exception("Could not create provider instance")
            models = provider_instance.get_available_models()
            # Default model lists for fallback detection
            default_models_map = {
                "openai": sorted(["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]),
                "google": sorted(["gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-1.0-pro"]),
                "mistral": sorted([
                    "mistral-large-latest", "mistral-large-2402",
                    "mistral-medium-latest", "mistral-medium",
                    "mistral-small-latest", "mistral-small",
                    "mistral-tiny", "open-mistral-7b", "open-mixtral-8x7b"
                ]),
                "grok": sorted(["grok-1", "grok-1.5-flash", "grok-1.5"]),
                "deepseek": sorted(["deepseek-chat", "deepseek-coder"]),
                # Anthropic always returns static list, so treat as always valid
            }
            if provider_lower == "anthropic":
                status_icon.setText("✔️")
                status_icon.setStyleSheet("color: green;")
                status_icon.setToolTip("Key is valid!")
                return
            default_models = default_models_map.get(provider_lower)
            if default_models and sorted(models) == default_models:
                status_icon.setText("❌")
                status_icon.setStyleSheet("color: red;")
                status_icon.setToolTip("Invalid key: using fallback models")
            elif models and isinstance(models, list):
                status_icon.setText("✔️")
                status_icon.setStyleSheet("color: green;")
                status_icon.setToolTip("Key is valid!")
            else:
                status_icon.setText("❌")
                status_icon.setStyleSheet("color: red;")
                status_icon.setToolTip("Key test failed: No models returned")
        except Exception as e:
            status_icon.setText("❌")
            status_icon.setStyleSheet("color: red;")
            status_icon.setToolTip(f"Invalid key: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Load and apply modern theme
    theme_path = Path(__file__).parent / "modern_theme.qss"
    if theme_path.exists():
        with open(theme_path, "r") as f:
            app.setStyleSheet(f.read())
    
    # Add global QLineEdit style for uniformity
    app.setStyleSheet(app.styleSheet() + "\nQLineEdit { height: 32px; font-size: 15px; padding: 4px 8px; }")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
