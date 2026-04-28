import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


gui_logger = logging.getLogger("QuallyGUI")


class ApiKeysPage(QWidget):
    keys_saved = pyqtSignal()

    def __init__(self, api_key_manager, parent=None):
        super().__init__(parent)
        self.api_key_manager = api_key_manager
        self.api_key_inputs = {}
        self.api_key_status_icons = {}
        self.api_key_test_buttons = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

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
        help_text.setStyleSheet(
            """
            QLabel {
                padding: 15px;
                background-color: #e9ecef;
                border: 1px solid #ced4da;
                border-radius: 5px;
                margin-bottom: 20px;
                font-size: 13px;
            }
        """
        )
        layout.addWidget(help_text)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        api_keys_group = QWidget()
        api_keys_layout = QVBoxLayout(api_keys_group)
        api_keys_layout.setSpacing(10)

        providers = ["OpenAI", "Anthropic", "Google", "Mistral", "Grok", "DeepSeek"]
        for provider in providers:
            key_layout = QHBoxLayout()
            key_label = QLabel(f"{provider} API Key:")
            key_label.setFixedWidth(120)
            key_input = QLineEdit()
            key_input.setEchoMode(QLineEdit.EchoMode.Password)
            key_input.setPlaceholderText(f"Enter {provider} key here")
            key_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

            status_icon = QLabel()
            status_icon.setFixedWidth(24)
            status_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
            status_icon.setText("⏺️")
            status_icon.setToolTip("Not tested")

            test_button = QPushButton("Test")
            test_button.setFixedWidth(50)
            test_button.setToolTip(f"Test {provider} API key")

            key_layout.addWidget(key_label)
            key_layout.addWidget(key_input, 1)
            key_layout.addWidget(status_icon, 0)
            key_layout.addWidget(test_button, 0)
            api_keys_layout.addLayout(key_layout)

            provider_lower = provider.lower()
            self.api_key_inputs[provider_lower] = key_input
            self.api_key_status_icons[provider_lower] = status_icon
            self.api_key_test_buttons[provider_lower] = test_button

        scroll_area.setWidget(api_keys_group)
        layout.addWidget(scroll_area)

        save_button = QPushButton("Save All API Keys")
        save_button.setIcon(QIcon.fromTheme("document-save"))
        save_button.clicked.connect(self.save_api_keys)
        layout.addWidget(save_button, 0, Qt.AlignmentFlag.AlignRight)

        layout.addStretch()

        self.load_api_keys_to_ui()
        for provider_lower, test_button in self.api_key_test_buttons.items():
            test_button.clicked.connect(lambda checked, p=provider_lower: self.test_api_key(p))

    def load_api_keys_to_ui(self):
        """Loads existing keys as placeholder indicators."""
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
                self.keys_saved.emit()
            else:
                QMessageBox.information(self, "No Changes", "No new API keys were entered to save.")

            self.load_api_keys_to_ui()

        except Exception as e:
            gui_logger.error(f"Failed to save API keys: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to save API keys: {str(e)}")

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

        status_icon.setText("⏳")
        status_icon.setStyleSheet("")
        status_icon.setToolTip("Testing...")
        QApplication.processEvents()

        from qually_tool import ProviderFactory

        try:
            provider_instance = ProviderFactory.create_provider(provider_lower, api_key)
            if not provider_instance:
                raise Exception("Could not create provider instance")
            models = provider_instance.get_available_models()
            default_models_map = {
                "openai": sorted(["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]),
                "google": sorted(["gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-1.0-pro"]),
                "mistral": sorted(
                    [
                        "mistral-large-latest",
                        "mistral-large-2402",
                        "mistral-medium-latest",
                        "mistral-medium",
                        "mistral-small-latest",
                        "mistral-small",
                        "mistral-tiny",
                        "open-mistral-7b",
                        "open-mixtral-8x7b",
                    ]
                ),
                "grok": sorted(["grok-1", "grok-1.5-flash", "grok-1.5"]),
                "deepseek": sorted(["deepseek-chat", "deepseek-coder"]),
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
