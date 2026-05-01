import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.widgets import SectionHeader, StatusBadge

gui_logger = logging.getLogger("QuallyGUI")

_PROVIDERS = ["OpenAI", "Anthropic", "Google", "Mistral", "Grok", "DeepSeek"]


class ApiKeysPage(QWidget):
    keys_saved = pyqtSignal()

    def __init__(self, api_key_manager, parent=None):
        super().__init__(parent)
        self.api_key_manager = api_key_manager
        self.api_key_inputs: dict[str, QLineEdit] = {}
        self.api_key_badges: dict[str, StatusBadge] = {}
        self.api_key_test_buttons: dict[str, QPushButton] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        layout.addWidget(SectionHeader(
            "API Keys",
            "Add keys for the LLM providers you want to use. "
            "Keys are stored encrypted inside your project folder.",
        ))

        # Column header row
        header_row = QHBoxLayout()
        for text, width in [("Provider", 120), ("Status", 116), ("API Key", 0), ("", 64)]:
            lbl = QLabel(text)
            lbl.setObjectName("tableColumnHeader")
            if width:
                lbl.setFixedWidth(width)
            else:
                lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            header_row.addWidget(lbl)
        layout.addLayout(header_row)

        # One row per provider
        for provider in _PROVIDERS:
            p = provider.lower()
            row = QHBoxLayout()
            row.setSpacing(8)

            name_lbl = QLabel(provider)
            name_lbl.setFixedWidth(120)
            row.addWidget(name_lbl)

            badge = StatusBadge("unknown")
            row.addWidget(badge)
            self.api_key_badges[p] = badge

            key_input = QLineEdit()
            key_input.setEchoMode(QLineEdit.EchoMode.Password)
            key_input.setPlaceholderText(f"Enter {provider} API key")
            key_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            row.addWidget(key_input, 1)
            self.api_key_inputs[p] = key_input

            test_btn = QPushButton("Test")
            test_btn.setFixedWidth(60)
            test_btn.setToolTip(f"Test {provider} API key")
            row.addWidget(test_btn)
            self.api_key_test_buttons[p] = test_btn

            layout.addLayout(row)

        layout.addStretch()

        save_btn = QPushButton("Save All API Keys")
        save_btn.setObjectName("primaryButton")
        save_btn.clicked.connect(self.save_api_keys)
        layout.addWidget(save_btn, 0, Qt.AlignmentFlag.AlignRight)

        self.load_api_keys_to_ui()
        for p, btn in self.api_key_test_buttons.items():
            btn.clicked.connect(lambda _, prov=p: self.test_api_key(prov))

    # ------------------------------------------------------------------
    # Key management
    # ------------------------------------------------------------------

    def load_api_keys_to_ui(self):
        if not self.api_key_manager:
            return
        for p, inp in self.api_key_inputs.items():
            if self.api_key_manager.get_key(p):
                inp.setPlaceholderText("Key is set — enter a new key to replace")
                self.api_key_badges[p].set_state("unknown")
            else:
                inp.setPlaceholderText(f"Enter {p.capitalize()} API key")
                self.api_key_badges[p].set_state("not_configured")
            inp.clear()

    def save_api_keys(self):
        if not self.api_key_manager:
            QMessageBox.critical(self, "Error", "API Key Manager not initialized.")
            return

        saved = 0
        try:
            for p, inp in self.api_key_inputs.items():
                key_text = inp.text().strip()
                if key_text:
                    self.api_key_manager.save_key(p, key_text)
                    saved += 1
                    gui_logger.info(f"Saved API key for: {p}")

            if saved > 0:
                QMessageBox.information(self, "Saved", f"{saved} API key(s) saved successfully.")
                self.keys_saved.emit()
            else:
                QMessageBox.information(self, "No Changes", "No new API keys were entered.")

            self.load_api_keys_to_ui()

        except Exception as e:
            gui_logger.error(f"Failed to save API keys: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to save API keys:\n{str(e)}")

    def test_api_key(self, p: str):
        badge = self.api_key_badges[p]
        inp = self.api_key_inputs[p]
        api_key = inp.text().strip() or self.api_key_manager.get_key(p)

        if not api_key:
            badge.set_state("not_configured")
            return

        badge.set_state("testing")
        QApplication.processEvents()

        from qually_tool import ProviderFactory

        try:
            provider_instance = ProviderFactory.create_provider(p, api_key)
            if not provider_instance:
                raise Exception("Could not create provider instance")

            models = provider_instance.get_available_models()

            # Anthropic doesn't expose a model-list endpoint — a successful
            # instantiation is sufficient to confirm the key is valid.
            if p == "anthropic":
                badge.set_state("connected")
                return

            default_models_map = {
                "openai":    sorted(["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]),
                "google":    sorted(["gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-1.0-pro"]),
                "mistral":   sorted(["mistral-large-latest", "mistral-large-2402", "mistral-medium-latest",
                                     "mistral-medium", "mistral-small-latest", "mistral-small",
                                     "mistral-tiny", "open-mistral-7b", "open-mixtral-8x7b"]),
                "grok":      sorted(["grok-1", "grok-1.5-flash", "grok-1.5"]),
                "deepseek":  sorted(["deepseek-chat", "deepseek-coder"]),
            }
            fallback = default_models_map.get(p)
            if fallback and sorted(models) == fallback:
                badge.set_state("invalid")
                badge.setToolTip("Invalid key: API returned fallback model list")
            elif models and isinstance(models, list):
                badge.set_state("connected")
            else:
                badge.set_state("invalid")
                badge.setToolTip("Key test failed: no models returned")

        except Exception as e:
            badge.set_state("invalid")
            badge.setToolTip(f"Invalid key: {str(e)}")
