import json
from pathlib import Path


class SettingsManager:
    def __init__(self, project_folder: Path):
        self.settings_path = Path(project_folder) / "settings.json"

    def load_settings(self):
        try:
            with open(self.settings_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_settings(self, settings):
        with open(self.settings_path, "w") as f:
            json.dump(settings, f, indent=4)
