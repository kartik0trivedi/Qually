import logging
import os
import sys
from pathlib import Path

from PyQt6.QtGui import QFont, QFontDatabase


gui_logger = logging.getLogger("QuallyGUI")


def resolve_application_path() -> Path:
    """Return the app root in development and packaged builds."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent.parent
    return Path(__file__).resolve().parent.parent


def resolve_resources_path() -> Path:
    """Return the resources directory for development, py2app, and PyInstaller builds."""
    application_path = resolve_application_path()

    if getattr(sys, "frozen", False):
        resources_path = application_path / "Resources"
        gui_logger.info(f"Running as bundled app from: {application_path}")
        gui_logger.info(f"Resources path: {resources_path}")

        if not resources_path.exists():
            alternate_paths = [
                application_path.parent / "Resources",
                application_path / "Contents" / "Resources",
                Path(sys.executable).parent / "resources",
            ]
            for alt_path in alternate_paths:
                if alt_path.exists():
                    resources_path = alt_path
                    gui_logger.info(f"Found resources in alternate path: {resources_path}")
                    break
            else:
                gui_logger.warning("Could not find resources directory in any expected location")
        return resources_path

    resources_path = application_path / "resources"
    os.chdir(application_path)
    gui_logger.info(f"Running as script from: {application_path}")
    gui_logger.info(f"Resources path: {resources_path}")
    return resources_path


resources_path = resolve_resources_path()
icon_path = resources_path / "icon.png"
gui_logger.info(f"Icon path: {icon_path} (exists: {icon_path.exists()})")


class FontManager:
    def __init__(self, resources_path_: Path):
        self.resources_path = resources_path_
        self.fonts_path = resources_path_ / "fonts"
        self.font_ids = {}
        self.load_fonts()

    def load_fonts(self):
        """Load all font files from the fonts directory."""
        font_files = {
            "Inter-Regular-18": "Inter_18pt-Regular.ttf",
            "Inter-Regular-24": "Inter_24pt-Regular.ttf",
            "Inter-Regular-28": "Inter_28pt-Regular.ttf",
        }

        for font_name, font_file in font_files.items():
            font_path = self.fonts_path / font_file
            if font_path.exists():
                font_id = QFontDatabase.addApplicationFont(str(font_path))
                if font_id != -1:
                    self.font_ids[font_name] = font_id
                    gui_logger.info(f"Successfully loaded font: {font_name}")
                else:
                    gui_logger.error(f"Failed to load font: {font_name}")
            else:
                gui_logger.error(f"Font file not found: {font_path}")

    def get_font(self, font_name, size=None):
        """Get a QFont instance for the specified font name and size."""
        if font_name in self.font_ids:
            font = QFont(QFontDatabase.applicationFontFamilies(self.font_ids[font_name])[0])
            if size:
                font.setPointSize(size)
            return font
        return QFont()


def load_app_stylesheet(resources_path_: Path) -> str:
    """Load the base and experiment-specific QSS files."""
    theme_parts = []
    base_theme_path = resources_path_ / "modern_theme.qss"
    experiment_theme_path = resources_path_ / "experiment_tab_additions.qss"

    if base_theme_path.exists():
        theme_parts.append(base_theme_path.read_text())
    else:
        gui_logger.warning(f"Base theme file not found: {base_theme_path}")

    if experiment_theme_path.exists():
        theme_parts.append(experiment_theme_path.read_text())
    else:
        gui_logger.warning(f"Experiment theme file not found: {experiment_theme_path}")

    stylesheet = "\n".join(theme_parts)

    # Replace relative icon paths with absolute paths so Qt resolves them
    # regardless of working directory (important in bundled .app builds).
    icons_dir = str(resources_path_ / "icons").replace("\\", "/")
    stylesheet = stylesheet.replace("url(resources/icons/", f"url({icons_dir}/")

    return stylesheet
