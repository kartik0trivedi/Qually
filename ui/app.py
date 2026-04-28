import sys

from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow
from ui.theme import FontManager, load_app_stylesheet, resources_path


def run():
    app = QApplication(sys.argv)

    font_manager = FontManager(resources_path)
    app.setFont(font_manager.get_font("Inter-Regular-18", 12))
    app.setStyleSheet(load_app_stylesheet(resources_path))

    window = MainWindow()
    window.show()
    return sys.exit(app.exec())
