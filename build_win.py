#!/usr/bin/env python3
"""
Windows build script for Qually LE using PyInstaller.
"""

import PyInstaller.__main__
import os
import shutil
from pathlib import Path

# --- Configuration ---
APP_NAME = "Qually LE"
SCRIPT_FILE = "qually_gui.py"
# IMPORTANT: PyInstaller on Windows expects an .ico file for the application icon.
# You may need to convert your 'icon.png' or 'app_icon.icns' to an .ico file.
# If 'resources/app_icon.ico' doesn't exist, PyInstaller will use a default icon.
# Alternatively, PyInstaller can sometimes use a .png directly, but .ico is preferred.
ICON_FILE = "resources/icon.png" # Using png, recommend converting to .ico for best results

# --- Main ---
if __name__ == '__main__':
    # Clean up previous builds
    if Path('dist').exists():
        shutil.rmtree('dist')
    if Path('build').exists():
        shutil.rmtree('build')
    spec_file = Path(f'{APP_NAME}.spec')
    if spec_file.exists():
        spec_file.unlink()

    # PyInstaller arguments
    # Note: The format for --add-data is 'source{os.pathsep}destination_in_bundle'
    # 'source' is the path to the file/folder on your disk.
    # 'destination_in_bundle' is the path relative to the bundle's root directory.
    # '.' for destination means the root of the bundle.
    pyinstaller_args = [
        '--name', APP_NAME,
        '--windowed',  # Use '--console' or remove '--windowed' for debugging
        # '--onefile',  # Optional: uncomment to create a single executable file (can be slower to start)
        '--icon', ICON_FILE,

        # Data files
        # This places 'resources/icon.png' into a 'resources' folder in the bundle
        '--add-data', f'resources{os.sep}icon.png{os.pathsep}resources',
        # This places 'modern_theme.qss' into the root of the bundle (alongside the executable)
        '--add-data', f'modern_theme.qss{os.pathsep}.',
        # This places the 'resources/fonts' directory into 'resources/fonts' in the bundle
        '--add-data', f'resources{os.sep}fonts{os.pathsep}resources{os.sep}fonts',
        # This places 'resources/experiment_tab_additions.qss' into a 'resources' folder in the bundle
        '--add-data', f'resources{os.sep}experiment_tab_additions.qss{os.pathsep}resources',

        # Hidden imports for modules PyInstaller might miss
        '--hidden-import', 'PyQt6.sip',
        '--hidden-import', 'PyQt6.QtNetwork', # For QNetworkAccessManager if used indirectly
        # Common hidden imports for pandas (add more if build fails on pandas issues)
        '--hidden-import', 'pandas._libs.tslibs.np_datetime',
        '--hidden-import', 'pandas._libs.tslibs.nattype',
        '--hidden-import', 'pandas._libs.skiplist',
        '--hidden-import', 'cryptography.hazmat.backends.openssl', # Often needed for cryptography

        # Exclude modules not needed to potentially reduce bundle size
        '--exclude-module', 'tkinter',
        '--exclude-module', 'PySide6', # Assuming you are using PyQt6 exclusively
        '--exclude-module', 'matplotlib', # If not used
        '--exclude-module', 'scipy',      # If not used

        SCRIPT_FILE,
    ]

    print("Running PyInstaller with args:")
    print(" ".join(pyinstaller_args))

    # Run PyInstaller
    PyInstaller.__main__.run(pyinstaller_args)

    print(f"\nBuild complete. Executable and associated files are in the 'dist/{APP_NAME}' folder.")
    if '--onefile' in pyinstaller_args:
        print(f"The single executable is 'dist/{APP_NAME}.exe'.")
    else:
        print(f"The main executable is 'dist/{APP_NAME}/{APP_NAME}.exe'.")