#!/usr/bin/env python3
"""
macOS build script for Qually using py2app.
This script is only for creating a macOS .app bundle.
"""

import os
import sys
from setuptools import setup

# Path to your main GUI entry point
APP = ['qually_gui.py']

# Any additional non-Python files to include
DATA_FILES = [
    ('', [  # Put resources in the root Resources directory
        'resources/icon.png',
        'resources/app_icon.icns',
        'qually_theme.qss'
    ]),
    ('fonts', [  # Include fonts directory
        'resources/fonts/Inter_18pt-Regular.ttf',
        'resources/fonts/Inter_24pt-Regular.ttf',
        'resources/fonts/Inter_28pt-Regular.ttf'
    ])
]

# py2app specific build options
OPTIONS = {
    'argv_emulation': False,
    'packages': ['PyQt6', 'pandas', 'numpy', 'cryptography', 'requests'],
    'includes': ['PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets', 'PyQt6.sip'],
    'excludes': ['tkinter', 'PySide6', 'PyInstaller'],
    'iconfile': 'resources/app_icon.icns',
    'resources': ['resources'],
    'plist': {
        'CFBundleName': 'Qually LE',
        'CFBundleDisplayName': 'Qually LE',
        'CFBundleIdentifier': 'in.kartiktrivedi.qually-le',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'LSMinimumSystemVersion': '10.13',
        'NSHighResolutionCapable': True
    }
}

setup(
    name="Qually LE",
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
) 