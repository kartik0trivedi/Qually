#!/usr/bin/env python3
"""
macOS build script for Qually using py2app.
This script is only for creating a macOS .app bundle.

Requires Python 3.11+ and py2app.
"""

import os
import sys
from setuptools import setup

# Path to your main GUI entry point
APP = ['qually_gui.py']

# Any additional non-Python files to include.
# Paths are relative to your project root (Qually_LE2/).
DATA_FILES = [
    # Files to be placed in YourApp.app/Contents/Resources/
    ('', [
        'resources/icon.png',                     # Source: Qually_LE2/resources/icon.png
        'resources/app_icon.icns',                # Source: Qually_LE2/resources/app_icon.icns
        'resources/qually_theme.qss',             # Source: Qually_LE2/resources/qually_theme.qss
        'resources/modern_theme.qss',             # Source: Qually_LE2/resources/modern_theme.qss
        'resources/experiment_tab_additions.qss'  # Source: Qually_LE2/resources/experiment_tab_additions.qss
    ]),
    # Fonts to be placed in YourApp.app/Contents/Resources/fonts/
    ('fonts', [
        'resources/fonts/Inter_18pt-Regular.ttf',
        'resources/fonts/Inter_24pt-Regular.ttf',
        'resources/fonts/Inter_28pt-Regular.ttf'
    ])
]

# py2app specific build options
OPTIONS = {
    'argv_emulation': False,
    'packages': ['PyQt6', 'pandas', 'numpy', 'cryptography', 'requests'],
    'includes': [
        'PyQt6.QtCore', 
        'PyQt6.QtGui', 
        'PyQt6.QtWidgets', 
        'PyQt6.sip',
        'cmath',
        'numpy.core._methods',
        'numpy.lib.format',
        'pandas._libs.tslibs.base',
        'pandas._libs.tslibs.np_datetime',
        'pandas._libs.tslibs.offsets',
        'pandas._libs.tslibs.period',
        'pandas._libs.tslibs.strptime',
        'pandas._libs.tslibs.timedeltas',
        'pandas._libs.tslibs.timestamps',
        'pandas._libs.tslibs.timezones',
        'pandas._libs.tslibs.tzconversion'
    ],
    'excludes': ['tkinter', 'PySide6', 'PyInstaller'],
    'iconfile': 'resources/app_icon.icns', # Sets the .app bundle icon. Source: Qually_LE2/resources/app_icon.icns
    'plist': {
        'CFBundleName': 'Qually LE',
        'CFBundleDisplayName': 'Qually LE',
        'CFBundleIdentifier': 'in.kartiktrivedi.qually-le',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'LSMinimumSystemVersion': '10.13', # macOS High Sierra or later
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