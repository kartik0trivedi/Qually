#!/usr/bin/env python3
"""
Environment setup script for Qually LE.
This script creates a virtual environment and installs all required dependencies.
Works on both Windows and macOS.
"""

import os
import platform
import subprocess
import sys
import venv
from pathlib import Path

# Required packages and their versions
REQUIRED_PACKAGES = [
    'PyQt6>=6.4.0',
    'pandas>=2.0.0',
    'numpy>=1.24.0',
    'cryptography>=41.0.0',
    'requests>=2.31.0',
    'pyinstaller>=6.0.0',  # For building executables
]

def create_venv():
    """Create a virtual environment."""
    venv_path = Path('venv')
    if venv_path.exists():
        print("Virtual environment already exists.")
        return
    
    print("Creating virtual environment...")
    venv.create(venv_path, with_pip=True)
    print("Virtual environment created successfully.")

def get_python_executable():
    """Get the path to the Python executable in the virtual environment."""
    if platform.system() == 'Windows':
        return Path('venv/Scripts/python.exe')
    return Path('venv/bin/python')

def get_pip_executable():
    """Get the path to the pip executable in the virtual environment."""
    if platform.system() == 'Windows':
        return Path('venv/Scripts/pip.exe')
    return Path('venv/bin/pip')

def upgrade_pip():
    """Upgrade pip to the latest version."""
    print("Upgrading pip...")
    pip_path = get_pip_executable()
    subprocess.run([str(pip_path), 'install', '--upgrade', 'pip'], check=True)
    print("Pip upgraded successfully.")

def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    pip_path = get_pip_executable()
    
    for package in REQUIRED_PACKAGES:
        print(f"Installing {package}...")
        subprocess.run([str(pip_path), 'install', package], check=True)
    
    print("All packages installed successfully.")

def verify_installation():
    """Verify that all required packages are installed correctly."""
    print("\nVerifying installation...")
    python_path = get_python_executable()
    
    # Create a verification script
    verify_script = """
import sys
import importlib

required_packages = [
    'PyQt6',
    'pandas',
    'numpy',
    'cryptography',
    'requests',
    'pyinstaller'
]

missing_packages = []
for package in required_packages:
    try:
        importlib.import_module(package)
        print(f"✓ {package} is installed")
    except ImportError:
        missing_packages.append(package)
        print(f"✗ {package} is NOT installed")

if missing_packages:
    print("\\nSome packages are missing. Please run the setup script again.")
    sys.exit(1)
else:
    print("\\nAll required packages are installed correctly!")
"""
    
    # Write verification script to a temporary file
    verify_script_path = Path('verify_install.py')
    verify_script_path.write_text(verify_script)
    
    try:
        subprocess.run([str(python_path), str(verify_script_path)], check=True)
    finally:
        # Clean up the temporary script
        verify_script_path.unlink()

def main():
    """Main function to set up the environment."""
    print("Setting up Qually LE development environment...")
    
    # Create virtual environment
    create_venv()
    
    # Upgrade pip
    upgrade_pip()
    
    # Install requirements
    install_requirements()
    
    # Verify installation
    verify_installation()
    
    print("\nEnvironment setup completed successfully!")
    print("\nTo activate the virtual environment:")
    if platform.system() == 'Windows':
        print("    .\\venv\\Scripts\\activate")
    else:
        print("    source venv/bin/activate")

if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nError during setup: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1) 