# Qually
## Overview
Qually Leading Edge is an LLM Audit Tool designed specifically for Social Science Research. This platform provides researchers with powerful capabilities to evaluate and audit Language Learning Models in the context of social science research.

## Features
- LLM evaluation and auditing tools
- Research-focused interface
- Data analysis capabilities
- Results export and logging
- MacOS native application

## Prerequisites
- Python 3.11 or higher
- MacOS with Cocoa environment
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Qually_LE.git
cd Qually_LE
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages include:
- PyQt6 (≥6.5.0)
- pandas (≥2.1.0)
- requests (≥2.28.2)
- cryptography (≥40.0.2)
- Pillow (≥10.0.0)

## Running the Application

### Development Mode
```bash
python qually_gui.py
```

### Building the Application
The application can be built into a native MacOS application using py2app:
```bash
python setup.py py2app
```

## Project Structure
```
Qually_LE/
├── qually_gui.py    # Main entry point
├── settings.json    # Configuration settings
├── logs/           # Application logs
├── results/        # Analysis results
└── tests/          # Test files
```



## Support
For support, please visit [kartiktrivedi.in](https://www.kartiktrivedi.in) or open an issue in the GitHub repository.

## Author
- **Kartik Trivedi** - [kartiktrivedi.in](https://www.kartiktrivedi.in)
- Email: hello@kartiktrivedi.in

## Version
Current version: 1.1.0

## Acknowledgments
- Thanks to all contributors who have helped shape Qually
- Special thanks to our research partners and early adopters

---
Made with ❤️ by Kartik Trivedi
