# Qually
## Overview
Qually is an LLM Audit Tool designed specifically for Social Science Research. This platform provides researchers with powerful capabilities to evaluate and audit Language Learning Models in the context of social science research.

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
Because the code is organized under a `src/` directory, you must set the `PYTHONPATH` so Python can find your modules:
```bash
PYTHONPATH=src python qually_gui.py
```

### Building the Application
The application can be built into a native MacOS application using py2app:
```bash
python build_mac.py py2app
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
- Funding for this application was provided by the Advanced Rehabilitation Research and Training (ARRT) Program on Employment at the University of New Hampshire, which is funded by the National Institute for Disability, Independent Living, and Rehabilitation Research, in the Administration for Community Living, at the U.S. Department of Health and Human Services (DHHS) under grant number 90AREM000401. The contents do not necessarily represent the policy of DHHS and you should not assume endorsement by the federal government (EDGAR, 75.620 (b)).
- Thanks to all contributors who have helped shape Qually
- Special thanks to our research partners and early adopters

---
Made with ❤️ by Kartik Trivedi
