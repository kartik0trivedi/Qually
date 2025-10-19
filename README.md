# Qually
## Overview
Qually is an advanced LLM Audit Tool designed specifically for Social Science Research. This platform provides researchers with powerful capabilities to evaluate and audit Language Learning Models in the context of social science research without getting into the weeds of coding and programming.

## Features
- **Multi-Provider LLM Support**: OpenAI, Anthropic (Claude), Google (Gemini), Mistral, Grok, and DeepSeek
- **Rate Limiting Protection**: Built-in rate limiting to prevent API quota issues
- **Experiment Management**: Create, run, and manage LLM experiments with multiple conditions
- **Data Import/Export**: CSV import for prompts and data, multiple export formats
- **Research-Focused Interface**: User-friendly GUI designed for researchers
- **Cross-Platform**: MacOS and Windows support
- **Secure API Key Management**: Encrypted storage of API keys

## Prerequisites
- Python 3.11 or higher
- pip (Python package manager)
- API keys for your preferred LLM providers (OpenAI, Anthropic, Google, etc.)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kartik0trivedi/Qually.git
cd Qually
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages include:
- PyQt6 (≥6.9.0)
- pandas (≥2.2.0)
- requests (≥2.32.0)
- cryptography (≥44.0.0)

## Running the Application

### Development Mode
Run the application directly:
```bash
python qually_gui.py
```

### Configuration
1. Launch the application
2. Go to Settings and add your API keys for the LLM providers you want to use
3. Configure rate limiting settings if needed (see `settings.json`)

### Building the Application

#### MacOS
```bash
python build_mac.py
```

#### Windows
```bash
python build_win.py
```

## Project Structure
```
Qually/
├── qually_gui.py           # Main GUI application
├── qually_tool.py          # Core LLM provider and experiment logic
├── requirements.txt        # Python dependencies
├── settings.json          # Configuration settings (ignored by git)
├── build_mac.py           # macOS build script
├── build_win.py           # Windows build script
├── resources/             # UI resources (icons, themes, fonts)
└── README.md              # This file
```

## Usage Guide

1. **Setup**: Install dependencies and configure API keys
2. **Import Data**: Load CSV files with your prompts and data
3. **Create Experiments**: Set up experiments with different LLM providers and conditions
4. **Run Experiments**: Execute experiments and monitor progress
5. **Export Results**: Download results in CSV format for analysis

## Rate Limiting

Qually LE2 includes built-in rate limiting to prevent hitting API quotas. Configure delays in `settings.json`:
- OpenAI: 1 second delay
- Anthropic: 2 second delay (more conservative)
- Other providers: 1-1.5 second delays



## Support
For support, please visit [kartiktrivedi.in](https://www.kartiktrivedi.in) or open an issue in the GitHub repository.

## Author
- **Kartik Trivedi** - [kartiktrivedi.in](https://www.kartiktrivedi.in)
- Email: hello@kartiktrivedi.in

## Version
Current version: 2.0.0 (LE2)

## Acknowledgments
- Funding for this application was provided by the Advanced Rehabilitation Research and Training (ARRT) Program on Employment at the University of New Hampshire, which is funded by the National Institute for Disability, Independent Living, and Rehabilitation Research, in the Administration for Community Living, at the U.S. Department of Health and Human Services (DHHS) under grant number 90AREM000401. The contents do not necessarily represent the policy of DHHS and you should not assume endorsement by the federal government (EDGAR, 75.620 (b)).
- Thanks to all contributors who have helped shape Qually
- Special thanks to our research partners and early adopters

---
Made with ❤️ by Kartik Trivedi
