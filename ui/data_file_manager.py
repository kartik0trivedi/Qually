import json
import logging

import pandas as pd


gui_logger = logging.getLogger("QuallyGUI")


class DataFileManager:
    def __init__(self):
        self.data = None
        self.file_info = None

    def import_csv(self, file_path):
        """Import a CSV file and store both the data and file info."""
        try:
            self.data = pd.read_csv(file_path, dtype=str)
            self.data.fillna("", inplace=True)

            possible_text_cols = ["text", "content", "body", "main_text"]
            text_column = None
            for col in possible_text_cols:
                if col in self.data.columns:
                    text_column = col
                    break

            self.file_info = {
                "file_path": file_path,
                "text_column": text_column,
                "total_rows": len(self.data),
                "file_type": "csv",
            }

            return self.data.head(5), list(self.data.columns)
        except Exception as e:
            gui_logger.error(f"Failed to import CSV: {e}", exc_info=True)
            raise

    def import_json(self, file_path):
        """Import a JSON file and store both the data and file info."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            if isinstance(json_data, list):
                self.data = pd.DataFrame(json_data)
            elif isinstance(json_data, dict):
                self.data = pd.DataFrame([json_data])
            else:
                raise ValueError("JSON file must contain either an array of objects or a single object")

            self.data.fillna("", inplace=True)

            possible_text_cols = ["text", "content", "body", "main_text"]
            text_column = None
            for col in possible_text_cols:
                if col in self.data.columns:
                    text_column = col
                    break

            self.file_info = {
                "file_path": file_path,
                "text_column": text_column,
                "total_rows": len(self.data),
                "file_type": "json",
            }

            return self.data.head(5), list(self.data.columns)
        except Exception as e:
            gui_logger.error(f"Failed to import JSON: {e}", exc_info=True)
            raise

    def has_data(self):
        """Check if data is loaded."""
        return self.data is not None and self.file_info is not None

    def import_excel(self, file_path):
        """Import an Excel file and store both the data and file info."""
        try:
            self.data = pd.read_excel(file_path, dtype=str)
            self.data.fillna("", inplace=True)

            possible_text_cols = ["text", "content", "body", "main_text"]
            text_column = None
            for col in possible_text_cols:
                if col in self.data.columns:
                    text_column = col
                    break

            self.file_info = {
                "file_path": file_path,
                "text_column": text_column,
                "total_rows": len(self.data),
                "file_type": "excel",
            }

            return self.data.head(5), list(self.data.columns)
        except Exception as e:
            gui_logger.error(f"Failed to import Excel file: {e}", exc_info=True)
            raise
