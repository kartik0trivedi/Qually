import os
import json
import csv
import datetime
import pandas as pd
# import numpy as np # Removed unused import
import requests
from typing import Dict, List, Optional, Tuple
import logging
# import argparse # Removed unused import (CLI mode removed)
from pathlib import Path
import time
import random
# import yaml # Removed unused import
from cryptography.fernet import Fernet
from requests.adapters import HTTPAdapter, Retry
import shutil # Added for os.replace fallback
import threading
# import sys # Removed unused import (CLI mode removed)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("llm_audit.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class APIKeyManager:
    """Manages API keys for different LLM providers with encryption."""

    def __init__(self, key_file: str = "api_keys.json", encryption_key_file: str = "encryption_key.key"):
        """Initialize API key manager with file paths."""
        self.key_file = key_file
        self.encryption_key_file = encryption_key_file
        self.api_keys = {}
        self._init_encryption()
        self._load_keys()

    def _init_encryption(self):
        """Initialize or load encryption key."""
        key_dir = Path(self.encryption_key_file).parent
        key_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

        if not os.path.exists(self.encryption_key_file):
            # Generate new key and save it
            self.encryption_key = Fernet.generate_key()
            with open(self.encryption_key_file, 'wb') as f:
                f.write(self.encryption_key)
            logger.info(f"Generated new encryption key at {self.encryption_key_file}")
        else:
            # Load existing key
            with open(self.encryption_key_file, 'rb') as f:
                self.encryption_key = f.read()
            logger.info(f"Loaded encryption key from {self.encryption_key_file}")

        self.cipher = Fernet(self.encryption_key)

    def _load_keys(self):
        """Load API keys from file if it exists."""
        if os.path.exists(self.key_file):
            try:
                with open(self.key_file, 'rb') as f:
                    encrypted_data = f.read()
                    # Handle case where file might be empty
                    if not encrypted_data:
                        logger.warning(f"API key file is empty: {self.key_file}")
                        self.api_keys = {}
                        return
                    decrypted_data = self.cipher.decrypt(encrypted_data)
                    self.api_keys = json.loads(decrypted_data)
                    logger.info(f"Loaded {len(self.api_keys)} API keys from {self.key_file}")
            except FileNotFoundError:
                 logger.info(f"API key file not found: {self.key_file}. Initializing empty keys.")
                 self.api_keys = {}
            except Exception as e:
                logger.error(f"Error loading or decrypting API keys from {self.key_file}: {str(e)}")
                # Ask user if they want to reset the keys
                print(f"\nError loading API keys: {str(e)}")
                print("This could be due to a corrupted file or mismatched encryption key.")
                response = input("Would you like to reset the API keys? (y/n): ")
                if response.lower() == 'y':
                    self.api_keys = {}
                    self._save_keys()
                    print("API keys have been reset.")
                else:
                    print("Continuing with empty API keys. You may need to re-enter your API keys.")
                self.api_keys = {}

    def _save_keys(self):
        """Save API keys to file."""
        try:
            encrypted_data = self.cipher.encrypt(json.dumps(self.api_keys).encode('utf-8'))
            with open(self.key_file, 'wb') as f:
                f.write(encrypted_data)
            logger.info(f"Saved {len(self.api_keys)} API keys to {self.key_file}")
        except Exception as e:
            logger.error(f"Error saving API keys to {self.key_file}: {str(e)}")
            raise

    def save_key(self, provider: str, key: str):
        """Save API key for a provider (uses lowercase provider name)."""
        provider_lower = provider.lower() # Ensure consistency
        self.api_keys[provider_lower] = key
        logger.info(f"API key for provider '{provider_lower}' updated in memory.")

        # Encrypt and save to file
        try:
            encrypted_data = self.cipher.encrypt(json.dumps(self.api_keys).encode('utf-8'))
            key_dir = Path(self.key_file).parent
            key_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            with open(self.key_file, 'wb') as f:
                f.write(encrypted_data)
            logger.info(f"API keys successfully encrypted and saved to {self.key_file}")
        except Exception as e:
             logger.error(f"Failed to encrypt or save API keys to {self.key_file}: {e}")


    def get_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider (uses lowercase provider name)."""
        return self.api_keys.get(provider.lower()) # Ensure consistency

    def list_providers(self) -> List[str]:
        """List all providers (lowercase) with saved keys."""
        return list(self.api_keys.keys())


class PromptManager:
    """Manages prompts for LLM auditing."""

    def __init__(self, project_folder: str):
        """Initialize prompt manager with project folder."""
        project_path = Path(project_folder)
        project_path.mkdir(parents=True, exist_ok=True)

        self.prompts_file = project_path / "prompts.csv"
        self.prompts_backup_file = project_path / "prompts_backup.csv"
        self.prompts = []
        logger.info(f"PromptManager initialized for project: {project_folder}")
        self.load_prompts()

    def add_prompt(self, prompt_id: str, prompt_text: str, system_prompt: str = "",
                   tags: Optional[List[str]] = None,
                   prepend_text: Optional[str] = None,
                   data_text: Optional[str] = None):
        """Add a new prompt."""
        # Combine prepend text, data text, and prompt text
        combined_text = ""
        if prepend_text:
            combined_text += prepend_text
        if data_text:
            combined_text += data_text
        if prompt_text:
            combined_text += prompt_text

        prompt = {
            "id": str(prompt_id).strip(),
            "prompt_text": combined_text.strip(),
            "system_prompt": system_prompt.strip(),
            "tags": [t.strip() for t in tags if t.strip()] if tags else [],
            "prepend_text": prepend_text.strip() if prepend_text else None,
            "data_text": data_text.strip() if data_text else None
        }

        # Always append as a new prompt
        self.prompts.append(prompt)
        logger.info(f"Added new prompt with ID: {prompt['id']}")

    def list_prompts(self) -> List[Dict]:
        """Return list of all prompts."""
        return self.prompts

    def get_prompt(self, prompt_id: str) -> Optional[Dict]:
        """Get a specific prompt by ID."""
        for prompt in self.prompts:
            if prompt['id'] == prompt_id:
                return prompt
        return None

    def load_prompts(self) -> bool:
        """Load prompts from the default CSV file."""
        logger.info(f"Attempting to load prompts from {self.prompts_file}")
        if self.prompts_file.exists():
            return self.import_prompts(str(self.prompts_file))
        else:
            logger.info("Prompts file does not exist yet. No prompts loaded.")
            return True

    def import_prompts(self, csv_file: str) -> bool:
        """Import prompts from a standard CSV."""
        try:
            df = pd.read_csv(csv_file, dtype=str)
            df.fillna("", inplace=True)

            prompt_column_candidates = ["prompt_text", "text", "content"]
            found_prompt_column = next((col for col in prompt_column_candidates if col in df.columns), None)

            if not found_prompt_column:
                logger.error(f"No suitable prompt column found in CSV. Expected one of: {prompt_column_candidates}")
                return False

            df.rename(columns={found_prompt_column: "prompt_text"}, inplace=True)

            if "id" not in df.columns:
                df["id"] = [f"text_{i:03d}" for i in range(len(df))]

            for _, row in df.iterrows():
                self.add_prompt(
                    prompt_id=row["id"],
                    prompt_text=row["prompt_text"],
                    system_prompt=row.get("system_prompt", ""),
                    tags=row.get("tags", "").split(";") if "tags" in row else [],
                    prepend_text=row.get("prepend_text"),
                    data_text=row.get("data_text")
                )

            logger.info(f"Imported {len(df)} prompts from {csv_file}")
            return True
        except Exception as e:
            logger.error(f"Error importing prompts: {e}")
            return False

    def import_data_as_multiple_prompts(self, csv_file: str, system_prompt: str,
                                       prompt_pairs: List[Tuple[str, str]], text_column: Optional[str] = None,
                                       id_column: Optional[str] = None) -> bool:
        """Import text from a CSV and create multiple prompts per row based on provided prepend and prompt pairs."""
        try:
            df = pd.read_csv(csv_file, dtype=str)
            df.fillna("", inplace=True)

            # Validate text column
            if not text_column:
                # First try to use the column that was selected in the Data Files tab
                if hasattr(self, 'data_manager') and self.data_manager.file_info:
                    text_column = self.data_manager.file_info.get('text_column')
                    if text_column and text_column in df.columns:
                        logger.info(f"Using text column '{text_column}' from data manager")
                    else:
                        # If that fails, try common column names
                        possible_text_cols = ["text", "content", "body", "main_text", "abstract", "title", "combined"]
                        for col in possible_text_cols:
                            if col in df.columns:
                                text_column = col
                                logger.info(f"Auto-detected text column: '{text_column}'")
                                break
                        if not text_column:
                            # If still no match, show all available columns in the error
                            available_cols = list(df.columns)
                            raise ValueError(
                                f"Could not auto-detect text column. Available columns: {available_cols}\n"
                                f"Please select a text column in the Data Files tab first."
                            )
            elif text_column not in df.columns:
                raise ValueError(f"Specified text column '{text_column}' not found in CSV file")

            # Validate ID column
            if not id_column:
                if hasattr(self, 'data_manager') and self.data_manager.file_info:
                    id_column = self.data_manager.file_info.get('id_column')
                if not id_column or id_column not in df.columns:
                    # If no ID column specified or found, use row numbers
                    logger.warning("No ID column specified or found. Using row numbers as IDs.")
                    df['_row_id'] = [f"row{i+1:03d}" for i in range(len(df))]
                    id_column = '_row_id'
            elif id_column not in df.columns:
                raise ValueError(f"Specified ID column '{id_column}' not found in CSV file")

            # Check for duplicate IDs
            duplicate_ids = df[id_column].duplicated()
            if duplicate_ids.any():
                duplicate_values = df.loc[duplicate_ids, id_column].unique()
                raise ValueError(f"Duplicate IDs found in column '{id_column}': {duplicate_values}")

            # Log the columns being used
            logger.info(f"Using column '{text_column}' for text data")
            logger.info(f"Using column '{id_column}' for IDs")

            # Find the next task number by checking existing prompts
            task_numbers = []
            for prompt in self.prompts:
                prompt_tags = prompt.get("tags", [])
                for tag in prompt_tags:
                    if tag.startswith("Task"):
                        try:
                            task_num = int(tag[4:])  # Extract number after "Task"
                            task_numbers.append(task_num)
                        except ValueError:
                            continue
            next_task_num = max(task_numbers) + 1 if task_numbers else 1

            # Process each prompt configuration first
            for task_idx, (prepend, prompt) in enumerate(prompt_pairs, start=0):
                current_task_num = next_task_num + task_idx
                
                # Then process each row for this prompt configuration
                for idx, row in df.iterrows():
                    text = row[text_column].strip()
                    if not text:
                        logger.warning(f"Skipping empty text in row {idx+1}")
                        continue

                    # Get the row ID from the ID column
                    row_id = str(row[id_column]).strip() if id_column else f"row{idx+1:03d}"
                    if not row_id:
                        logger.warning(f"Empty ID in row {idx+1}, skipping")
                        continue

                    # Create prompt with the current task number
                    prompt_id = f"{row_id}_task{current_task_num}"

                    self.add_prompt(
                        prompt_id=prompt_id,
                        prompt_text=prompt,
                        system_prompt=system_prompt,
                        tags=[f"Task{current_task_num}"],
                        prepend_text=prepend,
                        data_text=text
                    )

            logger.info(f"Successfully imported {len(df) * len(prompt_pairs)} prompts from {csv_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to import prompts with multiple tasks: {e}")
            return False

    def export_prompts(self) -> bool:
        """Export prompts to CSV file in project folder with backup."""
        if not self.prompts:
            logger.info("No prompts to export.")
            return True

        try:
            if self.prompts_file.exists():
                try:
                    self.prompts_file.replace(self.prompts_backup_file)
                    logger.info(f"Backed up existing prompts file to {self.prompts_backup_file}")
                except OSError as e:
                    logger.warning(f"Backup using os.replace failed ({e}). Trying copy and delete fallback.")
                    shutil.copy2(self.prompts_file, self.prompts_backup_file)
                    self.prompts_file.unlink()

            with open(self.prompts_file, "w", newline="", encoding="utf-8") as f:
                header = ["id", "prompt_text", "system_prompt", "tags", "prepend_text", "data_text"]
                writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
                writer.writeheader()
                for prompt in self.prompts:
                    writer.writerow({
                        "id": prompt.get("id", ""),
                        "prompt_text": prompt.get("prompt_text", ""),
                        "system_prompt": prompt.get("system_prompt", ""),
                        "tags": ";".join(prompt.get("tags", [])),
                        "prepend_text": prompt.get("prepend_text", ""),
                        "data_text": prompt.get("data_text", "")
                    })
            logger.info(f"Exported {len(self.prompts)} prompts to {self.prompts_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to export prompts: {e}")
            return False

    def delete_prompt(self, prompt_id: str) -> bool:
        """Delete a prompt by ID."""
        prompt_id = str(prompt_id).strip()
        initial_count = len(self.prompts)
        self.prompts = [p for p in self.prompts if p["id"] != prompt_id]
        
        if len(self.prompts) < initial_count:
            logger.info(f"Deleted prompt with ID: {prompt_id}")
            return True
        else:
            logger.warning(f"Prompt with ID {prompt_id} not found for deletion")
            return False

    def clear_prompts(self) -> bool:
        """Clear all prompts."""
        initial_count = len(self.prompts)
        if initial_count == 0:
            logger.info("No prompts to clear")
            return True
            
        self.prompts = []
        logger.info(f"Cleared all {initial_count} prompts")
        return True


class LLMProvider:
    """Base class for LLM providers."""

    def __init__(self, api_key: str = None):
        """Initialize LLM provider with API key."""
        self.api_key = api_key
        self.session = requests.Session()
        
        # Rate limiting configuration
        self._last_request_time = 0
        self._request_lock = threading.Lock()
        self.min_request_interval = self._get_default_request_interval()
        self._load_rate_limiting_config()
        
        # Configure retries for common server errors and network issues with jitter
        retries = Retry(
            total=3,
            backoff_factor=0.5, # E.g., {0.5s, 1s, 2s} delays
            status_forcelist=[429, 500, 502, 503, 504], # Retry on rate limits and server errors
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"], # Retry on relevant methods
            raise_on_status=False  # Don't raise on retries
            )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.mount("http://", HTTPAdapter(max_retries=retries)) # Also mount for http
        logger.debug(f"Initialized LLMProvider base for {type(self).__name__} with rate limiting")

    def _get_default_request_interval(self) -> float:
        """Get default request interval based on provider type."""
        provider_name = type(self).__name__.lower()
        # Conservative defaults to avoid rate limits
        intervals = {
            'openai': 1.0,      # 1 request per second
            'anthropic': 2.0,   # 1 request per 2 seconds (more conservative for Claude)
            'google': 1.5,      # 1 request per 1.5 seconds
            'mistral': 1.0,     # 1 request per second
            'grok': 1.0,        # 1 request per second
            'deepseek': 1.0     # 1 request per second
        }
        return intervals.get(provider_name, 1.0)

    def _load_rate_limiting_config(self):
        """Load rate limiting configuration from settings.json."""
        try:
            settings_file = Path("settings.json")
            if settings_file.exists():
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                    rate_config = settings.get("rate_limiting", {})
                    
                    if rate_config.get("enabled", True):
                        provider_name = type(self).__name__.lower().replace("provider", "")
                        delay_key = f"{provider_name}_delay_seconds"
                        if delay_key in rate_config:
                            self.min_request_interval = rate_config[delay_key]
                            logger.info(f"Loaded rate limiting config for {provider_name}: {self.min_request_interval}s")
        except Exception as e:
            logger.warning(f"Could not load rate limiting config: {e}. Using defaults.")

    def _throttle_request(self):
        """Throttle requests to respect rate limits."""
        with self._request_lock:
            current_time = time.time()
            time_since_last_request = current_time - self._last_request_time
            
            if time_since_last_request < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last_request
                # Add small random jitter to prevent thundering herd
                jitter = random.uniform(0.1, 0.3)
                total_sleep = sleep_time + jitter
                
                logger.debug(f"Rate limiting: sleeping for {total_sleep:.2f}s")
                time.sleep(total_sleep)
            
            self._last_request_time = time.time()


    def generate(self, prompt: str, system_prompt: str = "", parameters: Dict = None) -> Dict:
        """Generate output from prompt. To be implemented by subclasses."""
        raise NotImplementedError(f"{type(self).__name__} must implement the 'generate' method.")

    def get_available_models(self) -> List[str]:
        """Get available models for this provider. To be implemented by subclasses."""
        raise NotImplementedError(f"{type(self).__name__} must implement the 'get_available_models' method.")


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    def generate(self, prompt: str, system_prompt: str = "", parameters: Dict = None) -> Dict:
        """Generate output from prompt using OpenAI API."""
        if parameters is None:
            parameters = {}

        model = parameters.get("model", "gpt-4") # Default model
        temperature = parameters.get("temperature", 0.7)
        max_tokens = parameters.get("max_tokens", 500)
        top_p = parameters.get("top_p", 1.0)
        frequency_penalty = parameters.get("frequency_penalty", 0.0)
        presence_penalty = parameters.get("presence_penalty", 0.0)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        }

        api_url = "https://api.openai.com/v1/chat/completions"
        response = None # Initialize response to None

        try:
            # Apply rate limiting before making the request
            self._throttle_request()
            
            response = self.session.post(
                api_url,
                headers=headers,
                json=data,
                timeout=60 # Increased timeout
            )

            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            result = response.json()

            # Defensive coding: check response structure before accessing
            if "choices" in result and len(result["choices"]) > 0 and "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                output_text = result["choices"][0]["message"]["content"]
            else:
                logger.error(f"Unexpected response structure from OpenAI: {result}")
                raise ValueError("Invalid response format received from OpenAI API.")


            return {
                "text": output_text,
                "model": model,
                "provider": "openai",
                "parameters": parameters,
                "raw_response": result
            }
        except requests.exceptions.HTTPError as e:
            # Handle HTTP errors including rate limits
            if e.response.status_code == 429:
                logger.warning(f"Rate limit hit for OpenAI API: {str(e)}")
                return {
                    "text": f"Error: Rate limit exceeded - {str(e)}",
                    "model": model,
                    "provider": "openai",
                    "parameters": parameters,
                    "error": f"Rate limit: {str(e)}"
                }
            else:
                logger.error(f"HTTP error calling OpenAI API: {str(e)}")
                return {
                    "text": f"Error: HTTP {e.response.status_code} - {str(e)}",
                    "model": model,
                    "provider": "openai",
                    "parameters": parameters,
                    "error": f"HTTP {e.response.status_code}: {str(e)}"
                }
        except requests.exceptions.RequestException as e:
            # Handle connection errors, timeouts, etc.
            logger.error(f"Network error calling OpenAI API: {str(e)}")
            return {
                "text": f"Error: Network error - {str(e)}",
                "model": model,
                "provider": "openai",
                "parameters": parameters,
                "error": f"Network error: {str(e)}"
            }
        except Exception as e:
            # Handle other errors like JSON decoding, invalid response format, etc.
            logger.error(f"Error calling OpenAI API: {str(e)}")
            error_details = str(e)
            # Include response text in error if available and response object exists
            if response is not None and hasattr(response, 'text'):
                error_details += f" | Response: {response.text[:500]}" # Log first 500 chars

            return {
                "text": f"Error: {error_details}",
                "model": model,
                "provider": "openai",
                "parameters": parameters,
                "error": error_details
            }

    def get_available_models(self) -> List[str]:
        """Get available models for OpenAI."""
        api_url = "https://api.openai.com/v1/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        default_models = sorted(["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]) # Define default list once
        response = None

        try:
            response = self.session.get(api_url, headers=headers, timeout=15) # Increased timeout
            response.raise_for_status()
            models_data = response.json()

            # Defensive check for 'data' key
            if "data" not in models_data:
                 logger.error(f"Unexpected response format from OpenAI models endpoint: {models_data}")
                 raise ValueError("Invalid response format received from OpenAI models API.")

            models = models_data["data"]

            # Filter for chat models (usually contain 'gpt') and sort them
            chat_models = sorted([model["id"] for model in models if "id" in model and "gpt" in model["id"]])
            if not chat_models:
                 logger.warning("No models containing 'gpt' found via API. Returning defaults.")
                 return default_models
            return chat_models

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error getting OpenAI models: {str(e)}")
        except Exception as e:
            logger.error(f"Error getting OpenAI models: {str(e)}")
            if response is not None and hasattr(response, 'text'):
                 logger.error(f"OpenAI models response text: {response.text[:500]}")


        # Return default models if API call fails or response is invalid
        logger.warning("Returning default OpenAI models due to API error.")
        return default_models


class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) LLM provider."""

    def generate(self, prompt: str, system_prompt: str = "", parameters: Dict = None) -> Dict:
        """Generate output from prompt using Anthropic API."""
        if parameters is None:
            parameters = {}

        model = parameters.get("model", "claude-3-opus-20240229") # Default model
        temperature = parameters.get("temperature", 0.7)
        max_tokens = parameters.get("max_tokens", 500)
        # top_p = parameters.get("top_p", 1.0) # Note: Anthropic might use top_k instead or combine

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01" # Use the recommended version
        }

        data = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            # "top_p": top_p, # Include if supported and desired
            # "top_k": parameters.get("top_k", 50), # Example if using top_k
        }

        # Add system prompt if provided
        if system_prompt:
            data["system"] = system_prompt

        # Add prompt using the messages format
        data["messages"] = [{"role": "user", "content": prompt}]

        api_url = "https://api.anthropic.com/v1/messages"
        response = None

        try:
            # Apply rate limiting before making the request
            self._throttle_request()
            
            response = self.session.post(
                api_url,
                headers=headers,
                json=data,
                timeout=60 # Increased timeout
            )

            response.raise_for_status()
            result = response.json()

            # Defensive coding for response structure
            if "content" in result and len(result["content"]) > 0 and "text" in result["content"][0]:
                 output_text = result["content"][0]["text"]
            # Handle potential block reason
            elif "stop_reason" in result and result["stop_reason"] == "max_tokens":
                 output_text = result["content"][0]["text"] + " [OUTPUT TRUNCATED DUE TO MAX TOKENS]"
                 logger.warning(f"Anthropic response truncated for model {model} due to max_tokens.")
            else:
                logger.error(f"Unexpected response structure from Anthropic: {result}")
                raise ValueError("Invalid response format received from Anthropic API.")


            return {
                "text": output_text,
                "model": model,
                "provider": "anthropic",
                "parameters": parameters,
                "raw_response": result
            }
        except requests.exceptions.HTTPError as e:
            # Handle HTTP errors including rate limits
            if e.response.status_code == 429:
                logger.warning(f"Rate limit hit for Anthropic API: {str(e)}")
                return {
                    "text": f"Error: Rate limit exceeded - {str(e)}",
                    "model": model,
                    "provider": "anthropic",
                    "parameters": parameters,
                    "error": f"Rate limit: {str(e)}"
                }
            else:
                logger.error(f"HTTP error calling Anthropic API: {str(e)}")
                return {
                    "text": f"Error: HTTP {e.response.status_code} - {str(e)}",
                    "model": model,
                    "provider": "anthropic",
                    "parameters": parameters,
                    "error": f"HTTP {e.response.status_code}: {str(e)}"
                }
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error calling Anthropic API: {str(e)}")
            return {
                "text": f"Error: Network error - {str(e)}",
                "model": model,
                "provider": "anthropic",
                "parameters": parameters,
                "error": f"Network error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {str(e)}")
            error_details = str(e)
            if response is not None and hasattr(response, 'text'):
                error_details += f" | Response: {response.text[:500]}"

            return {
                "text": f"Error: {error_details}",
                "model": model,
                "provider": "anthropic",
                "parameters": parameters,
                "error": error_details
            }

    def get_available_models(self) -> List[str]:
        """Return list of available Anthropic models."""
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
            "claude-3-7-sonnet-20250219"
        ]


class GoogleProvider(LLMProvider):
    """Google (Gemini) LLM provider."""

    def generate(self, prompt: str, system_prompt: str = "", parameters: Dict = None) -> Dict:
        """Generate output from prompt using Google Gemini API."""
        if parameters is None:
            parameters = {}

        model = parameters.get("model", "gemini-1.5-flash") # Default to a common model
        temperature = parameters.get("temperature", 0.7)
        max_tokens = parameters.get("max_tokens", 8192) # Gemini often has larger defaults/limits
        top_p = parameters.get("top_p", 1.0)
        top_k = parameters.get("top_k", None) # Gemini uses topK

        # Construct API URL - ensure model name is appropriate (e.g., models/gemini-1.5-flash)
        # Check Google AI documentation for correct model naming conventions
        formatted_model = model if model.startswith("models/") else f"models/{model}"
        api_url = f"https://generativelanguage.googleapis.com/v1beta/{formatted_model}:generateContent?key={self.api_key}"

        headers = {
            "Content-Type": "application/json"
        }

        # Prepare contents structure
        contents = []
        # Google's API structure for system prompts can vary.
        # One common way is to put it as the first turn for the 'model'.
        # Another might be a dedicated 'system_instruction' field if the API supports it.
        # Prepending to the user prompt is a fallback. Check API docs for the specific model.
        # For gemini-pro and early models, prepending was common. Newer models might support system_instruction.
        # Let's assume system_instruction might be supported and try it, falling back to prepend if needed.

        system_instruction = None
        if system_prompt:
             # Try the dedicated field first (adjust structure based on actual API spec if different)
             system_instruction = {"parts": [{"text": system_prompt}]}
             # If not using system_instruction, prepend here:
             # prompt = f"System instruction: {system_prompt}\n\nUser: {prompt}"


        contents.append({"role": "user", "parts": [{"text": prompt}]})


        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
                "maxOutputTokens": max_tokens,
            }
            # Add topK if provided
            # "safetySettings": [ ... ] # Add safety settings if needed
        }
        if top_k is not None:
             data["generationConfig"]["topK"] = top_k
        if system_instruction:
             data["system_instruction"] = system_instruction


        response = None
        try:
            # Apply rate limiting before making the request
            self._throttle_request()
            
            response = self.session.post(
                api_url,
                headers=headers,
                json=data,
                timeout=90 # Increased timeout for potentially longer Gemini responses
            )

            response.raise_for_status()
            result = response.json()

            # Extract text, handling potential errors or blocked content
            output_text = ""
            error_message = None

            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"] and len(candidate["content"]["parts"]) > 0:
                    # Combine parts if there are multiple
                    output_text = "\n".join([part.get("text", "") for part in candidate["content"]["parts"]])
                elif "finishReason" in candidate and candidate["finishReason"] != "STOP":
                    # Handle cases like safety blocks, recitation, etc.
                    finish_reason = candidate.get("finishReason", "UNKNOWN")
                    safety_ratings = candidate.get("safetyRatings", [])
                    error_message = f"Generation stopped: {finish_reason}. Safety Ratings: {safety_ratings}"
                    logger.warning(f"{error_message} for model {model}")
                    output_text = f"Error: {error_message}" # Set text to error message
                else:
                    # No content and no specific finish reason? Unusual.
                    logger.error(f"Unexpected candidate structure in Google API response: {candidate}")
                    error_message = "Invalid candidate format received from Google API."
                    output_text = f"Error: {error_message}"

            elif "promptFeedback" in result and "blockReason" in result["promptFeedback"]:
                 # Handle prompt blocking
                 block_reason = result["promptFeedback"]["blockReason"]
                 safety_ratings = result["promptFeedback"].get("safetyRatings", [])
                 error_message = f"Prompt blocked due to: {block_reason}. Safety Ratings: {safety_ratings}"
                 logger.warning(f"{error_message} for model {model}")
                 output_text = f"Error: {error_message}" # Set text to error message
            else:
                logger.error(f"Unexpected response structure from Google API: {result}")
                error_message = "Invalid response format received from Google API."
                output_text = f"Error: {error_message}"


            return {
                "text": output_text,
                "model": model, # Return the original model name used
                "provider": "google",
                "parameters": parameters,
                "raw_response": result,
                "error": error_message # Include specific error message if generation failed/blocked
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error calling Google API: {str(e)}")
            return {
                "text": f"Error: Network error - {str(e)}",
                "model": model,
                "provider": "google",
                "parameters": parameters,
                "error": f"Network error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Error calling Google API: {str(e)}")
            error_details = str(e)
            if response is not None and hasattr(response, 'text'):
                 error_details += f" | Response: {response.text[:500]}"

            return {
                "text": f"Error: {error_details}",
                "model": model,
                "provider": "google",
                "parameters": parameters,
                "error": error_details
            }

    def get_available_models(self) -> List[str]:
        """Get available models for Google (Gemini)."""
        # Google provides an API endpoint for listing models
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        default_models = sorted(["gemini-1.5-pro-latest", "gemini-1.5-flash-latest", "gemini-1.0-pro"])
        response = None

        try:
            response = self.session.get(api_url, headers=headers, timeout=15)
            response.raise_for_status()
            models_data = response.json()

            if "models" not in models_data:
                 logger.error(f"Unexpected response format from Google models endpoint: {models_data}")
                 raise ValueError("Invalid response format received from Google models API.")

            # Extract model names (e.g., "models/gemini-1.5-pro-latest")
            # Filter for generative models if needed (check 'supportedGenerationMethods')
            available_models = []
            for model in models_data["models"]:
                # Check if model supports 'generateContent' and get the display name or short name
                if "generateContent" in model.get("supportedGenerationMethods", []):
                    # Prefer displayName if available, otherwise parse from name
                    model_id = model.get("displayName") or model.get("name", "").split('/')[-1]
                    if model_id:
                        available_models.append(model_id)

            if not available_models:
                 logger.warning("No models supporting 'generateContent' found via API. Returning defaults.")
                 return default_models
            return sorted(available_models)

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error getting Google models: {str(e)}")
        except Exception as e:
            logger.error(f"Error getting Google models: {str(e)}")
            if response is not None and hasattr(response, 'text'):
                 logger.error(f"Google models response text: {response.text[:500]}")


        # Return default/common models if API fails
        logger.warning("Returning default Google models due to API error.")
        return default_models


# --- NEW PROVIDER CLASSES ---

class MistralProvider(LLMProvider):
    """Mistral LLM provider."""

    def generate(self, prompt: str, system_prompt: str = "", parameters: Dict = None) -> Dict:
        """Generate output from prompt using Mistral API."""
        if parameters is None:
            parameters = {}

        # Common Mistral models: mistral-tiny, mistral-small, mistral-medium, mistral-large-latest
        # Newer models: open-mistral-7b, open-mixtral-8x7b, mistral-large-2402
        model = parameters.get("model", "mistral-large-latest") # Consider updating default
        temperature = parameters.get("temperature", 0.7)
        max_tokens = parameters.get("max_tokens", 1024) # Mistral often supports larger contexts
        top_p = parameters.get("top_p", 1.0)
        safe_prompt = parameters.get("safe_prompt", False) # Mistral specific param
        random_seed = parameters.get("random_seed", None) # Another potential param

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        messages = []
        # Mistral API generally follows OpenAI structure for messages
        if system_prompt:
            # Mistral recommendation is often to put system prompt as first user message
            # if the dedicated system role isn't performing as expected, or for specific models.
            # Check Mistral docs for best practice with the specific model.
            # Using dedicated role here:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "safe_prompt": safe_prompt,
        }
        if random_seed is not None:
             data["random_seed"] = random_seed


        # Use the official Mistral API endpoint
        api_url = "https://api.mistral.ai/v1/chat/completions"
        response = None

        try:
            # Apply rate limiting before making the request
            self._throttle_request()
            
            response = self.session.post(
                api_url,
                headers=headers,
                json=data,
                timeout=90 # Increased timeout
            )
            response.raise_for_status()
            result = response.json()

            # Check response structure (similar to OpenAI)
            output_text = ""
            error_message = None
            if "choices" in result and len(result["choices"]) > 0:
                 choice = result["choices"][0]
                 if "message" in choice and "content" in choice["message"]:
                     output_text = choice["message"]["content"]
                     # Check finish reason
                     finish_reason = choice.get("finish_reason")
                     if finish_reason == "length":
                          output_text += " [OUTPUT TRUNCATED DUE TO MAX TOKENS/LENGTH]"
                          logger.warning(f"Mistral response truncated for model {model} due to length.")
                     elif finish_reason not in [None, "stop"]: # Handle other potential reasons like 'tool_calls' if added later
                          logger.warning(f"Mistral response finished with reason: {finish_reason}")

                 else:
                     error_message = "Invalid choice message format received from Mistral API."
                     logger.error(f"Unexpected choice structure from Mistral: {choice}")
                     output_text = f"Error: {error_message}"

            else:
                error_message = "Invalid response format (no choices) received from Mistral API."
                logger.error(f"Unexpected response structure from Mistral: {result}")
                output_text = f"Error: {error_message}"


            return {
                "text": output_text,
                "model": model,
                "provider": "mistral",
                "parameters": parameters,
                "raw_response": result,
                "error": error_message
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error calling Mistral API: {str(e)}")
            return {
                "text": f"Error: Network error - {str(e)}",
                "model": model,
                "provider": "mistral",
                "parameters": parameters,
                "error": f"Network error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Error calling Mistral API: {str(e)}")
            error_details = str(e)
            if response is not None and hasattr(response, 'text'):
                 error_details += f" | Response: {response.text[:500]}"
            return {
                "text": f"Error: {error_details}",
                "model": model,
                "provider": "mistral",
                "parameters": parameters,
                "error": error_details
            }

    def get_available_models(self) -> List[str]:
        """Get available models for Mistral."""
        # Mistral has a models endpoint
        api_url = "https://api.mistral.ai/v1/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        # Update default list with newer models
        default_models = sorted([
            "mistral-large-latest", "mistral-large-2402",
            "mistral-medium-latest", "mistral-medium",
            "mistral-small-latest", "mistral-small",
            "mistral-tiny", # Older model, might be deprecated
            "open-mistral-7b", "open-mixtral-8x7b"
            ])
        response = None

        try:
            response = self.session.get(api_url, headers=headers, timeout=15)
            response.raise_for_status()
            models_data = response.json()

            if "data" not in models_data:
                 logger.error(f"Unexpected response format from Mistral models endpoint: {models_data}")
                 raise ValueError("Invalid response format received from Mistral models API.")

            # Extract model IDs
            available_models = sorted([model["id"] for model in models_data.get("data", []) if "id" in model])
            if not available_models:
                 logger.warning("No models found via Mistral API. Returning defaults.")
                 return default_models
            return available_models

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error getting Mistral models: {str(e)}")
        except Exception as e:
            logger.error(f"Error getting Mistral models: {str(e)}")
            if response is not None and hasattr(response, 'text'):
                 logger.error(f"Mistral models response text: {response.text[:500]}")


        # Return default/common models if API fails
        logger.warning("Returning default Mistral models due to API error.")
        return default_models


class GrokProvider(LLMProvider):
    """Grok LLM provider (xAI). Implementation using xAI's REST API."""
    # NOTE: API details based on publicly available information, verify with official docs.

    def generate(self, prompt: str, system_prompt: str = "", parameters: Dict = None) -> Dict:
        """Generate output from prompt using xAI Grok API."""
        if parameters is None:
            parameters = {}

        # Default parameters - check xAI docs for actual model names
        model = parameters.get("model", "grok-1")  # Example model name
        temperature = parameters.get("temperature", 0.7)
        max_tokens = parameters.get("max_tokens", 1024) # Grok might support large contexts
        top_p = parameters.get("top_p", 1.0)
        # stop_sequences = parameters.get("stop", None) # Potential parameter

        # API configuration - verify endpoint from official xAI docs
        api_url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Construct messages (assuming OpenAI compatible structure)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Request payload
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            # "stop": stop_sequences, # Add if using stop sequences
            "stream": False  # Non-streaming mode
        }

        response = None
        try:
            # Apply rate limiting before making the request
            self._throttle_request()
            
            # Make the API call
            response = self.session.post(api_url, headers=headers, json=data, timeout=90) # Increased timeout
            response.raise_for_status()  # Raise exception for 4xx/5xx errors
            result = response.json()

            # Extract output (assuming OpenAI compatible structure)
            output_text = ""
            error_message = None
            if "choices" in result and len(result["choices"]) > 0:
                 choice = result["choices"][0]
                 if "message" in choice and "content" in choice["message"]:
                     output_text = choice["message"]["content"]
                     finish_reason = choice.get("finish_reason")
                     if finish_reason == "length":
                          output_text += " [OUTPUT TRUNCATED DUE TO MAX TOKENS/LENGTH]"
                          logger.warning(f"Grok response truncated for model {model} due to length.")
                     elif finish_reason not in [None, "stop"]:
                          logger.warning(f"Grok response finished with reason: {finish_reason}")
                 else:
                     error_message = "Invalid choice message format received from Grok API."
                     logger.error(f"Unexpected choice structure from Grok: {choice}")
                     output_text = f"Error: {error_message}"
            else:
                error_message = "Invalid response format (no choices) received from Grok API."
                logger.error(f"Unexpected response structure from Grok: {result}")
                output_text = f"Error: {error_message}"


            return {
                "text": output_text,
                "model": model,
                "provider": "grok",
                "parameters": parameters,
                "raw_response": result,
                "error": error_message
            }

        except requests.exceptions.HTTPError as http_err:
            error_details = f"HTTP {http_err.response.status_code} error"
            try:
                 # Try to get more details from response body
                 error_body = http_err.response.json()
                 error_details += f": {error_body.get('error', {}).get('message', http_err.response.text)}"
            except json.JSONDecodeError:
                 error_details += f": {http_err.response.text[:500]}" # Show raw text if not JSON

            logger.error(f"HTTP error calling Grok API: {error_details}")
            return {
                "text": f"Error: {error_details}",
                "model": model,
                "provider": "grok",
                "parameters": parameters,
                "error": error_details
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error calling Grok API: {str(e)}")
            return {
                "text": f"Error: Network error - {str(e)}",
                "model": model,
                "provider": "grok",
                "parameters": parameters,
                "error": f"Network error: {str(e)}"
            }

        except Exception as e:
            logger.error(f"Unexpected error calling Grok API: {str(e)}")
            error_details = str(e)
            if response is not None and hasattr(response, 'text'):
                 error_details += f" | Response: {response.text[:500]}"
            return {
                "text": f"Error: {error_details}",
                "model": model,
                "provider": "grok",
                "parameters": parameters,
                "error": error_details
            }

    def get_available_models(self) -> List[str]:
        """Get available models for Grok from xAI API (Hypothetical endpoint)."""
        # Verify the correct endpoint from xAI documentation. '/v1/models' is common.
        api_url = "https://api.x.ai/v1/models"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        default_models = sorted(["grok-1", "grok-1.5-flash", "grok-1.5"]) # Example hardcoded models
        response = None

        try:
            response = self.session.get(api_url, headers=headers, timeout=15)
            # Handle 404 specifically if the models endpoint doesn't exist
            if response.status_code == 404:
                 logger.warning(f"Grok models endpoint '{api_url}' not found (404). Falling back to defaults.")
                 return default_models

            response.raise_for_status() # Raise for other errors (4xx, 5xx)
            models_data = response.json()

            # Assume OpenAI-like structure {"data": [{"id": "grok-1"}, ...]} - adjust if needed
            if "data" not in models_data or not isinstance(models_data["data"], list):
                 logger.error(f"Unexpected response format from Grok models endpoint: {models_data}")
                 raise ValueError("Invalid response format received from Grok models API.")

            available_models = sorted([model["id"] for model in models_data["data"] if "id" in model])
            if not available_models:
                logger.warning("No models returned from Grok API. Falling back to defaults.")
                return default_models

            logger.info(f"Retrieved Grok models from API: {available_models}")
            return available_models

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error getting Grok models: {str(e)}")
        except Exception as e:
            logger.error(f"Error getting Grok models: {str(e)}")
            if response is not None and hasattr(response, 'text'):
                 logger.error(f"Grok models response text: {response.text[:500]}")


        # Fallback to hardcoded list
        logger.warning("Returning default Grok models due to API error or missing endpoint.")
        return default_models


class DeepSeekProvider(LLMProvider):
    """DeepSeek LLM provider."""
    # Based on DeepSeek API documentation (often OpenAI compatible)

    def generate(self, prompt: str, system_prompt: str = "", parameters: Dict = None) -> Dict:
        """Generate output from prompt using DeepSeek API."""
        if parameters is None:
            parameters = {}

        # Common DeepSeek models: deepseek-chat, deepseek-coder
        model = parameters.get("model", "deepseek-chat")
        temperature = parameters.get("temperature", 0.7)
        max_tokens = parameters.get("max_tokens", 2048) # Check DeepSeek limits
        top_p = parameters.get("top_p", 1.0)
        frequency_penalty = parameters.get("frequency_penalty", 0.0)
        presence_penalty = parameters.get("presence_penalty", 0.0)
        # stop = parameters.get("stop", None) # Potential parameter

        # DeepSeek API endpoint (confirm from official docs)
        api_url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            # "stop": stop,
            "stream": False # Ensure stream is false for single completion
        }

        response = None
        try:
            # Apply rate limiting before making the request
            self._throttle_request()
            
            response = self.session.post(
                api_url,
                headers=headers,
                json=data,
                timeout=90 # Increased timeout
            )
            response.raise_for_status()
            result = response.json()

            # Check response structure (usually OpenAI compatible)
            output_text = ""
            error_message = None
            if "choices" in result and len(result["choices"]) > 0:
                 choice = result["choices"][0]
                 if "message" in choice and "content" in choice["message"]:
                     output_text = choice["message"]["content"]
                     finish_reason = choice.get("finish_reason")
                     if finish_reason == "length":
                          output_text += " [OUTPUT TRUNCATED DUE TO MAX TOKENS/LENGTH]"
                          logger.warning(f"DeepSeek response truncated for model {model} due to length.")
                     elif finish_reason not in [None, "stop"]:
                          logger.warning(f"DeepSeek response finished with reason: {finish_reason}")
                 else:
                     error_message = "Invalid choice message format received from DeepSeek API."
                     logger.error(f"Unexpected choice structure from DeepSeek: {choice}")
                     output_text = f"Error: {error_message}"
            else:
                error_message = "Invalid response format (no choices) received from DeepSeek API."
                logger.error(f"Unexpected response structure from DeepSeek: {result}")
                output_text = f"Error: {error_message}"


            return {
                "text": output_text,
                "model": model,
                "provider": "deepseek",
                "parameters": parameters,
                "raw_response": result,
                "error": error_message
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error calling DeepSeek API: {str(e)}")
            return {
                "text": f"Error: Network error - {str(e)}",
                "model": model,
                "provider": "deepseek",
                "parameters": parameters,
                "error": f"Network error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {str(e)}")
            error_details = str(e)
            if response is not None and hasattr(response, 'text'):
                 error_details += f" | Response: {response.text[:500]}"
            return {
                "text": f"Error: {error_details}",
                "model": model,
                "provider": "deepseek",
                "parameters": parameters,
                "error": error_details
            }

    def get_available_models(self) -> List[str]:
        """Get available models for DeepSeek."""
        # DeepSeek might have a models endpoint, often similar to OpenAI's
        api_url = "https://api.deepseek.com/v1/models" # Check if this endpoint exists and works
        headers = {"Authorization": f"Bearer {self.api_key}"}
        default_models = sorted(["deepseek-chat", "deepseek-coder"])
        response = None

        try:
            response = self.session.get(api_url, headers=headers, timeout=15)
            # Handle 404 specifically
            if response.status_code == 404:
                logger.warning("DeepSeek models endpoint not found (404). Returning defaults.")
                return default_models

            response.raise_for_status() # Raise for other errors
            models_data = response.json()

            # Assume OpenAI-like structure {"data": [{"id": "model_name"}, ...]}
            if "data" not in models_data or not isinstance(models_data["data"], list):
                 logger.error(f"Unexpected response format from DeepSeek models endpoint: {models_data}")
                 raise ValueError("Invalid response format received from DeepSeek models API.")

            available_models = sorted([model["id"] for model in models_data.get("data", []) if "id" in model])
            if not available_models:
                 logger.warning("No models found via DeepSeek API. Returning defaults.")
                 return default_models
            return available_models

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error getting DeepSeek models: {str(e)}")
        except Exception as e:
            logger.error(f"Error getting DeepSeek models: {str(e)}")
            if response is not None and hasattr(response, 'text'):
                 logger.error(f"DeepSeek models response text: {response.text[:500]}")


        # Return default/common models if API fails or endpoint doesn't exist
        logger.warning("Returning default DeepSeek models due to API error or missing endpoint.")
        return default_models


# --- END NEW PROVIDER CLASSES ---


class ProviderFactory:
    """Factory to create LLM providers."""

    @staticmethod
    def create_provider(provider_name: str, api_key: str) -> Optional[LLMProvider]:
        """Create a provider instance based on name."""
        if not api_key:
            logger.error(f"Cannot create provider '{provider_name}': No API key provided")
            return None

        provider_name_lower = provider_name.lower() # Use lowercase for matching
        provider_class_map = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "google": GoogleProvider,
            "mistral": MistralProvider,
            "grok": GrokProvider,
            "deepseek": DeepSeekProvider,
        }

        provider_class = provider_class_map.get(provider_name_lower)

        if provider_class:
            try:
                provider = provider_class(api_key)
                # Test the provider by getting available models
                models = provider.get_available_models()
                if not models:
                    logger.error(f"Provider '{provider_name}' failed to return any available models")
                    return None
                return provider
            except Exception as e:
                logger.error(f"Failed to instantiate provider '{provider_name}': {e}")
                return None
        else:
            logger.error(f"Unknown provider requested: {provider_name}")
            return None


class ExperimentManager:
    """Manages LLM auditing experiments."""

    def __init__(self, project_folder: str):
        """Initialize experiment manager with project folder."""
        self.project_folder = Path(project_folder) # Use pathlib for better path handling
        self.results_folder = self.project_folder / "results"
        self.results_folder.mkdir(parents=True, exist_ok=True) # Create folder if not exists

        # Determine paths for API keys relative to project folder
        key_file = str(self.project_folder / "api_keys.json")
        encryption_key_file = str(self.project_folder / "encryption_key.key")

        # Create API key manager using project paths
        self.api_key_manager = APIKeyManager(key_file=key_file, encryption_key_file=encryption_key_file)

        # Create prompt manager, passing project folder path as string
        self.prompt_manager = PromptManager(str(self.project_folder))
        # Note: PromptManager loads prompts itself during its __init__

        # Create experiments file path
        self.experiments_file = self.project_folder / "experiments.json"
        self.experiments = self._load_experiments()
        logger.info(f"ExperimentManager initialized for project: {self.project_folder.resolve()}")


    def _load_experiments(self) -> Dict:
        """Load experiments from file if it exists."""
        if self.experiments_file.exists():
            try:
                with open(self.experiments_file, "r", encoding="utf-8") as f:
                    # Handle empty file case
                    content = f.read()
                    if not content:
                        logger.warning(f"Experiments file is empty: {self.experiments_file}")
                        return {"experiments": []}
                    return json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from experiments file {self.experiments_file}: {str(e)}. Initializing empty list.")
            except Exception as e:
                logger.error(f"Error loading experiments from {self.experiments_file}: {str(e)}. Initializing empty list.")
        else:
             logger.info(f"Experiments file not found: {self.experiments_file}. Initializing empty list.")

        return {"experiments": []} # Return default structure if file doesn't exist or fails to load


    def _save_experiments(self):
        """Save experiments to file."""
        try:
            with open(self.experiments_file, "w", encoding="utf-8") as f:
                json.dump(self.experiments, f, indent=2, ensure_ascii=False) # Use ensure_ascii=False for broader char support
            logger.debug(f"Experiments saved to {self.experiments_file}")
        except Exception as e:
            logger.error(f"Error saving experiments to {self.experiments_file}: {str(e)}")

    def create_experiment(self, name: str, description: str = "") -> str:
        """Create a new experiment and return its ID."""
        experiment_id = f"exp_{int(time.time())}"

        experiment = {
            "id": experiment_id,
            "name": name.strip(),
            "description": description.strip(),
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), # Use UTC time
            "conditions": []
        }

        # Ensure 'experiments' key exists and is a list
        if "experiments" not in self.experiments or not isinstance(self.experiments["experiments"], list):
             self.experiments["experiments"] = []

        self.experiments["experiments"].append(experiment)
        self._save_experiments()
        logger.info(f"Created experiment '{name}' with ID: {experiment_id}")

        return experiment_id

    def add_condition(self, experiment_id: str, condition_name: str,
                     provider: str, model: str, parameters: Dict = None) -> Optional[str]:
        """Add a condition to an experiment and return its ID."""
        if parameters is None:
            parameters = {}

        # Find experiment
        experiment = self.get_experiment(experiment_id) # Use helper method

        if experiment is None:
            logger.error(f"Cannot add condition: Experiment not found with ID {experiment_id}")
            return None # Return None if experiment not found

        # Create condition
        condition_id = f"cond_{int(time.time())}_{len(experiment.get('conditions', []))}" # Make ID slightly more unique

        condition = {
            "id": condition_id,
            "name": condition_name.strip(),
            "provider": provider.lower().strip(), # Store provider name consistently lowercase
            "model": model.strip(),
            "parameters": parameters, # Assumes parameters are already processed (e.g., numeric types)
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat() # Use UTC time
        }

        # Ensure 'conditions' list exists
        if "conditions" not in experiment:
            experiment["conditions"] = []

        experiment["conditions"].append(condition)
        self._save_experiments()
        logger.info(f"Added condition '{condition_name}' (ID: {condition_id}) to experiment '{experiment['name']}' (ID: {experiment_id})")

        return condition_id

    def run_experiment(self, experiment_id: str, prompt_ids: List[str]) -> Optional[Dict]:
        """Run an experiment with the given prompts and return results."""
        # Find experiment
        experiment = self.get_experiment(experiment_id)

        if experiment is None:
            logger.error(f"Experiment not found during run: {experiment_id}")
            return None # Return None if experiment not found

        # Get prompts
        prompts_to_run = []
        valid_prompt_ids = set(p['id'] for p in self.prompt_manager.list_prompts())
        for prompt_id in prompt_ids:
            prompt_id_str = str(prompt_id).strip()
            if prompt_id_str in valid_prompt_ids:
                 prompt = self.prompt_manager.get_prompt(prompt_id_str)
                 if prompt: # Should always be found if ID was in the set, but double check
                     prompts_to_run.append(prompt)
                 else: # Should not happen if logic is correct
                      logger.error(f"Internal inconsistency: Prompt ID {prompt_id_str} in set but not retrieved.")
            else:
                logger.warning(f"Prompt ID '{prompt_id_str}' not found in loaded prompts. Skipping.")


        if not prompts_to_run:
            logger.error("No valid prompts selected or found for the experiment.")
            return None # Return None if no prompts

        # Create results structure
        results = {
            "experiment_id": experiment_id,
            "experiment_name": experiment.get("name", "Unknown Experiment"),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), # Use UTC time
            "conditions": {}, # Will store details of conditions run
            "prompt_results": {} # Will store results per prompt
        }

        conditions_to_run = experiment.get("conditions", [])
        if not conditions_to_run:
            logger.warning(f"Experiment '{experiment_id}' has no conditions defined. Nothing to run.")
            return results # Return empty results structure


        # Run each condition for each prompt
        total_generations = len(conditions_to_run) * len(prompts_to_run)
        current_generation = 0
        logger.info(f"Starting experiment '{experiment.get('name', '')}'. Total generations to attempt: {total_generations}")

        for condition in conditions_to_run:
            condition_id = condition.get("id")
            provider_name = condition.get("provider") # Already lowercase from add_condition
            model = condition.get("model")
            parameters = condition.get("parameters", {}) # Ensure parameters dict exists

            if not all([condition_id, provider_name, model]):
                 logger.warning(f"Skipping invalid condition data in experiment {experiment_id}: {condition}")
                 continue


            # Get API key
            api_key = self.api_key_manager.get_key(provider_name)
            if not api_key:
                logger.error(f"API key not found for provider: '{provider_name}'. Skipping condition '{condition.get('name', condition_id)}'.")
                # Log skipped condition in results? Maybe add an error entry for the condition itself?
                # For now, just skip.
                continue

            # Create provider
            provider = ProviderFactory.create_provider(provider_name, api_key)
            if provider is None:
                 logger.error(f"Failed to create provider '{provider_name}'. Skipping condition '{condition.get('name', condition_id)}'.")
                 continue # Skip this condition if provider creation failed


            # Add condition details to results (only if provider was created)
            results["conditions"][condition_id] = {
                "name": condition.get("name", "Unnamed Condition"),
                "provider": provider_name,
                "model": model,
                "parameters": parameters
            }

            # Process each prompt for this condition
            for prompt in prompts_to_run:
                current_generation += 1
                prompt_id = prompt["id"]
                prompt_text = prompt.get("prompt_text", "")
                system_prompt = prompt.get("system_prompt", "")

                # Initialize prompt result structure if it doesn't exist
                if prompt_id not in results["prompt_results"]:
                    results["prompt_results"][prompt_id] = {
                        "prompt_text": prompt_text,
                        "system_prompt": system_prompt,
                        "personas": prompt.get("personas", []), # Include personas/tags if available
                        "tags": prompt.get("tags", []),
                        "condition_results": {}
                    }

                # Combine base parameters with model-specific ones if needed
                # Here, the model is already part of the condition's parameters or default
                full_parameters = parameters.copy()
                # Ensure the 'model' parameter used for generation matches the condition's model
                full_parameters["model"] = model

                # Generate response
                response_data = {} # To store the result of generation
                try:
                    logger.info(f"({current_generation}/{total_generations}) Generating for prompt '{prompt_id}' with condition '{condition.get('name', condition_id)}' (Provider: {provider_name}, Model: {model})")

                    start_time = time.monotonic()
                    response = provider.generate(
                        prompt=prompt_text,
                        system_prompt=system_prompt,
                        parameters=full_parameters # Pass the combined parameters
                    )
                    end_time = time.monotonic()
                    duration = end_time - start_time
                    logger.info(f"Response received in {duration:.2f} seconds.")

                    # Store response data
                    response_data = {
                        "text": response.get("text", "Error: No text returned"), # Handle missing text
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "duration_seconds": duration,
                        "raw_response": response.get("raw_response"), # Include raw response if available
                        "error": response.get("error") # Include error if generation failed (e.g., content filter)
                    }
                    if response_data["error"]:
                         logger.warning(f"Generation for prompt '{prompt_id}', condition '{condition.get('name', condition_id)}' resulted in an error: {response_data['error']}")


                except Exception as e:
                    # Catch unexpected errors during the generate call itself
                    end_time = time.monotonic()
                    duration = end_time - start_time if 'start_time' in locals() else None
                    logger.error(f"Unexpected error during generation for prompt '{prompt_id}', condition '{condition.get('name', condition_id)}': {str(e)}", exc_info=True) # Log traceback
                    response_data = {
                        "text": f"Error: Unexpected failure - {str(e)}",
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "duration_seconds": duration,
                        "error": f"Unexpected failure: {str(e)}"
                    }

                # Add response data (or error data) to results
                results["prompt_results"][prompt_id]["condition_results"][condition_id] = response_data


        # Save results to JSON file
        results_filename = f"results_{experiment_id}_{int(time.time())}.json"
        results_file_path = self.results_folder / results_filename

        try:
            with open(results_file_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Experiment '{experiment.get('name', '')}' finished. Results saved to {results_file_path}")
            # Add the results file path to the results dictionary
            results["results_file"] = str(results_file_path)
        except Exception as e:
            logger.error(f"Error saving results JSON to {results_file_path}: {str(e)}")

        return results

    def export_results_csv(self, results: Dict) -> str:
        """Export results to CSV file for analysis."""
        experiment_id = results.get("experiment_id", "unknown_exp")
        timestamp = int(time.time())

        csv_filename = f"results_{experiment_id}_{timestamp}.csv"
        csv_file_path = self.results_folder / csv_filename

        try:
            rows = []
            # Check if results structure is valid
            if not results or "prompt_results" not in results or "conditions" not in results:
                 logger.error("Cannot export CSV: Invalid or empty results structure provided.")
                 return ""

            for prompt_id, prompt_data in results.get("prompt_results", {}).items():
                prompt_text = prompt_data.get("prompt_text", "")
                system_prompt = prompt_data.get("system_prompt", "")
                tags = ";".join(prompt_data.get("tags", [])) # Join lists back

                # Iterate through conditions actually run for this prompt
                for condition_id, condition_result in prompt_data.get("condition_results", {}).items():
                    # Get condition details from the main conditions dict
                    condition_details = results.get("conditions", {}).get(condition_id)
                    if not condition_details:
                        logger.warning(f"Details not found for condition_id '{condition_id}' in results. Skipping row for prompt '{prompt_id}'.")
                        continue

                    condition_name = condition_details.get("name", "Unknown Condition")
                    provider = condition_details.get("provider", "Unknown Provider")
                    model = condition_details.get("model", "Unknown Model")
                    # Safely serialize parameters dictionary to JSON string
                    try:
                        parameters_str = json.dumps(condition_details.get("parameters", {}))
                    except TypeError as e:
                        logger.warning(f"Could not JSON serialize parameters for condition {condition_id}: {e}. Saving as string.")
                        parameters_str = str(condition_details.get("parameters", {}))

                    response_text = condition_result.get("text", "")
                    error = condition_result.get("error", "") # Get error string if present
                    response_timestamp = condition_result.get("timestamp", "") # Get response timestamp
                    duration = condition_result.get("duration_seconds", None) # Get duration

                    row = {
                        "experiment_id": experiment_id,
                        "experiment_name": results.get("experiment_name", "Unknown Experiment"),
                        "run_timestamp": results.get("timestamp", ""), # Timestamp of the overall run
                        "prompt_id": prompt_id,
                        "prompt_text": prompt_text,
                        "system_prompt": system_prompt,
                        "prompt_tags": tags,
                        "condition_id": condition_id,
                        "condition_name": condition_name,
                        "provider": provider,
                        "model": model,
                        "parameters": parameters_str,
                        "response_text": response_text,
                        "response_timestamp": response_timestamp,
                        "response_duration_sec": duration,
                        "error": error # Include the error message string
                    }
                    rows.append(row)

            if not rows:
                logger.warning("No data rows generated for CSV export. Experiment might have failed or had no valid results.")
                return ""

            # Create DataFrame and save to CSV
            df = pd.DataFrame(rows)
            # Define column order for clarity
            column_order = [
                "experiment_id", "experiment_name", "run_timestamp",
                "prompt_id", "prompt_text", "system_prompt", "prompt_tags",
                "condition_id", "condition_name", "provider", "model", "parameters",
                "response_text", "response_timestamp", "response_duration_sec", "error"
            ]
            # Ensure all expected columns exist, add missing ones with default value (e.g., None or "")
            for col in column_order:
                 if col not in df.columns:
                     df[col] = None # Or "" if preferred

            df = df[column_order] # Reorder columns

            df.to_csv(csv_file_path, index=False, encoding="utf-8")

            logger.info(f"CSV results saved to {csv_file_path}")
            return str(csv_file_path) # Return path as string

        except Exception as e:
            logger.error(f"Error exporting results to CSV ({csv_file_path}): {str(e)}")
            return ""

    def export_blinded_csv(self, results: Dict) -> str:
        """Export blinded results to CSV file for review."""
        experiment_id = results.get("experiment_id", "unknown_exp")
        timestamp = int(time.time())

        csv_filename = f"blinded_results_{experiment_id}_{timestamp}.csv"
        csv_file_path = self.results_folder / csv_filename
        mapping_filename = f"condition_mapping_{experiment_id}_{timestamp}.json"
        mapping_file_path = self.results_folder / mapping_filename

        try:
             # Check if results structure is valid
            if not results or "prompt_results" not in results or "conditions" not in results:
                 logger.error("Cannot export blinded CSV: Invalid or empty results structure provided.")
                 return ""

            conditions_dict = results.get("conditions", {})
            if not conditions_dict:
                 logger.warning("Cannot export blinded CSV: No conditions found in results.")
                 return ""


            # Create mapping for conditions to anonymize them
            # Ensure consistent ordering for mapping generation
            sorted_condition_ids = sorted(conditions_dict.keys())

            # Generate mapping like Condition_A, Condition_B, etc.
            condition_mapping = {cond_id: f"Condition_{chr(65+i)}" for i, cond_id in enumerate(sorted_condition_ids)}
            # Store the reverse mapping (Anonymized -> Original Details) as well for the JSON file
            mapping_details = {
                 f"Condition_{chr(65+i)}": {
                     "original_id": cond_id,
                     "name": conditions_dict[cond_id].get("name", "Unknown"),
                     "provider": conditions_dict[cond_id].get("provider", "Unknown"),
                     "model": conditions_dict[cond_id].get("model", "Unknown"),
                     "parameters": conditions_dict[cond_id].get("parameters", {})
                 }
                 for i, cond_id in enumerate(sorted_condition_ids)
            }


            rows = []
            for prompt_id, prompt_data in results.get("prompt_results", {}).items():
                prompt_text = prompt_data.get("prompt_text", "")
                system_prompt = prompt_data.get("system_prompt", "") # Include system prompt if needed for context

                # Group responses by prompt
                prompt_row_base = {
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_text,
                    "system_prompt": system_prompt,
                }

                responses_for_prompt = {}
                # Ensure consistent order of conditions in the output row
                for condition_id in sorted_condition_ids:
                    condition_result = prompt_data.get("condition_results", {}).get(condition_id)
                    anonymized_condition_label = condition_mapping.get(condition_id) # Should always exist if condition_id is from sorted keys

                    if condition_result and anonymized_condition_label:
                        response_text = condition_result.get("text", "")
                        error = condition_result.get("error", "") # Include error status if needed

                        # Add response under its anonymized label
                        responses_for_prompt[f"{anonymized_condition_label}_response"] = response_text
                        if error: # Optionally add error column per condition
                             responses_for_prompt[f"{anonymized_condition_label}_error"] = error
                    elif anonymized_condition_label:
                         # Handle case where a condition was defined but didn't run/produce results for this prompt
                         responses_for_prompt[f"{anonymized_condition_label}_response"] = "[NO RESULT]"
                         responses_for_prompt[f"{anonymized_condition_label}_error"] = "[NO RESULT]"


                # Combine base prompt info with all anonymized responses for that prompt
                if responses_for_prompt: # Only add row if there were responses/placeholders
                    full_row = {**prompt_row_base, **responses_for_prompt}
                    rows.append(full_row)


            if not rows:
                logger.warning("No data rows generated for blinded CSV export.")
                return ""

            # Create DataFrame
            df = pd.DataFrame(rows)

            # Define column order dynamically based on generated labels
            base_cols = ["prompt_id", "prompt_text", "system_prompt"]
            # Generate condition column names in sorted order (A, B, C...)
            condition_response_cols = []
            condition_error_cols = []
            for i in range(len(sorted_condition_ids)):
                 label = f"Condition_{chr(65+i)}"
                 condition_response_cols.append(f"{label}_response")
                 # Check if any error columns were actually created before adding them
                 if any(f"{label}_error" in row for row in rows):
                      condition_error_cols.append(f"{label}_error")


            column_order = base_cols + condition_response_cols + condition_error_cols

            # Ensure all expected columns exist in the DataFrame, adding if necessary
            for col in column_order:
                 if col not in df.columns:
                     df[col] = "[N/A]" # Or some other placeholder

            df = df[column_order] # Reorder columns


            # Save blinded CSV
            df.to_csv(csv_file_path, index=False, encoding="utf-8")

            # Save mapping details for reference
            try:
                with open(mapping_file_path, "w", encoding="utf-8") as f:
                    json.dump(mapping_details, f, indent=2, ensure_ascii=False)
                logger.info(f"Blinded condition mapping saved to {mapping_file_path}")
            except Exception as e:
                 logger.error(f"Error saving condition mapping file {mapping_file_path}: {str(e)}")


            logger.info(f"Blinded CSV results saved to {csv_file_path}")
            return str(csv_file_path) # Return path as string

        except Exception as e:
            logger.error(f"Error exporting blinded results to CSV ({csv_file_path}): {str(e)}")
            # Clean up potentially incomplete files if error occurred
            if csv_file_path.exists():
                 csv_file_path.unlink(missing_ok=True)
            if mapping_file_path.exists():
                 mapping_file_path.unlink(missing_ok=True)

            return ""


    def list_experiments(self) -> List[Dict]:
        """List all experiments."""
        # Ensure 'experiments' key exists and is a list
        return self.experiments.get("experiments", [])


    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get experiment by ID."""
        exp_id_str = str(experiment_id).strip()
        for experiment in self.list_experiments(): # Use list_experiments to access safely
            if experiment.get("id") == exp_id_str:
                return experiment
        return None

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment by ID."""
        exp_id_str = str(experiment_id).strip()
        experiments = self.list_experiments()
        
        # Find and remove the experiment
        for i, experiment in enumerate(experiments):
            if experiment.get("id") == exp_id_str:
                # Remove the experiment
                experiments.pop(i)
                self._save_experiments()
                logger.info(f"Deleted experiment with ID: {exp_id_str}")
                return True
        
        logger.warning(f"Experiment not found for deletion: {exp_id_str}")
        return False

    def create_experiment_with_conditions(self, name: str, description: str = "", conditions: List[Dict] = None) -> str:
        """Create a new experiment with multiple conditions and return its ID."""
        if conditions is None:
            conditions = []

        experiment_id = f"exp_{int(time.time())}"

        experiment = {
            "id": experiment_id,
            "name": name.strip(),
            "description": description.strip(),
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "conditions": []
        }

        # Ensure 'experiments' key exists and is a list
        if "experiments" not in self.experiments or not isinstance(self.experiments["experiments"], list):
            self.experiments["experiments"] = []

        # Add conditions to the experiment
        for i, condition in enumerate(conditions):
            condition_id = f"cond_{int(time.time())}_{i}"
            experiment["conditions"].append({
                "id": condition_id,
                "name": condition.get("name", "").strip(),
                "provider": condition.get("provider", "").lower().strip(),
                "model": condition.get("model", "").strip(),
                "parameters": condition.get("parameters", {}),
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            })

        self.experiments["experiments"].append(experiment)
        self._save_experiments()
        logger.info(f"Created experiment '{name}' with ID: {experiment_id} and {len(conditions)} conditions")

        return experiment_id


