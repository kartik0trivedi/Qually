import os
import json
import csv
import datetime
import pandas as pd
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from pathlib import Path
import time
from cryptography.fernet import Fernet
from requests.adapters import HTTPAdapter, Retry
import shutil

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
                                f"No suitable text column found. Available columns: {available_cols}")

            # Validate ID column
            if not id_column:
                # Try common ID column names
                possible_id_cols = ["id", "ID", "Id", "index", "Index"]
                for col in possible_id_cols:
                    if col in df.columns:
                        id_column = col
                        logger.info(f"Auto-detected ID column: '{id_column}'")
                        break
                if not id_column:
                    # If no ID column found, create one
                    id_column = "id"
                    df[id_column] = [f"text_{i:03d}" for i in range(len(df))]
                    logger.info("Created new ID column")

            # Process each row
            for _, row in df.iterrows():
                text = row[text_column]
                prompt_id = row[id_column]

                # Create a prompt for each prompt pair
                for prepend, prompt in prompt_pairs:
                    combined_id = f"{prompt_id}_{prepend[:10]}"  # Use first 10 chars of prepend for uniqueness
                    self.add_prompt(
                        prompt_id=combined_id,
                        prompt_text=prompt,
                        system_prompt=system_prompt,
                        prepend_text=prepend,
                        data_text=text
                    )

            logger.info(f"Imported {len(df)} rows as {len(df) * len(prompt_pairs)} prompts")
            return True
        except Exception as e:
            logger.error(f"Error importing data as prompts: {e}")
            return False

    def export_prompts(self) -> bool:
        """Export prompts to CSV file."""
        try:
            # Create backup of existing file if it exists
            if self.prompts_file.exists():
                shutil.copy2(self.prompts_file, self.prompts_backup_file)
                logger.info(f"Created backup of prompts at {self.prompts_backup_file}")

            # Export to CSV
            with open(self.prompts_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['id', 'prompt_text', 'system_prompt', 'tags', 'prepend_text', 'data_text'])
                writer.writeheader()
                for prompt in self.prompts:
                    # Convert tags list to semicolon-separated string
                    prompt_copy = prompt.copy()
                    prompt_copy['tags'] = ';'.join(prompt['tags']) if prompt['tags'] else ''
                    writer.writerow(prompt_copy)

            logger.info(f"Exported {len(self.prompts)} prompts to {self.prompts_file}")
            return True
        except Exception as e:
            logger.error(f"Error exporting prompts: {e}")
            return False

    def delete_prompt(self, prompt_id: str) -> bool:
        """Delete a prompt by ID."""
        try:
            initial_count = len(self.prompts)
            self.prompts = [p for p in self.prompts if p['id'] != prompt_id]
            if len(self.prompts) < initial_count:
                logger.info(f"Deleted prompt with ID: {prompt_id}")
                return True
            else:
                logger.warning(f"No prompt found with ID: {prompt_id}")
                return False
        except Exception as e:
            logger.error(f"Error deleting prompt: {e}")
            return False

    def clear_prompts(self) -> bool:
        """Clear all prompts."""
        try:
            self.prompts = []
            logger.info("Cleared all prompts")
            return True
        except Exception as e:
            logger.error(f"Error clearing prompts: {e}")
            return False

class LLMProvider:
    """Base class for LLM providers."""

    def __init__(self, api_key: str = None):
        """Initialize provider with API key."""
        self.api_key = api_key

    def generate(self, prompt: str, system_prompt: str = "", parameters: Dict = None) -> Dict:
        """Generate text from the LLM."""
        raise NotImplementedError

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def generate(self, prompt: str, system_prompt: str = "", parameters: Dict = None) -> Dict:
        """Generate text using OpenAI API."""
        if not self.api_key:
            raise ValueError("OpenAI API key not set")

        # Set up retry strategy
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5)
        session.mount('https://', HTTPAdapter(max_retries=retries))

        # Default parameters
        default_params = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }

        # Update with user parameters
        if parameters:
            default_params.update(parameters)

        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": default_params.get("model", "gpt-3.5-turbo"),
                    "messages": messages,
                    "temperature": default_params["temperature"],
                    "max_tokens": default_params["max_tokens"],
                    "top_p": default_params["top_p"],
                    "frequency_penalty": default_params["frequency_penalty"],
                    "presence_penalty": default_params["presence_penalty"]
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return {
                "text": result["choices"][0]["message"]["content"],
                "model": result["model"],
                "usage": result["usage"]
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI API request failed: {e}")
            raise

    def get_available_models(self) -> List[str]:
        """Get list of available OpenAI models."""
        return [
            "gpt-4-turbo-preview",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k"
        ]

class AnthropicProvider(LLMProvider):
    """Anthropic API provider."""

    def generate(self, prompt: str, system_prompt: str = "", parameters: Dict = None) -> Dict:
        """Generate text using Anthropic API."""
        if not self.api_key:
            raise ValueError("Anthropic API key not set")

        # Set up retry strategy
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5)
        session.mount('https://', HTTPAdapter(max_retries=retries))

        # Default parameters
        default_params = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0
        }

        # Update with user parameters
        if parameters:
            default_params.update(parameters)

        try:
            response = session.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": default_params.get("model", "claude-3-opus-20240229"),
                    "messages": [{"role": "user", "content": prompt}],
                    "system": system_prompt,
                    "max_tokens": default_params["max_tokens"],
                    "temperature": default_params["temperature"],
                    "top_p": default_params["top_p"]
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return {
                "text": result["content"][0]["text"],
                "model": result["model"],
                "usage": {
                    "prompt_tokens": result["usage"]["input_tokens"],
                    "completion_tokens": result["usage"]["output_tokens"],
                    "total_tokens": result["usage"]["input_tokens"] + result["usage"]["output_tokens"]
                }
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Anthropic API request failed: {e}")
            raise

    def get_available_models(self) -> List[str]:
        """Get list of available Anthropic models."""
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]

class GoogleProvider(LLMProvider):
    """Google AI API provider."""

    def generate(self, prompt: str, system_prompt: str = "", parameters: Dict = None) -> Dict:
        """Generate text using Google AI API."""
        if not self.api_key:
            raise ValueError("Google AI API key not set")

        # Set up retry strategy
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5)
        session.mount('https://', HTTPAdapter(max_retries=retries))

        # Default parameters
        default_params = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0,
            "top_k": 40
        }

        # Update with user parameters
        if parameters:
            default_params.update(parameters)

        # Prepare prompt
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        try:
            response = session.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
                headers={
                    "x-goog-api-key": self.api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "contents": [{"parts": [{"text": full_prompt}]}],
                    "generationConfig": {
                        "temperature": default_params["temperature"],
                        "maxOutputTokens": default_params["max_tokens"],
                        "topP": default_params["top_p"],
                        "topK": default_params["top_k"]
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return {
                "text": result["candidates"][0]["content"]["parts"][0]["text"],
                "model": "gemini-pro",
                "usage": {
                    "prompt_tokens": len(full_prompt.split()),
                    "completion_tokens": len(result["candidates"][0]["content"]["parts"][0]["text"].split()),
                    "total_tokens": len(full_prompt.split()) + len(result["candidates"][0]["content"]["parts"][0]["text"].split())
                }
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Google AI API request failed: {e}")
            raise

    def get_available_models(self) -> List[str]:
        """Get list of available Google AI models."""
        return [
            "gemini-pro",
            "gemini-pro-vision"
        ]

class MistralProvider(LLMProvider):
    """Mistral AI API provider."""

    def generate(self, prompt: str, system_prompt: str = "", parameters: Dict = None) -> Dict:
        """Generate text using Mistral AI API."""
        if not self.api_key:
            raise ValueError("Mistral AI API key not set")

        # Set up retry strategy
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5)
        session.mount('https://', HTTPAdapter(max_retries=retries))

        # Default parameters
        default_params = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0
        }

        # Update with user parameters
        if parameters:
            default_params.update(parameters)

        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = session.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": default_params.get("model", "mistral-large-latest"),
                    "messages": messages,
                    "temperature": default_params["temperature"],
                    "max_tokens": default_params["max_tokens"],
                    "top_p": default_params["top_p"]
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return {
                "text": result["choices"][0]["message"]["content"],
                "model": result["model"],
                "usage": result["usage"]
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Mistral AI API request failed: {e}")
            raise

    def get_available_models(self) -> List[str]:
        """Get list of available Mistral AI models."""
        return [
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest"
        ]

class GrokProvider(LLMProvider):
    """Grok API provider."""

    def generate(self, prompt: str, system_prompt: str = "", parameters: Dict = None) -> Dict:
        """Generate text using Grok API."""
        if not self.api_key:
            raise ValueError("Grok API key not set")

        # Set up retry strategy
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5)
        session.mount('https://', HTTPAdapter(max_retries=retries))

        # Default parameters
        default_params = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0
        }

        # Update with user parameters
        if parameters:
            default_params.update(parameters)

        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = session.post(
                "https://api.grok.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": default_params.get("model", "grok-1"),
                    "messages": messages,
                    "temperature": default_params["temperature"],
                    "max_tokens": default_params["max_tokens"],
                    "top_p": default_params["top_p"]
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return {
                "text": result["choices"][0]["message"]["content"],
                "model": result["model"],
                "usage": result["usage"]
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Grok API request failed: {e}")
            raise

    def get_available_models(self) -> List[str]:
        """Get list of available Grok models."""
        return [
            "grok-1"
        ]

class DeepSeekProvider(LLMProvider):
    """DeepSeek AI API provider."""

    def generate(self, prompt: str, system_prompt: str = "", parameters: Dict = None) -> Dict:
        """Generate text using DeepSeek AI API."""
        if not self.api_key:
            raise ValueError("DeepSeek AI API key not set")

        # Set up retry strategy
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5)
        session.mount('https://', HTTPAdapter(max_retries=retries))

        # Default parameters
        default_params = {
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1.0
        }

        # Update with user parameters
        if parameters:
            default_params.update(parameters)

        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = session.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": default_params.get("model", "deepseek-chat"),
                    "messages": messages,
                    "temperature": default_params["temperature"],
                    "max_tokens": default_params["max_tokens"],
                    "top_p": default_params["top_p"]
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return {
                "text": result["choices"][0]["message"]["content"],
                "model": result["model"],
                "usage": result["usage"]
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"DeepSeek AI API request failed: {e}")
            raise

    def get_available_models(self) -> List[str]:
        """Get list of available DeepSeek AI models."""
        return [
            "deepseek-chat",
            "deepseek-coder"
        ]

class ProviderFactory:
    """Factory for creating LLM providers."""

    @staticmethod
    def create_provider(provider_name: str, api_key: str) -> Optional[LLMProvider]:
        """Create a provider instance based on the provider name."""
        providers = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "google": GoogleProvider,
            "mistral": MistralProvider,
            "grok": GrokProvider,
            "deepseek": DeepSeekProvider
        }

        provider_class = providers.get(provider_name.lower())
        if provider_class:
            return provider_class(api_key)
        else:
            logger.error(f"Unknown provider: {provider_name}")
            return None

class ExperimentManager:
    """Manages LLM experiments."""

    def __init__(self, project_folder: str):
        """Initialize experiment manager with project folder."""
        self.project_folder = Path(project_folder)
        self.project_folder.mkdir(parents=True, exist_ok=True)
        self.experiments_file = self.project_folder / "experiments.json"
        self.experiments = self._load_experiments()
        # Initialize API key manager
        self.api_key_manager = APIKeyManager(
            key_file=str(self.project_folder / "api_keys.json"),
            encryption_key_file=str(self.project_folder / "encryption_key.key")
        )
        logger.info(f"ExperimentManager initialized for project: {project_folder}")

    def _load_experiments(self) -> Dict:
        """Load experiments from JSON file."""
        if self.experiments_file.exists():
            try:
                with open(self.experiments_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading experiments: {e}")
                return {}
        return {}

    def _save_experiments(self):
        """Save experiments to JSON file."""
        try:
            with open(self.experiments_file, 'w') as f:
                json.dump(self.experiments, f, indent=2)
            logger.info(f"Saved {len(self.experiments)} experiments to {self.experiments_file}")
        except Exception as e:
            logger.error(f"Error saving experiments: {e}")
            raise

    def create_experiment(self, name: str, description: str = "") -> str:
        """Create a new experiment."""
        experiment_id = f"exp_{len(self.experiments) + 1:03d}"
        self.experiments[experiment_id] = {
            "name": name,
            "description": description,
            "created_at": datetime.datetime.now().isoformat(),
            "conditions": {},
            "results": {}
        }
        self._save_experiments()
        logger.info(f"Created new experiment: {name} (ID: {experiment_id})")
        return experiment_id

    def add_condition(self, experiment_id: str, condition_name: str,
                     provider: str, model: str, parameters: Dict = None) -> Optional[str]:
        """Add a condition to an experiment."""
        if experiment_id not in self.experiments:
            logger.error(f"Experiment not found: {experiment_id}")
            return None

        condition_id = f"cond_{len(self.experiments[experiment_id]['conditions']) + 1:03d}"
        self.experiments[experiment_id]['conditions'][condition_id] = {
            "name": condition_name,
            "provider": provider,
            "model": model,
            "parameters": parameters or {},
            "created_at": datetime.datetime.now().isoformat()
        }
        self._save_experiments()
        logger.info(f"Added condition '{condition_name}' to experiment {experiment_id}")
        return condition_id

    def run_experiment(self, experiment_id: str, prompt_ids: List[str]) -> Optional[Dict]:
        """Run an experiment with the given prompts."""
        if experiment_id not in self.experiments:
            logger.error(f"Experiment not found: {experiment_id}")
            return None

        experiment = self.experiments[experiment_id]
        results = {
            "experiment_id": experiment_id,
            "experiment_name": experiment["name"],
            "timestamp": datetime.datetime.now().isoformat(),
            "prompts": {},
            "conditions": {}
        }

        # Load prompts
        prompt_manager = PromptManager(str(self.project_folder))
        prompts = [prompt_manager.get_prompt(pid) for pid in prompt_ids if prompt_manager.get_prompt(pid)]

        if not prompts:
            logger.error("No valid prompts found")
            return None

        # Run each condition
        for condition_id, condition in experiment["conditions"].items():
            logger.info(f"Running condition: {condition['name']}")
            # Get API key for the provider
            api_key = self.api_key_manager.get_key(condition["provider"])
            if not api_key:
                logger.error(f"No API key found for provider: {condition['provider']}")
                continue
                
            provider = ProviderFactory.create_provider(condition["provider"], api_key)
            if not provider:
                logger.error(f"Failed to create provider for condition: {condition['name']}")
                continue

            condition_results = []
            for prompt in prompts:
                try:
                    response = provider.generate(
                        prompt["prompt_text"],
                        prompt.get("system_prompt", ""),
                        condition["parameters"]
                    )
                    condition_results.append({
                        "prompt_id": prompt["id"],
                        "response": response["text"],
                        "model": response["model"],
                        "usage": response["usage"]
                    })
                except Exception as e:
                    logger.error(f"Error running prompt {prompt['id']} for condition {condition['name']}: {e}")
                    condition_results.append({
                        "prompt_id": prompt["id"],
                        "error": str(e)
                    })

            results["conditions"][condition_id] = {
                "name": condition["name"],
                "provider": condition["provider"],
                "model": condition["model"],
                "parameters": condition["parameters"],
                "results": condition_results
            }

        # Save results
        experiment["results"][datetime.datetime.now().isoformat()] = results
        self._save_experiments()
        logger.info(f"Completed experiment: {experiment['name']}")

        return results

    def export_results_csv(self, results: Dict) -> str:
        """Export experiment results to CSV."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.project_folder / f"results_{timestamp}.csv"

        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow([
                    "Experiment ID",
                    "Experiment Name",
                    "Timestamp",
                    "Prompt ID",
                    "Condition Name",
                    "Provider",
                    "Model",
                    "Response",
                    "Error",
                    "Usage"
                ])

                # Write data
                for condition_id, condition in results["conditions"].items():
                    for result in condition["results"]:
                        writer.writerow([
                            results["experiment_id"],
                            results["experiment_name"],
                            results["timestamp"],
                            result["prompt_id"],
                            condition["name"],
                            condition["provider"],
                            condition["model"],
                            result.get("response", ""),
                            result.get("error", ""),
                            json.dumps(result.get("usage", {}))
                        ])

            logger.info(f"Exported results to {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            raise

    def export_blinded_csv(self, results: Dict) -> str:
        """Export experiment results to CSV with blinded condition names."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.project_folder / f"blinded_results_{timestamp}.csv"

        try:
            # Create mapping of condition names to blinded names
            condition_names = [c["name"] for c in results["conditions"].values()]
            blinded_names = [f"Condition_{i+1}" for i in range(len(condition_names))]
            name_mapping = dict(zip(condition_names, blinded_names))

            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow([
                    "Experiment ID",
                    "Experiment Name",
                    "Timestamp",
                    "Prompt ID",
                    "Blinded Condition",
                    "Response",
                    "Error",
                    "Usage"
                ])

                # Write data
                for condition_id, condition in results["conditions"].items():
                    for result in condition["results"]:
                        writer.writerow([
                            results["experiment_id"],
                            results["experiment_name"],
                            results["timestamp"],
                            result["prompt_id"],
                            name_mapping[condition["name"]],
                            result.get("response", ""),
                            result.get("error", ""),
                            json.dumps(result.get("usage", {}))
                        ])

            # Save mapping for reference
            mapping_file = self.project_folder / f"blinded_mapping_{timestamp}.json"
            with open(mapping_file, 'w') as f:
                json.dump(name_mapping, f, indent=2)

            logger.info(f"Exported blinded results to {output_file}")
            return str(output_file)
        except Exception as e:
            logger.error(f"Error exporting blinded results: {e}")
            raise

    def list_experiments(self) -> List[Dict]:
        """List all experiments."""
        return [
            {"id": k, **v}
            for k, v in self.experiments.items()
        ]

    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get a specific experiment by ID."""
        if experiment_id in self.experiments:
            return {"id": experiment_id, **self.experiments[experiment_id]}
        return None

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment."""
        if experiment_id in self.experiments:
            del self.experiments[experiment_id]
            self._save_experiments()
            logger.info(f"Deleted experiment: {experiment_id}")
            return True
        return False

    def create_experiment_with_conditions(self, name: str, description: str = "", conditions: List[Dict] = None) -> str:
        """Create a new experiment with predefined conditions."""
        experiment_id = self.create_experiment(name, description)
        if conditions:
            for condition in conditions:
                self.add_condition(
                    experiment_id,
                    condition["name"],
                    condition["provider"],
                    condition["model"],
                    condition.get("parameters")
                )
        return experiment_id 