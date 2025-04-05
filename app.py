# -*- coding: utf-8 -*-
"""
Gradio application to generate English learning podcast dialogues using AI.

This script utilizes text generation (OpenAI GPT or Google Gemini) and
text-to-speech (OpenAI TTS or Google Cloud TTS) APIs to create dialogue scripts
and corresponding audio files based on user-provided topics and CEFR levels.
Configuration is managed through environment variables loaded from a .env file.
"""

from __future__ import annotations
import gradio as gr
import io
import os
import uuid
import time
import traceback
import datetime
import json
from dotenv import load_dotenv
from typing import Optional, Dict, Any, Tuple, List, Union, Type

# --- Provider Specific Imports with Graceful Handling ---
try:
    from openai import OpenAI, APIError
except ImportError as e:
    print("Warning: OpenAI library not installed: {e}. OpenAI features will be unavailable.")
    OpenAI = None # type: ignore
    APIError = None # type: ignore

try:
    import google.generativeai as genai
    from google.cloud import texttospeech
    from google.api_core import exceptions as google_exceptions
except ImportError:
    print("Warning: Google libraries (google-generativeai, google-cloud-texttospeech) not installed. Google features will be unavailable.")
    genai = None # type: ignore
    texttospeech = None # type: ignore
    google_exceptions = None # type: ignore
# --- End Provider Specific Imports ---

# --- Constants ---
PROVIDER_OPENAI = "openai"
PROVIDER_GOOGLE = "google"

DEFAULT_GEMINI_MODEL = "gemini-1.5-flash-latest"
DEFAULT_SPEAKER_1_NAME = "Speaker 1"
DEFAULT_SPEAKER_2_NAME = "Speaker 2"

# Define temporary and log directories relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(SCRIPT_DIR, "gradio_cached_examples", "tmp")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")

# Pricing Data (Note: These prices can change. Update periodically.)
# Prices are per 1000 units (tokens or characters) in USD.
MODEL_PRICES = {
    "gpt-4o": {"input": 0.005 / 1000, "output": 0.015 / 1000},
    "gemini-1.5-flash-latest": {"input": 0.000125 / 1000, "output": 0.000375 / 1000}, # Example, confirm official pricing
    "gemini-1.5-pro-latest": {"input": 0.00125 / 1000, "output": 0.00375 / 1000}, # Example, confirm official pricing
    "gemini-pro": {"input": 0.000125 / 1000, "output": 0.000375 / 1000}, # Example, confirm official pricing
    "gemini-2.0-flash": {"input": 0.0001 / 1000, "output": 0.0004 / 1000}, # Example, confirm official pricing
}
TTS_PRICES = {
    "tts-1": 0.015 / 1000,
    "tts-1-hd": 0.030 / 1000,
    "google-tts-standard": 0.004 / 1000,
    "google-tts-wavenet": 0.016 / 1000, # Check Google official pricing for Wavenet/Studio/etc.
    "google-tts-neural2": 0.016 / 1000, # Example for Neural2 voices
    "google-tts-chirp3-hd": 0.030 / 1000, 
}

# --- Configuration ---
class AppConfig:
    """
    Holds application configuration loaded from environment variables (.env file).

    Attributes:
        text_provider (str): The provider for text generation ('openai' or 'google').
        tts_provider (str): The provider for text-to-speech ('openai' or 'google').
        text_model (Optional[str]): The specific model for text generation (e.g., 'gpt-4o', 'gemini-1.5-pro-latest').
        audio_model (Optional[str]): The specific model for OpenAI TTS (e.g., 'tts-1', 'tts-1-hd'). Not used for Google TTS directly.
        speaker_1_voice (Optional[str]): Voice identifier for speaker 1 (provider-specific).
        speaker_2_voice (Optional[str]): Voice identifier for speaker 2 (provider-specific).
        speaker_1_name (str): Display name for speaker 1.
        speaker_2_name (str): Display name for speaker 2.
        openai_api_key (Optional[str]): API key for OpenAI.
        gemini_api_key (Optional[str]): API key for Google Gemini. Falls back to GOOGLE_API_KEY if not set.
        # Note: Google Cloud TTS typically uses Application Default Credentials (ADC)
        # or the GOOGLE_APPLICATION_CREDENTIALS environment variable.
    """
    def __init__(self):
        """Loads configuration from environment variables."""
        load_dotenv(override=True)
        print("Loading application configuration from environment variables...")
        self.text_provider: str = os.getenv("TEXT_PROVIDER", PROVIDER_OPENAI).lower()
        self.tts_provider: str = os.getenv("TTS_PROVIDER", PROVIDER_OPENAI).lower()
        self.text_model: Optional[str] = os.getenv("TEXT_MODEL")
        self.audio_model: Optional[str] = os.getenv("AUDIO_MODEL") # Used for OpenAI TTS model selection
        self.speaker_1_voice: Optional[str] = os.getenv("SPEAKER_1_VOICE")
        self.speaker_2_voice: Optional[str] = os.getenv("SPEAKER_2_VOICE")
        self.speaker_1_name: str = os.getenv("SPEAKER_1_NAME", DEFAULT_SPEAKER_1_NAME)
        self.speaker_2_name: str = os.getenv("SPEAKER_2_NAME", DEFAULT_SPEAKER_2_NAME)
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        # Set default text model if using Google and none is specified
        if self.text_provider == PROVIDER_GOOGLE and not self.text_model:
            self.text_model = DEFAULT_GEMINI_MODEL
            print(f"No Google text model specified (TEXT_MODEL), using default: {self.text_model}")
        # Add a default OpenAI model if desired, or leave as None to require explicit setting
        # elif self.text_provider == PROVIDER_OPENAI and not self.text_model:
        #     self.text_model = "gpt-4o" # Example default
        #     print(f"No OpenAI text model specified (TEXT_MODEL), using default: {self.text_model}")

        print(f"Configuration Loaded:")
        print(f"  Text Provider: {self.text_provider}")
        print(f"  TTS Provider: {self.tts_provider}")
        print(f"  Text Model: {self.text_model or 'Not Set'}")
        print(f"  Audio Model (OpenAI TTS): {self.audio_model or 'Not Set'}")
        print(f"  Speaker 1 Voice: {self.speaker_1_voice or 'Not Set'}")
        print(f"  Speaker 2 Voice: {self.speaker_2_voice or 'Not Set'}")
        print("-" * 20)


# --- Cost Tracking ---
class CostTracker:
    """
    Tracks estimated API costs for a single generation request.

    Attributes:
        text_generation_cost (float): Accumulated cost for text generation APIs.
        audio_generation_cost (float): Accumulated cost for text-to-speech APIs.
        details (List[Dict[str, Any]]): A list containing details of each cost-incurring API call.
    """
    def __init__(self):
        """Initializes the cost tracker with zero costs."""
        self.text_generation_cost: float = 0.0
        self.audio_generation_cost: float = 0.0
        self.details: List[Dict[str, Any]] = []

    def add_cost(self, cost_type: str, provider: str, cost: float, details: Dict[str, Any]):
        """
        Adds a cost entry to the tracker.

        Args:
            cost_type (str): Type of cost ('text_generation' or 'audio_generation').
            provider (str): The API provider ('openai' or 'google').
            cost (float): The calculated cost for this specific call.
            details (Dict[str, Any]): Dictionary containing specifics of the API call
                                       (e.g., model, tokens, characters, tier).
        """
        if cost_type == "text_generation":
            self.text_generation_cost += cost
        elif cost_type == "audio_generation":
            self.audio_generation_cost += cost
        else:
            print(f"Warning: Unknown cost type '{cost_type}' encountered.")
            return

        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": cost_type,
            "provider": provider,
            "cost_usd": f"{cost:.8f}", # Store as string for precision if needed later
            **details
        }
        self.details.append(entry)
        # Minimal console log for cost addition
        unit = details.get('output_tokens', details.get('character_count', 'N/A'))
        unit_name = 'tokens' if 'tokens' in details else 'chars'
        print(f"Cost Added: {provider}/{cost_type} (${cost:.6f}) - Model/Voice: {details.get('model', details.get('voice', 'N/A'))}, Units({unit_name}): {unit}")


    @property
    def total_cost(self) -> float:
        """Calculates the total estimated cost in USD."""
        return self.text_generation_cost + self.audio_generation_cost

    def get_summary(self, text_provider: str, tts_provider: str, text_model: Optional[str],
                     s1_voice: Optional[str], s2_voice: Optional[str], jpy_rate: int = 150) -> str:
        """
        Generates a formatted string summarizing the estimated costs.

        Args:
            text_provider (str): The text generation provider used.
            tts_provider (str): The TTS provider used.
            text_model (Optional[str]): The text generation model used.
            s1_voice (Optional[str]): The voice used for speaker 1.
            s2_voice (Optional[str]): The voice used for speaker 2.
            jpy_rate (int): The approximate USD to JPY exchange rate for display.

        Returns:
            str: A multi-line string containing the cost summary.
        """
        total_tts_chars = sum(d.get('character_count', 0) for d in self.details if d['type'] == 'audio_generation')
        total_input_tokens = sum(d.get('input_tokens', 0) for d in self.details if d['type'] == 'text_generation')
        total_output_tokens = sum(d.get('output_tokens', 0) for d in self.details if d['type'] == 'text_generation')
        jpy_cost = self.total_cost * jpy_rate

        # Generate cost details string
        details_str_list = []
        for i, entry in enumerate(self.details):
            cost_usd = float(entry['cost_usd'])
            details_str_list.append(
                f"  Call {i+1}: {entry['provider']} {entry['type']} (${cost_usd:.6f}) "
                f"- Details: { {k: v for k, v in entry.items() if k not in ['timestamp', 'type', 'provider', 'cost_usd']} }"
            )
        
        return f"""
        ==== API使用料 ====
        テキスト生成 ({text_provider.upper()}): ${self.text_generation_cost:.6f} (In: {total_input_tokens:,} tokens, Out: {total_output_tokens:,} tokens)
        音声合成 ({tts_provider.upper()}): ${self.audio_generation_cost:.6f} ({total_tts_chars:,} characters)
        ---------------------------------
        合計: ${self.total_cost:.6f}
        概算: ¥{jpy_cost:,.3f} (at {jpy_rate} JPY/USD)
        =================================
        使用した設定:
          テキスト生成モデル: {text_model or 'N/A'}
          スピーカー1の音声: {s1_voice or 'N/A'}
          スピーカー2の音声: {s2_voice or 'N/A'}
        ---------------------------------
        詳細 ({len(self.details)} API call(s)):
        """

# --- Provider Handlers ---
class BaseProviderHandler:
    """Base class for API provider handlers."""
    def __init__(self, config: AppConfig):
        """
        Initializes the handler with application configuration.

        Args:
            config (AppConfig): The application configuration object.
        """
        self.config = config

    def _calculate_cost(self, cost_tracker: CostTracker, cost_type: str, provider: str, cost: float, details: Dict[str, Any]):
        """Helper method to add cost entries via the CostTracker."""
        cost_tracker.add_cost(cost_type, provider, cost, details)


class OpenAIHandler(BaseProviderHandler):
    """Handles interactions with OpenAI APIs (Chat Completion and TTS)."""
    def __init__(self, config: AppConfig):
        """
        Initializes the OpenAI client if the API key is provided and the library is installed.

        Args:
            config (AppConfig): The application configuration object.
        """
        super().__init__(config)
        self.client: Optional[OpenAI] = None
        if OpenAI and config.openai_api_key:
            try:
                self.client = OpenAI(api_key=config.openai_api_key)
                print("OpenAI Client initialized successfully.")
            except Exception as e:
                print(f"Error initializing OpenAI Client: {e}. OpenAI features may fail.")
        elif config.text_provider == PROVIDER_OPENAI or config.tts_provider == PROVIDER_OPENAI:
            print("Warning: OpenAI provider selected, but API key (OPENAI_API_KEY) is missing or 'openai' library not installed.")

    def is_available(self) -> bool:
        """Checks if the OpenAI client was initialized successfully."""
        return self.client is not None

    def generate_text(self, prompt: str, max_tokens: int, cost_tracker: CostTracker) -> str:
        """
        Generates text using the configured OpenAI chat model.

        Args:
            prompt (str): The prompt to send to the model.
            max_tokens (int): The maximum number of tokens to generate in the response.
            cost_tracker (CostTracker): The cost tracker instance for this request.

        Returns:
            str: The generated text content.

        Raises:
            RuntimeError: If the OpenAI client is not available.
            ValueError: If the text model is not configured.
            APIError: If the OpenAI API returns an error.
            Exception: For other unexpected errors during generation.
        """
        if self.text_provider not in [PROVIDER_OPENAI, PROVIDER_GOOGLE]:
            raise ValueError(f"Invalid text provider: {self.text_provider}")
        if self.tts_provider not in [PROVIDER_OPENAI, PROVIDER_GOOGLE]:
            raise ValueError(f"Invalid TTS provider: {self.tts_provider}")
        if not self.is_available():
            raise RuntimeError("OpenAI client is not available. Check API key and installation.")
        if not self.config.text_model:
            raise ValueError("OpenAI text model (TEXT_MODEL) is not configured in environment variables.")
        if not self.client: # Redundant check, but satisfies type checker
             raise RuntimeError("OpenAI client is None despite is_available() being true.")

        try:
            print(f"Sending request to OpenAI model: {self.config.text_model}")
            response = self.client.chat.completions.create(
                model=self.config.text_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7, # Consider making this configurable
            )
            dialogue_content = response.choices[0].message.content

            # Cost Calculation
            cost = 0.0
            cost_details = {"model": self.config.text_model}
            if response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                prices = MODEL_PRICES.get(self.config.text_model)
                if prices:
                    input_cost = input_tokens * prices["input"]
                    output_cost = output_tokens * prices["output"]
                    cost = input_cost + output_cost
                    cost_details.update({
                        "input_tokens": input_tokens, "output_tokens": output_tokens,
                        "input_cost": f"{input_cost:.8f}", "output_cost": f"{output_cost:.8f}"
                    })
                else:
                    print(f"Warning: Pricing not found for OpenAI model '{self.config.text_model}'. Cost calculation may be inaccurate.")
                self._calculate_cost(cost_tracker, "text_generation", PROVIDER_OPENAI, cost, cost_details)
            else:
                print("Warning: OpenAI response did not include usage data. Cost calculation skipped for this call.")

            return dialogue_content.strip() if dialogue_content else ""

        except APIError as e:
            print(f"OpenAI API Error (Text Generation): Status={e.status_code}, Message={e.message}")
            raise # Re-raise to be handled by the main wrapper
        except Exception as e:
            print(f"Unexpected error during OpenAI text generation: {e}")
            traceback.print_exc()
            raise

    def synthesize_speech(self, text: str, voice: str, cost_tracker: CostTracker) -> Optional[bytes]:
        """
        Synthesizes speech from text using the configured OpenAI TTS model and voice.

        Args:
            text (str): The text to synthesize.
            voice (str): The voice identifier (e.g., 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer').
            cost_tracker (CostTracker): The cost tracker instance for this request.

        Returns:
            Optional[bytes]: The synthesized audio data in MP3 format, or None if synthesis failed.

        Raises:
            RuntimeError: If the OpenAI client is not available.
            ValueError: If the audio model or voice is not configured.
            APIError: If the OpenAI API returns an error.
            Exception: For other unexpected errors during synthesis.
        """
        if not self.is_available():
            raise RuntimeError("OpenAI client is not available. Check API key and installation.")
        if not self.config.audio_model:
            raise ValueError("OpenAI audio model (AUDIO_MODEL, e.g., 'tts-1') is not configured in environment variables.")
        if not voice:
            raise ValueError("OpenAI TTS requires a voice identifier (SPEAKER_1_VOICE/SPEAKER_2_VOICE).")
        if not self.client: # Redundant check
             raise RuntimeError("OpenAI client is None despite is_available() being true.")

        # Pre-calculate cost based on character count
        cost = 0.0
        char_count = len(text)
        price = TTS_PRICES.get(self.config.audio_model)
        cost_details = {
            "model": self.config.audio_model,
            "voice": voice,
            "character_count": char_count
        }
        if price:
            cost = char_count * price
            cost_details["cost_per_1k_chars"] = f"{price * 1000:.4f}"
        else:
            print(f"Warning: Pricing not found for OpenAI TTS model '{self.config.audio_model}'. Cost calculation may be inaccurate.")
        self._calculate_cost(cost_tracker, "audio_generation", PROVIDER_OPENAI, cost, cost_details)

        try:
            print(f"Sending request to OpenAI TTS: Model={self.config.audio_model}, Voice={voice}")
            # Use streaming response to potentially handle large audio files better
            with self.client.audio.speech.with_streaming_response.create(
                model=self.config.audio_model,
                voice=voice, # type: ignore - SDK expects Literal, but we use str
                input=text,
                response_format="mp3" # Currently fixed to mp3
            ) as response:
                # Consider checking response.status_code if available in your SDK version
                audio_bytes = response.read() # Read the entire stream into bytes

            if not audio_bytes:
                print(f"Warning: OpenAI TTS returned empty audio data for voice '{voice}'.")
                return None # Indicate failure clearly
            return audio_bytes

        except APIError as e:
            print(f"OpenAI TTS API Error: Status={e.status_code}, Message={e.message}")
            # Consider specific handling for common errors like invalid voice
            if "voice" in str(e.message).lower():
                print(f"Hint: Check if '{voice}' is a valid OpenAI TTS voice name.")
            raise
        except Exception as e:
            print(f"Unexpected error during OpenAI speech synthesis: {e}")
            traceback.print_exc()
            raise


class GoogleHandler(BaseProviderHandler):
    """Handles interactions with Google APIs (Gemini for text, Cloud TTS for speech)."""
    def __init__(self, config: AppConfig):
        """
        Initializes Google clients (Gemini and Cloud TTS) if libraries are installed
        and credentials/API keys are available.

        Args:
            config (AppConfig): The application configuration object.
        """
        super().__init__(config)
        # Type hint requires checking if 'genai' was imported successfully
        self.gemini_model: Optional[genai.GenerativeModel] = None if genai else None
        # Type hint requires checking if 'texttospeech' was imported successfully
        self.tts_client: Optional[texttospeech.TextToSpeechClient] = None if texttospeech else None
        self._initialize_clients()

    def _initialize_clients(self):
        """Initializes the Gemini and TTS clients based on configuration."""
        # Initialize Gemini (Generative AI) Client
        if genai and self.config.gemini_api_key:
            if not self.config.text_model:
                print(f"Warning: Google provider selected, but TEXT_MODEL not set. Using default: {DEFAULT_GEMINI_MODEL}.")
                self.config.text_model = DEFAULT_GEMINI_MODEL

            try:
                # Configure the API key (standard way)
                genai.configure(api_key=self.config.gemini_api_key)

                # Create the GenerativeModel instance
                # Consider making safety settings configurable if needed
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ]
                self.gemini_model = genai.GenerativeModel(
                    self.config.text_model,
                    safety_settings=safety_settings
                )
                print(f"Google Gemini model '{self.config.text_model}' initialized successfully.")
            except Exception as e:
                print(f"Error initializing Google Gemini model '{self.config.text_model}': {e}. Google text generation may fail.")
                self.gemini_model = None # Ensure it's None on failure
        elif self.config.text_provider == PROVIDER_GOOGLE:
            print("Warning: Google text provider selected, but API key (GEMINI_API_KEY/GOOGLE_API_KEY) is missing or 'google-generativeai' library not installed.")

        # Initialize Google Cloud Text-to-Speech Client
        if texttospeech:
            try:
                # The client automatically finds credentials (ADC or GOOGLE_APPLICATION_CREDENTIALS)
                self.tts_client = texttospeech.TextToSpeechClient()
                print("Google Cloud TTS Client initialized successfully.")
            except Exception as e:
                print(f"Error initializing Google Cloud TTS Client: {e}. Ensure credentials (ADC or GOOGLE_APPLICATION_CREDENTIALS) are set correctly. Google TTS features may fail.")
                self.tts_client = None # Ensure it's None on failure
        elif self.config.tts_provider == PROVIDER_GOOGLE:
            print("Warning: Google TTS provider selected, but 'google-cloud-texttospeech' library not installed or client initialization failed.")


    def is_text_available(self) -> bool:
        """Checks if the Gemini model client was initialized successfully."""
        return self.gemini_model is not None

    def is_tts_available(self) -> bool:
        """Checks if the Google Cloud TTS client was initialized successfully."""
        return self.tts_client is not None

    @staticmethod
    def _get_google_tts_tier(voice_name: Optional[str]) -> str:
        """
        Determines the likely Google TTS pricing tier based on the voice name.
        This is an estimation and might need adjustment based on actual voice types used.

        Args:
            voice_name (Optional[str]): The name of the Google TTS voice.

        Returns:
            str: The estimated pricing tier key (e.g., 'google-tts-wavenet').
        """
        if not voice_name:
            return "google-tts-standard" # Default if no name provided

        name_lower = voice_name.lower()
        if "wavenet" in name_lower:
            return "google-tts-wavenet"
        if "neural2" in name_lower: # Example for another tier
             return "google-tts-neural2"
        # Add checks for other tiers like 'Studio' if applicable
        # if "studio" in name_lower:
        #     return "google-tts-studio"

        # Default to standard if no specific tier keyword is found
        return "google-tts-standard"

    def generate_text(self, prompt: str, max_words: int, cost_tracker: CostTracker) -> str:
        """
        Generates text using the configured Google Gemini model.

        Args:
            prompt (str): The prompt to send to the model.
            max_words (int): Target word count (used to estimate max output tokens).
            cost_tracker (CostTracker): The cost tracker instance for this request.

        Returns:
            str: The generated text content.

        Raises:
            RuntimeError: If the Gemini model client is not available.
            ValueError: If the generation is blocked by safety filters or fails unexpectedly.
            google_exceptions.GoogleAPICallError: If the Google API returns an error.
            Exception: For other unexpected errors during generation.
        """
        if not self.is_text_available():
             raise RuntimeError(f"Google Gemini model '{self.config.text_model}' is not available. Check API key and installation.")
        if not self.gemini_model: # Redundant check
            raise RuntimeError("Gemini model is None.")

        try:
            # Estimate max tokens needed. This is a rough guess; adjust multiplier if needed.
            # Gemini API might have its own internal limits as well.
            max_output_tokens = max_words * 4 # More generous estimate
            generation_config = genai.types.GenerationConfig( # Use typed config
                max_output_tokens=max_output_tokens,
                temperature=0.7, # Consider making configurable
            )

            print(f"Sending request to Google Gemini model: {self.config.text_model}")
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=generation_config
            )
            print(f"Received response from Google Gemini.")

            dialogue_content = ""
            # Safely extract text content from the response object
            # Handles different potential response structures gracefully.
            try:
                if hasattr(response, 'text'):
                    dialogue_content = response.text
                elif hasattr(response, 'parts') and response.parts:
                    dialogue_content = "".join(part.text for part in response.parts if hasattr(part, 'text'))
                elif hasattr(response, 'candidates') and response.candidates:
                    first_candidate = response.candidates[0]
                    if hasattr(first_candidate, 'content') and hasattr(first_candidate.content, 'parts'):
                        dialogue_content = "".join(part.text for part in first_candidate.content.parts if hasattr(part, 'text'))
                    # Add more fallbacks if needed based on observed response structures
            except (AttributeError, IndexError, TypeError) as e:
                 print(f"Warning: Could not extract text from Gemini response structure: {e}. Response dump: {response}")
                 # Keep dialogue_content as empty string

            # Cost Calculation: Gemini pricing is typically token-based.
            # If usage metadata with tokens is available, use it. Otherwise, fallback to character estimation.
            cost = 0.0
            cost_details = {"model": self.config.text_model}
            prices = MODEL_PRICES.get(self.config.text_model)

            # Check for token usage metadata (adapt field names based on actual API response)
            input_tokens, output_tokens = 0, 0
            if hasattr(response, 'usage_metadata'):
                input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) # Or 'completion_token_count'
                cost_details["token_source"] = "metadata"
            else:
                cost_details["token_source"] = "estimated_from_chars"
                # Fallback estimation (very rough)
                input_tokens = len(prompt) // 4 # Guessing 4 chars per token
                output_tokens = len(dialogue_content) // 4
                print("Warning: Gemini response missing usage_metadata. Estimating tokens from character count for cost calculation.")


            if prices and (input_tokens > 0 or output_tokens > 0):
                input_cost = input_tokens * prices["input"]
                output_cost = output_tokens * prices["output"]
                cost = input_cost + output_cost
                cost_details.update({
                    "input_tokens": input_tokens, "output_tokens": output_tokens,
                    "input_cost": f"{input_cost:.8f}", "output_cost": f"{output_cost:.8f}"
                })
            elif not prices:
                print(f"Warning: Pricing not found for Google model '{self.config.text_model}'. Cost calculation may be inaccurate.")

            self._calculate_cost(cost_tracker, "text_generation", PROVIDER_GOOGLE, cost, cost_details)

            # Safety Checks (Important for Gemini)
            # Check if the prompt itself was blocked
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                reason = response.prompt_feedback.block_reason.name # Get the reason name
                error_msg = f"Prompt blocked by Google safety filters. Reason: {reason}"
                print(f"Error: {error_msg}")
                raise ValueError(error_msg)

            # Check if the response content was blocked or generation stopped for safety/other reasons
            if not dialogue_content and hasattr(response, 'candidates') and response.candidates:
                 finish_reason = getattr(response.candidates[0], 'finish_reason', None)
                 if finish_reason and finish_reason.name != 'STOP':
                     reason_name = finish_reason.name
                     error_msg = f"Google Gemini generation finished unexpectedly. Reason: {reason_name}"
                     print(f"Warning: {error_msg}")
                     # Raise an error if blocked, otherwise just warn (e.g., MAX_TOKENS)
                     if reason_name in ['SAFETY', 'RECITATION']: # Consider 'OTHER' as potential error too
                         raise ValueError(error_msg)

            return dialogue_content.strip() if dialogue_content else ""

        except google_exceptions.GoogleAPICallError as e:
            print(f"Google API Call Error (Text Generation): {e}")
            # Provide specific feedback for common issues like quota limits
            if isinstance(e, google_exceptions.ResourceExhausted):
                print("Hint: Google API quota likely exceeded. Check your usage limits in Google Cloud Console or AI Studio.")
            elif isinstance(e, google_exceptions.PermissionDenied):
                 print("Hint: Permission denied. Check if the API key is valid and has the correct permissions for the Gemini API.")
            raise
        except ValueError as ve: # Catch safety filter blocks raised above
            raise ve
        except Exception as e:
            print(f"Unexpected error during Google Gemini text generation: {e}")
            traceback.print_exc()
            raise

    def synthesize_speech(self, text: str, voice_name: str, cost_tracker: CostTracker) -> Optional[bytes]:
        """
        Synthesizes speech from text using Google Cloud Text-to-Speech.

        Args:
            text (str): The text to synthesize.
            voice_name (str): The Google Cloud TTS voice name (e.g., 'en-US-Wavenet-D').
            cost_tracker (CostTracker): The cost tracker instance for this request.

        Returns:
            Optional[bytes]: The synthesized audio data in MP3 format, or None if synthesis failed.

        Raises:
            RuntimeError: If the Google Cloud TTS client is not available.
            ValueError: If the voice name is not provided.
            google_exceptions.GoogleAPICallError: If the Google API returns an error.
            Exception: For other unexpected errors during synthesis.
        """
        if not self.is_tts_available():
            raise RuntimeError("Google Cloud TTS client is not available. Check credentials and installation.")
        if not voice_name:
            raise ValueError("Google TTS requires a voice name (SPEAKER_1_VOICE/SPEAKER_2_VOICE).")
        if not self.tts_client: # Redundant check
             raise RuntimeError("TTS Client is None.")
        if not texttospeech: # Redundant check
            raise RuntimeError("texttospeech module not loaded.")


        # Pre-calculate cost based on character count and estimated tier
        cost = 0.0
        char_count = len(text)
        # Determine language code from voice name if possible (e.g., 'en-US')
        # Defaulting to en-US, make this more robust or configurable if needed.
        language_code = "-".join(voice_name.split('-')[:2]) if '-' in voice_name else "en-US"

        tts_tier = self._get_google_tts_tier(voice_name)
        price = TTS_PRICES.get(tts_tier)
        cost_details = {
            "model": "google-cloud-tts", # Generic identifier for the service
            "voice": voice_name,
            "character_count": char_count,
            "language_code": language_code,
            "estimated_tier": tts_tier
        }
        if price:
            cost = char_count * price
            cost_details["cost_per_1k_chars"] = f"{price * 1000:.4f}"
        else:
            print(f"Warning: Pricing not found for estimated Google TTS tier '{tts_tier}' (voice: {voice_name}). Cost calculation may be inaccurate.")
        self._calculate_cost(cost_tracker, "audio_generation", PROVIDER_GOOGLE, cost, cost_details)

        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3 # Make configurable if other formats are needed
            )

            print(f"Sending request to Google Cloud TTS: Voice={voice_name}, Lang={language_code}")
            response = self.tts_client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            if not response.audio_content:
                print(f"Warning: Google TTS returned empty audio data for voice '{voice_name}'.")
                return None # Indicate failure clearly
            return response.audio_content

        except google_exceptions.GoogleAPICallError as e:
            print(f"Google Cloud TTS API Call Error: {e}")
            # Provide hints for common errors
            if isinstance(e, google_exceptions.InvalidArgument):
                print(f"Hint: Invalid argument. Check if the voice name '{voice_name}' and language code '{language_code}' are valid and compatible for Google Cloud TTS.")
            elif isinstance(e, google_exceptions.PermissionDenied):
                 print("Hint: Permission denied for Google Cloud TTS. Check your Application Default Credentials (ADC) or service account permissions.")
            raise
        except Exception as e:
            print(f"Unexpected error during Google speech synthesis: {e}")
            traceback.print_exc()
            raise

# --- Core Logic ---

# Note on Prompt Engineering:
# The prompt below is crafted for a specific style and structure.
# Achieving the desired output, especially regarding word count and natural flow,
# often requires significant tuning and experimentation depending on the model used.
# Models might interpret instructions like "STRICTLY {word_count} words" differently.
DIALOGUE_PROMPT_TEMPLATE = """
Generate an engaging yet comfortably paced podcast dialogue for English language learners at CEFR level {cefr_level}.
Topic: '{topic}'.{additional_info}

Style: Sound like a clear, informative, and welcoming educational podcast episode. Maintain a natural, conversational flow between the hosts, similar to a relaxed NPR segment, but with a clear sense of podcast structure and progression. Use accessible language for the CEFR level. The tone should be friendly and approachable for the listener.
Speakers: Clearly identify turns for "{speaker1_name}" and "{speaker2_name}". Alternate turns naturally. Ensure a balanced conversation.
Format: Use EXACTLY this format without deviation:
{speaker1_name}: [Dialogue text for speaker 1]
{speaker2_name}: [Dialogue text for speaker 2]
{speaker1_name}: [Dialogue text for speaker 1]
...etc.

Content Requirements:
- Total Length: Aim for approximately {word_count} words total (across all dialogue text). This is a target, focus on natural conversation length around this number.
- Structure:
    - Clear Podcast Intro: Welcome listener, introduce hosts briefly, state the episode topic clearly.
    - Structured Body: Discuss main points logically. Use transition phrases (e.g., "Okay, so moving on...", "That brings us to...").
    - Clear Podcast Outro: Summarize key takeaways conversationally, thank the listener, maybe a brief call to action or teaser.
- Educational Elements: Define complex terms simply within the conversation. Frame facts as discussion points.
- Engagement: Use relatable examples or brief anecdotes. Keep it interesting.
- Conversational Flow & Podcast Feel: Incorporate natural spoken English features sparingly:
    - Occasional fillers ('well,' 'you know,' 'so').
    - Natural pauses indicated by (...).
    - Agreement/transition phrases ('Right,' 'That's interesting,' 'So, what about...?').
    - **Rare** natural repetitions for emphasis or hesitation (e.g., "It's a... a really key point").
- Technical Constraints:
    - Output ONLY the dialogue in the specified format. No introductory text, explanations, titles, or summaries before or after the dialogue block.
    - Do NOT use markdown (like * or _) within the dialogue text itself.
    - Do NOT include quotation marks (" ") around the dialogue text. Write plain text after the speaker name and colon.
    - Ensure the vocabulary and grammar strictly match the {cefr_level} level.

Primary Goal: Create an authentic-sounding, well-structured podcast episode suitable for {cefr_level} English learners, hitting close to the {word_count} word target naturally. It must sound like a real, educational podcast segment.
"""

def generate_dialogue(
    config: AppConfig,
    text_handler: Union[OpenAIHandler, GoogleHandler],
    cost_tracker: CostTracker,
    topic: str,
    cefr_level: str,
    word_count: int,
    additional_info: str = ""
) -> str:
    """
    Generates the podcast dialogue script using the configured text provider.

    Args:
        config (AppConfig): The application configuration.
        text_handler (Union[OpenAIHandler, GoogleHandler]): The initialized text handler.
        cost_tracker (CostTracker): The cost tracker for this request.
        topic (str): The main topic for the dialogue.
        cefr_level (str): The target CEFR level (e.g., 'B1').
        word_count (int): The target word count for the dialogue.
        additional_info (str): Optional additional constraints or information for the prompt.

    Returns:
        str: The generated dialogue script.

    Raises:
        ValueError: If the configured text provider is unsupported or misconfigured.
        Exception: Propagates exceptions from the underlying API handler.
    """
    additional_info_text = f"\nAdditional guidance: {additional_info}" if additional_info else ""
    prompt = DIALOGUE_PROMPT_TEMPLATE.format(
        cefr_level=cefr_level,
        topic=topic,
        additional_info=additional_info_text,
        speaker1_name=config.speaker_1_name,
        speaker2_name=config.speaker_2_name,
        word_count=word_count
    )

    start_time = time.time()
    dialogue_content = ""
    try:
        provider_name = type(text_handler).__name__.replace("Handler", "")
        print(f"Generating dialogue using {provider_name} ({config.text_model})...")

        if isinstance(text_handler, OpenAIHandler):
            # Estimate max tokens generously for OpenAI, adjust multiplier as needed.
            # Max tokens limit depends on the model (e.g., 4096 for older models, more for newer ones).
            max_tokens_openai = min(4090, word_count * 4) # Allow ample buffer
            dialogue_content = text_handler.generate_text(prompt, max_tokens_openai, cost_tracker)
        elif isinstance(text_handler, GoogleHandler):
            # Google handler uses max_words indirectly to set max_output_tokens
            dialogue_content = text_handler.generate_text(prompt, word_count, cost_tracker)
        else:
            # This case should ideally not be reached if handlers are initialized correctly
            raise ValueError(f"Unsupported text handler type: {type(text_handler).__name__}")

        end_time = time.time()
        if dialogue_content:
            actual_word_count = len(dialogue_content.split())
            print(f"Dialogue generated ({provider_name}) in {end_time - start_time:.2f} seconds.")
            print(f"Target word count: ~{word_count}, Actual word count: {actual_word_count}")
            # Optional: Add check for significant word count deviation and log a warning
            if abs(actual_word_count - word_count) > word_count * 0.25: # If > 25% deviation
               print(f"Warning: Actual word count ({actual_word_count}) deviates significantly from target ({word_count}). Prompt tuning may be needed.")
        else:
            # This might happen if the API returns empty content despite no error (e.g., filtered)
            print(f"Warning: {provider_name} returned empty dialogue content.")
            # The specific handler might have already raised an error for safety blocks.

        # Basic cleanup: Remove potential leading/trailing whitespace
        return dialogue_content.strip()

    except Exception as e:
        print(f"Error during dialogue generation: {e}")
        # Log details for debugging, error will be surfaced by the wrapper
        log_error_details("generate_dialogue", config.text_provider, config.text_model, e,
                          {"topic": topic, "cefr": cefr_level, "words": word_count})
        raise # Re-raise to be caught by the main Gradio function


def text_to_audio(
    config: AppConfig,
    tts_handler: Union[OpenAIHandler, GoogleHandler],
    cost_tracker: CostTracker,
    text: str,
) -> Optional[str]:
    """
    Converts the dialogue text into a single audio file using the configured TTS provider.

    Processes the dialogue line by line, determines the speaker, calls the appropriate
    TTS API, and concatenates the resulting audio segments.

    Args:
        config (AppConfig): The application configuration.
        tts_handler (Union[OpenAIHandler, GoogleHandler]): The initialized TTS handler.
        cost_tracker (CostTracker): The cost tracker for this request.
        text (str): The dialogue script generated by `generate_dialogue`.

    Returns:
        Optional[str]: The file path to the generated MP3 audio file if successful,
                       otherwise None.
    """
    # Ensure output directories exist
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Generate unique filenames for audio and log
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_uuid = uuid.uuid4()
    # Use absolute paths to avoid issues with CWD
    file_path = os.path.abspath(os.path.join(TEMP_DIR, f"dialogue_{current_time}_{file_uuid}.mp3"))
    log_path = os.path.abspath(os.path.join(LOG_DIR, f"log_{current_time}_{file_uuid}.txt"))

    # Prepare log content
    full_log = [
        f"--- Audio Generation Log ---",
        f"Timestamp: {current_time}",
        f"UUID: {file_uuid}",
        f"Text Provider: {config.text_provider} ({config.text_model or 'N/A'})",
        f"TTS Provider: {config.tts_provider} ({config.audio_model if config.tts_provider == PROVIDER_OPENAI else 'Google Cloud TTS'})",
        f"Speaker 1 ({config.speaker_1_name}): Voice='{config.speaker_1_voice or 'Not Set'}'",
        f"Speaker 2 ({config.speaker_2_name}): Voice='{config.speaker_2_voice or 'Not Set'}'",
        f"Output Audio File Target: {file_path}",
        f"--- Input Dialogue Text ---",
        text,
        f"--- Processing Lines ---"
    ]

    dialogue_lines = text.strip().split('\n')
    error_messages: List[str] = [] # Collect user-facing error messages
    success_count = 0
    skipped_empty_count = 0
    skipped_format_count = 0
    skipped_config_count = 0
    synthesis_error_count = 0

    audio_accumulator = io.BytesIO() # Accumulate audio bytes in memory

    provider_name = type(tts_handler).__name__.replace("Handler", "")
    print(f"--- Starting Audio Synthesis ({provider_name}) ---")

    for line_idx, line in enumerate(dialogue_lines):
        line_num = line_idx + 1
        line = line.strip()

        if not line:
            skipped_empty_count += 1
            full_log.append(f"L{line_num}: SKIPPED - Empty line.")
            continue

        content = ""
        voice = None
        speaker_name = "Unknown"

        # Parse speaker and content based on configured names
        s1_prefix = f"{config.speaker_1_name}:"
        s2_prefix = f"{config.speaker_2_name}:"

        if line.startswith(s1_prefix):
            speaker_name = config.speaker_1_name
            voice = config.speaker_1_voice
            content = line[len(s1_prefix):].strip()
        elif line.startswith(s2_prefix):
            speaker_name = config.speaker_2_name
            voice = config.speaker_2_voice
            content = line[len(s2_prefix):].strip()
        else:
            skipped_format_count += 1
            log_msg = f"L{line_num}: SKIPPED - Line does not start with expected speaker prefix ('{s1_prefix}' or '{s2_prefix}'). Line: '{line[:60]}...'"
            full_log.append(log_msg)
            # Optionally add to error_messages if this should be reported prominently
            # error_messages.append(f"Line {line_num}: Incorrect format, skipped.")
            continue

        if not content:
            skipped_empty_count += 1 # Count lines with prefix but no content as empty/skipped
            full_log.append(f"L{line_num}: SKIPPED - No dialogue content found for speaker '{speaker_name}'.")
            continue

        if not voice:
            skipped_config_count += 1
            log_msg = f"L{line_num}: SKIPPED - Configuration missing: Voice for speaker '{speaker_name}' (check SPEAKER_X_VOICE in .env)."
            full_log.append(log_msg)
            error_messages.append(f"Configuration Error: Voice for {speaker_name} not set.")
            continue

        line_log_prefix = f"L{line_num} ({speaker_name}, Voice: {voice}): "
        full_log.append(f"{line_log_prefix}Synthesizing \"{content[:60]}...\"")

        try:
            audio_bytes: Optional[bytes] = None
            if isinstance(tts_handler, OpenAIHandler):
                 audio_bytes = tts_handler.synthesize_speech(content, voice, cost_tracker)
            elif isinstance(tts_handler, GoogleHandler):
                 audio_bytes = tts_handler.synthesize_speech(content, voice, cost_tracker)
            else:
                 # Should not happen with proper initialization checks
                 raise ValueError(f"Unsupported TTS handler type: {type(tts_handler).__name__}")

            if audio_bytes and len(audio_bytes) > 0:
                audio_accumulator.write(audio_bytes)
                success_msg = f"SUCCESS - L{line_num}: Synthesized {len(audio_bytes)} bytes for {speaker_name}."
                full_log.append(f"{line_log_prefix}SUCCESS - Audio size: {len(audio_bytes)} bytes")
                print(success_msg) # Provide line-by-line success feedback
                success_count += 1
            elif audio_bytes is None: # Explicit check for synthesis function indicating failure
                synthesis_error_count += 1
                error_msg = f"Synthesis Error - L{line_num}: API call failed for '{content[:40]}...'. Check provider logs/status."
                full_log.append(f"{line_log_prefix}ERROR - Synthesis function returned None.")
                error_messages.append(f"Line {line_num} ({speaker_name}): Synthesis failed.")
                print(f"ERROR: {error_msg}")
            else: # audio_bytes is not None but empty
                synthesis_error_count += 1 # Treat empty audio as an error
                warn_msg = f"Synthesis Warning - L{line_num}: Received empty audio data (0 bytes) for '{content[:40]}...'"
                full_log.append(f"{line_log_prefix}WARNING - Received empty audio data.")
                error_messages.append(f"Line {line_num} ({speaker_name}): Received empty audio.")
                print(f"WARNING: {warn_msg}")

        except Exception as e:
            synthesis_error_count += 1
            error_msg = f"CRITICAL Synthesis Error - L{line_num} ({speaker_name}): {type(e).__name__} - {e}"
            full_log.append(f"{line_log_prefix}CRITICAL ERROR:\n{traceback.format_exc()}")
            error_messages.append(f"Line {line_num}: Critical Error - {type(e).__name__}")
            print(f"CRITICAL ERROR: {error_msg}")
            raise e
            # Decide whether to continue or break based on error type (e.g., auth errors might be fatal)
            # For now, continue processing other lines.

    # --- Finalization and Logging ---
    total_processed = line_idx + 1
    full_log.append(f"--- Processing Summary ---")
    full_log.append(f"Total lines in input: {len(dialogue_lines)}")
    full_log.append(f"Lines processed: {total_processed}")
    full_log.append(f"  Successfully synthesized: {success_count}")
    full_log.append(f"  Skipped (Empty/Content): {skipped_empty_count}")
    full_log.append(f"  Skipped (Format Error): {skipped_format_count}")
    full_log.append(f"  Skipped (Config Error): {skipped_config_count}")
    full_log.append(f"  Synthesis Errors/Warnings: {synthesis_error_count}")

    final_audio_bytes = audio_accumulator.getvalue()
    output_path: Optional[str] = None

    if final_audio_bytes and success_count > 0: # Only save if we got *some* successful audio
        try:
            with open(file_path, 'wb') as f_out:
                f_out.write(final_audio_bytes)
            final_size = len(final_audio_bytes)
            log_msg = f"Final audio file created successfully: {file_path} ({final_size} bytes)"
            full_log.append(log_msg)
            print(log_msg)
            output_path = file_path
        except IOError as e:
            error_msg = f"FATAL ERROR: Failed to write final audio file '{file_path}': {e}"
            full_log.append(error_msg)
            error_messages.append(f"Failed to save audio file: {e}")
            print(error_msg)
            # Ensure output_path remains None
            output_path = None
    elif success_count == 0:
        error_msg = "ERROR: No audio segments were successfully synthesized. No output file created."
        full_log.append(error_msg)
        if not error_messages: # Add a generic error if none were collected
            error_messages.append("Failed to generate any audio content.")
        print(error_msg)
    else: # Has bytes but success_count is 0? Should not happen.
         error_msg = "INTERNAL WARNING: Audio bytes exist but success count is zero. File not saved."
         full_log.append(error_msg)
         print(error_msg)


    # Add collected error messages to the log for easy viewing
    if error_messages:
        unique_errors = sorted(list(set(error_messages))) # Show unique errors
        full_log.append(f"--- Encountered Errors/Warnings ({len(unique_errors)} unique types) ---")
        print(f"--- Synthesis Errors/Warnings ({len(unique_errors)} unique types) ---")
        for err in unique_errors:
            full_log.append(f"- {err}")
            print(f"- {err}")
        print(f"See log file for full details: {log_path}")
        full_log.append("--- End Errors/Warnings ---")


    # Always attempt to save the log file
    try:
        with open(log_path, 'w', encoding='utf-8') as log_file:
            log_file.write("\n".join(full_log))
        print(f"Detailed log saved to: {log_path}")
    except Exception as log_e:
        print(f"CRITICAL ERROR: Failed to write log file '{log_path}': {log_e}")

    # Return the path ONLY if the file was successfully written
    return output_path


# --- Utility Functions ---
def log_error_details(function_name: str, provider: Optional[str], model: Optional[str],
                       error: Exception, params: Optional[Dict[str, Any]] = None):
    """
    Logs detailed information about an exception.

    Args:
        function_name (str): Name of the function where the error occurred.
        provider (Optional[str]): The API provider involved, if applicable.
        model (Optional[str]): The model involved, if applicable.
        error (Exception): The exception object.
        params (Optional[Dict[str, Any]]): Relevant parameters at the time of error.
    """
    error_details = {
        "timestamp": datetime.datetime.now().isoformat(),
        "function": function_name,
        "provider": provider or "N/A",
        "model": model or "N/A",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "parameters": params or {},
        "traceback": traceback.format_exc() # Full traceback
    }
    log_filename = os.path.join(LOG_DIR, f"error_log_{datetime.datetime.now().strftime('%Y%m%d')}.jsonl")
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"--- ERROR logged in {function_name} ---")
    print(f"Type: {error_details['error_type']}")
    print(f"Message: {error_details['error_message']}")
    print(f"Provider/Model: {error_details['provider']}/{error_details['model']}")
    print(f"See details below and in '{log_filename}'.")
    # Print selective details to console for quick diagnosis
    # print(json.dumps(error_details, indent=2)) # Avoid dumping full traceback to console usually

    # Append error details to a JSON Lines file for structured logging
    try:
        with open(log_filename, 'a', encoding='utf-8') as f:
            json.dump(error_details, f)
            f.write('\n')
    except Exception as log_e:
        print(f"CRITICAL: Failed to write to error log file '{log_filename}': {log_e}")
        # Fallback: print the whole thing if logging fails
        print("--- Full Error Details (Logging Failed) ---")
        try:
            print(json.dumps(error_details, indent=2))
        except TypeError: # Handle non-serializable params if any
            print(str(error_details)) # Fallback to string representation
        print("------------------------------------------")


# --- Global Instances (Initialized at Startup) ---
APP_CONFIG: AppConfig = AppConfig() # Load config immediately
# Initialize handlers to None; they will be set in initialize_handlers
TEXT_HANDLER: Optional[Union[OpenAIHandler, GoogleHandler]] = None
TTS_HANDLER: Optional[Union[OpenAIHandler, GoogleHandler]] = None

def initialize_handlers():
    """
    Initializes the appropriate provider handler instances based on AppConfig.
    Sets the global TEXT_HANDLER and TTS_HANDLER variables.
    """
    global TEXT_HANDLER, TTS_HANDLER
    print("Initializing API Handlers...")

    # --- Initialize Text Handler ---
    if APP_CONFIG.text_provider == PROVIDER_OPENAI:
        if OpenAI:
            TEXT_HANDLER = OpenAIHandler(APP_CONFIG)
            if not TEXT_HANDLER.is_available():
                print(f"Warning: OpenAI Text Handler initialized but client is not available (check API key/network). Text generation may fail.")
        else:
            print("Error: OpenAI Text Provider configured, but 'openai' library is not installed.")
            # TEXT_HANDLER remains None
    elif APP_CONFIG.text_provider == PROVIDER_GOOGLE:
        if genai:
            TEXT_HANDLER = GoogleHandler(APP_CONFIG) # GoogleHandler init handles Gemini setup
            if not TEXT_HANDLER.is_text_available():
                print(f"Warning: Google Text Handler (Gemini) initialized but model is not available (check API key/config/network). Text generation may fail.")
        else:
             print("Error: Google Text Provider configured, but 'google-generativeai' library is not installed.")
             # TEXT_HANDLER remains None
    else:
        print(f"Error: Unsupported TEXT_PROVIDER configured: '{APP_CONFIG.text_provider}'. Must be '{PROVIDER_OPENAI}' or '{PROVIDER_GOOGLE}'.")

    # --- Initialize TTS Handler ---
    if APP_CONFIG.tts_provider == PROVIDER_OPENAI:
        if OpenAI:
            # Reuse the text handler instance if it's already an OpenAIHandler
            if isinstance(TEXT_HANDLER, OpenAIHandler):
                TTS_HANDLER = TEXT_HANDLER
                print("Reusing existing OpenAIHandler for TTS.")
            else:
                TTS_HANDLER = OpenAIHandler(APP_CONFIG) # Initialize separately otherwise
                print("Initialized separate OpenAIHandler for TTS.")

            if not TTS_HANDLER or not TTS_HANDLER.is_available():
                 print(f"Warning: OpenAI TTS Handler initialized but client is not available (check API key/network). Audio generation may fail.")
        else:
            print("Error: OpenAI TTS Provider configured, but 'openai' library is not installed.")
            # TTS_HANDLER remains None

    elif APP_CONFIG.tts_provider == PROVIDER_GOOGLE:
         if texttospeech:
            # Reuse the text handler instance if it's already a GoogleHandler (which initializes TTS client)
            if isinstance(TEXT_HANDLER, GoogleHandler):
                 TTS_HANDLER = TEXT_HANDLER
                 print("Reusing existing GoogleHandler for TTS.")
                 # Need to check if the TTS part specifically is available
                 if not TTS_HANDLER.is_tts_available():
                     print(f"Warning: Reused GoogleHandler, but its TTS client is not available (check credentials/network). Audio generation may fail.")
            else:
                 TTS_HANDLER = GoogleHandler(APP_CONFIG) # Initialize separately
                 print("Initialized separate GoogleHandler for TTS.")
                 if not TTS_HANDLER or not TTS_HANDLER.is_tts_available():
                    print(f"Warning: Google TTS Handler initialized but client is not available (check credentials/network). Audio generation may fail.")
         else:
             print("Error: Google TTS Provider configured, but 'google-cloud-texttospeech' library is not installed.")
             # TTS_HANDLER remains None
    else:
        print(f"Error: Unsupported TTS_PROVIDER configured: '{APP_CONFIG.tts_provider}'. Must be '{PROVIDER_OPENAI}' or '{PROVIDER_GOOGLE}'.")

    print("Handler initialization complete.")


# --- Gradio Main Function ---
def generate_audio_dialogue_wrapper(
    topic: str,
    additional_info: str = "",
    cefr_level: str = "B1",
    word_count: int = 300,
) -> Tuple[str, Optional[str], str]:
    """
    Gradio wrapper function: Orchestrates dialogue and audio generation.

    Uses the pre-initialized global handlers (TEXT_HANDLER, TTS_HANDLER) based
    on the application configuration (APP_CONFIG).

    Args:
        topic (str): The topic for the podcast dialogue.
        additional_info (str): Optional additional instructions for the text generation.
        cefr_level (str): The target CEFR level.
        word_count (int): The target word count for the dialogue.

    Returns:
        Tuple[str, Optional[str], str]: A tuple containing:
            - The generated dialogue text (or an error message).
            - The file path to the generated audio file (or None if failed).
            - A summary of the estimated API costs (or an error message).
    """
    # Re-check handlers at request time for robustness
    if TEXT_HANDLER is None:
        return ("Error: Text Generation Handler not initialized. Check startup logs and .env configuration.",
                None,
                "Cost N/A - Handler Error")
    if TTS_HANDLER is None:
        return ("Error: Text-to-Speech Handler not initialized. Check startup logs and .env configuration.",
                None,
                "Cost N/A - Handler Error")

    # Create a new cost tracker for each request
    cost_tracker = CostTracker()

    request_id = str(uuid.uuid4())[:8] # Short ID for correlating logs
    print(f"\n--- New Request [{request_id}] ---")
    print(f"Params: Topic='{topic}', CEFR='{cefr_level}', Words={word_count}, AddInfo='{additional_info[:50]}...'")
    print(f"Text Handler: {type(TEXT_HANDLER).__name__} ({APP_CONFIG.text_provider}/{APP_CONFIG.text_model})")
    print(f"TTS Handler: {type(TTS_HANDLER).__name__} ({APP_CONFIG.tts_provider})")

    dialogue_text = "Error: Dialogue generation failed unexpectedly." # Default error text
    audio_output_path: Optional[str] = None
    cost_summary = "Error: Cost calculation failed or not performed." # Default error cost

    try:
        # === Step 1: Generate Dialogue Text ===
        print(f"[{request_id}] Step 1: Generating dialogue...")
        dialogue_text = generate_dialogue(
            APP_CONFIG, TEXT_HANDLER, cost_tracker,
            topic, cefr_level, int(word_count), additional_info
        )
        if not dialogue_text:
             # Handle case where generation succeeded but returned empty (e.g., filtered)
             dialogue_text = "Dialogue generation returned empty content. This might be due to safety filters or prompt issues. Check logs."
             # Proceed to cost calculation, but skip audio generation
             print(f"[{request_id}] Warning: Dialogue generation returned empty content.")
             cost_summary = cost_tracker.get_summary(
                APP_CONFIG.text_provider, APP_CONFIG.tts_provider, APP_CONFIG.text_model,
                APP_CONFIG.speaker_1_voice, APP_CONFIG.speaker_2_voice )
             return dialogue_text, None, cost_summary # Return early

        # === Step 2: Generate Audio from Text ===
        print(f"[{request_id}] Step 2: Generating audio...")
        audio_output_path = text_to_audio(
            APP_CONFIG, TTS_HANDLER, cost_tracker, dialogue_text
        )

        if not audio_output_path:
             warning_msg = "\n\n--- WARNING: Audio generation failed or produced no valid audio segments. Check logs for details. ---"
             dialogue_text += warning_msg # Append warning to the script output
             print(f"[{request_id}] Warning: Audio generation step failed or returned no path.")
        else:
             print(f"[{request_id}] Audio generation successful: {audio_output_path}")

        # === Step 3: Format Cost Summary ===
        print(f"[{request_id}] Step 3: Calculating cost summary...")
        cost_summary = cost_tracker.get_summary(
            APP_CONFIG.text_provider, APP_CONFIG.tts_provider, APP_CONFIG.text_model,
            APP_CONFIG.speaker_1_voice, APP_CONFIG.speaker_2_voice
        )
        print(f"[{request_id}] Request completed.")

    except Exception as e:
        print(f"--- CRITICAL ERROR in Request [{request_id}] ---")
        # Log detailed error info using the utility function
        log_error_details("generate_audio_dialogue_wrapper", "N/A", "N/A", e,
                          {"topic": topic, "cefr": cefr_level, "words": word_count})
        print(f"--------------------------------------")

        # Update outputs to clearly show an error occurred
        dialogue_text = (f"An unexpected error occurred during processing:\n"
                         f"Error Type: {type(e).__name__}\n"
                         f"Message: {e}\n\n"
                         f"Please check the application logs (in the '{LOG_DIR}' directory) for more details.\n\n"
                         f"Traceback snippet:\n{traceback.format_exc(limit=3)}") # Show limited traceback in UI
        audio_output_path = None # Ensure audio path is None on error
        cost_summary = f"An error occurred during processing. Cost calculation may be incomplete.\nError: {e}"

    # Ensure outputs strictly match Gradio component types (str, Optional[str], str)
    # Gradio handles None for Audio component gracefully (shows nothing)
    return dialogue_text, audio_output_path, cost_summary


# --- Gradio UI Definition ---
def create_gradio_interface() -> gr.Interface:
    """Creates and returns the Gradio user interface with Japanese labels."""

    # Dynamically build the description based on loaded config, using Japanese text
    # 少し調整して、設定が見つからない場合のメッセージも日本語にします
    text_model_display = APP_CONFIG.text_model or '未設定 - TEXT_MODELを確認'
    tts_backend_display = f"{APP_CONFIG.audio_model}" if APP_CONFIG.tts_provider == PROVIDER_OPENAI else 'Google Cloud TTS'
    s1_voice_display = APP_CONFIG.speaker_1_voice or '未設定 - SPEAKER_1_VOICEを確認'
    s2_voice_display = APP_CONFIG.speaker_2_voice or '未設定 - SPEAKER_2_VOICEを確認'

    ui_description = f"""
    トピックとCEFRレベルに基づいて、英語学習ポッドキャストの対話と音声を生成します。

    **現在の設定（`.env` ファイルから読込）:**
    *   **テキスト生成:** {APP_CONFIG.text_provider.upper()} (モデル: {text_model_display})
    *   **音声合成(TTS):** {APP_CONFIG.tts_provider.upper()} ({tts_backend_display})
    *   **スピーカー1 ({APP_CONFIG.speaker_1_name}):** ボイス: `{s1_voice_display}`
    *   **スピーカー2 ({APP_CONFIG.speaker_2_name}):** ボイス: `{s2_voice_display}`

    **注意:** APIキー、認証情報、モデル、ボイスなどが `.env` ファイルに正しく設定されているか確認してください。
    `.env` ファイルを変更した後は、アプリケーションを再起動する必要があります。詳細はREADMEを参照してください。
    """

    print("Gradioインターフェースを作成中...")
    
    with gr.Blocks() as demo:
        gr.Markdown("# CEFR English Podcast ジェネレーター")
        
        with gr.Accordion("設定情報", open=False):
            gr.Markdown(ui_description)
            
        with gr.Row():
            # 左側のカラム - 入力フォーム
            with gr.Column(scale=1):
                topic = gr.Textbox(label="トピック", placeholder="例：毎日の読書の利点")
                additional_info = gr.Textbox(label="追加情報／制約（任意）", lines=3,
                                    placeholder="例：非常に楽観的なトーンで。可能であれば具体的な本のタイトルに言及してください。")
                cefr_level = gr.Dropdown(choices=["A1", "A2", "B1", "B2", "C1", "C2"], label="対象 CEFR レベル", value="B1",
                                    info="おおよその英語能力レベルを選択してください。")
                word_count = gr.Slider(minimum=100, maximum=1500, value=350, step=50, label="目標単語数",
                                    info="対話スクリプトのおおよその合計単語数。")
                
                generate_btn = gr.Button("生成", variant="primary")
            
            # 右側のカラム - 出力結果
            with gr.Column(scale=1):
                output_dialogue = gr.Textbox(label="生成された対話スクリプト", lines=20, show_copy_button=True)
                output_audio = gr.Audio(label="生成されたポッドキャスト音声", type="filepath", format="mp3")
                output_cost = gr.Textbox(label="APIコスト見積もりと詳細", lines=5, show_copy_button=True)
        
        generate_btn.click(
            fn=generate_audio_dialogue_wrapper,
            inputs=[topic, additional_info, cefr_level, word_count],
            outputs=[output_dialogue, output_audio, output_cost]
        )
        
    return demo

# --- Main Execution Block ---
if __name__ == "__main__":
    print("\n" + "=" * 30)
    print("--- Application Starting ---")
    print("=" * 30)

    # Initialize configuration (already done globally) and API handlers
    initialize_handlers()

    # Perform essential startup checks based on initialized handlers
    print("\n--- Startup Configuration Checks ---")
    initialization_ok = True
    # Check Text Handler
    if APP_CONFIG.text_provider == PROVIDER_OPENAI and (TEXT_HANDLER is None or not TEXT_HANDLER.is_available()):
        print(f"❌ ERROR: OpenAI Text Provider configured, but Handler failed initialization or client unavailable.")
        initialization_ok = False
    elif APP_CONFIG.text_provider == PROVIDER_GOOGLE and (TEXT_HANDLER is None or not TEXT_HANDLER.is_text_available()):
        print(f"❌ ERROR: Google Text Provider configured, but Handler failed initialization or model unavailable.")
        initialization_ok = False
    elif TEXT_HANDLER is None:
         print(f"❌ ERROR: No valid Text Handler initialized. Check TEXT_PROVIDER setting.")
         initialization_ok = False
    else:
         print(f"✔️ Text Handler ({type(TEXT_HANDLER).__name__}) appears initialized.")

    # Check TTS Handler
    if APP_CONFIG.tts_provider == PROVIDER_OPENAI and (TTS_HANDLER is None or not TTS_HANDLER.is_available()):
         print(f"❌ ERROR: OpenAI TTS Provider configured, but Handler failed initialization or client unavailable.")
         initialization_ok = False
    elif APP_CONFIG.tts_provider == PROVIDER_GOOGLE and (TTS_HANDLER is None or not TTS_HANDLER.is_tts_available()):
         print(f"❌ ERROR: Google TTS Provider configured, but Handler failed initialization or client unavailable.")
         initialization_ok = False
    elif TTS_HANDLER is None:
         print(f"❌ ERROR: No valid TTS Handler initialized. Check TTS_PROVIDER setting.")
         initialization_ok = False
    else:
         print(f"✔️ TTS Handler ({type(TTS_HANDLER).__name__}) appears initialized.")

    # Check Voices (Crucial for TTS)
    if not APP_CONFIG.speaker_1_voice:
         print("⚠️ WARNING: SPEAKER_1_VOICE is not set in .env. Audio generation for Speaker 1 will likely fail.")
         # Depending on provider, this might be fatal or just use a default (less likely for Google)
    else:
        print(f"✔️ Speaker 1 Voice: '{APP_CONFIG.speaker_1_voice}'")

    if not APP_CONFIG.speaker_2_voice:
         print("⚠️ WARNING: SPEAKER_2_VOICE is not set in .env. Audio generation for Speaker 2 will likely fail.")
    else:
         print(f"✔️ Speaker 2 Voice: '{APP_CONFIG.speaker_2_voice}'")

    print("-" * 30)

    if not initialization_ok:
        print("\n🚨 Critical initialization errors detected.")
        print("The application might not function correctly or at all.")
        print("Please check your '.env' configuration (API keys, models, voices),")
        print("installed libraries (openai, google-generativeai, google-cloud-texttospeech),")
        print("and network connectivity/API access.")
        # Optionally exit here if critical components failed and UI launch is pointless
        # import sys
        # sys.exit(1)
    else:
        print("\n✅ Initialization checks passed (basic handler availability).")

    # Create and launch the Gradio UI
    print("\nLaunching Gradio interface...")
    gradio_ui = create_gradio_interface()

    # Launch the interface (consider adding share=True for public access if needed)
    try:
        gradio_ui.launch()
    except Exception as e:
        print("\n" + "=" * 30)
        print(f"🚨 FATAL ERROR during Gradio launch: {e}")
        print(traceback.format_exc())
        print("Please check if the port is already in use or if there are other system issues.")
        print("=" * 30)

    print("\n--- Application Session Ended ---")
