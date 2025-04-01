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
from typing import Optional, Dict, Any, Tuple, List, Union

# --- Provider Specific Imports ---
try:
    from openai import OpenAI, APIError
except ImportError:
    print("OpenAI library not installed. Skipping OpenAI specific imports.")
    OpenAI = None
    APIError = None

try:
    import google.generativeai as genai
    from google.cloud import texttospeech
    from google.api_core import exceptions as google_exceptions
except ImportError:
    print("Google libraries (google-generativeai, google-cloud-texttospeech) not installed. Skipping Google specific imports.")
    genai = None
    texttospeech = None
    google_exceptions = None
# --- End Provider Specific Imports ---

# --- Constants ---
PROVIDER_OPENAI = "openai"
PROVIDER_GOOGLE = "google"

DEFAULT_GEMINI_MODEL = "gemini-1.5-flash-latest"
DEFAULT_SPEAKER_1_NAME = "Speaker 1"
DEFAULT_SPEAKER_2_NAME = "Speaker 2"

TEMP_DIR = "./gradio_cached_examples/tmp/"
LOG_DIR = "./logs/"

# Pricing Data (Keep updated)
MODEL_PRICES = {
    "gpt-4o": {"input": 0.005 / 1000, "output": 0.015 / 1000},
    "gemini-1.5-flash-latest": {"input": 0.000125 / 1000, "output": 0.000375 / 1000},
    "gemini-1.5-pro-latest": {"input": 0.00125 / 1000, "output": 0.00375 / 1000},
    "gemini-pro": {"input": 0.000125 / 1000, "output": 0.000375 / 1000},
    "gemini-2.0-flash": {"input": 0.00010 / 1000, "output": 0.0004 / 1000},
}
TTS_PRICES = {
    "tts-1": 0.015 / 1000,
    "tts-1-hd": 0.030 / 1000,
    "google-tts-standard": 0.004 / 1000,
    "google-tts-wavenet": 0.030 / 1000,
    "google-tts-chirp3-hd": 0.030 / 1000, # Added missing price
}

# --- Configuration ---
class AppConfig:
    """Holds application configuration loaded from environment variables."""
    def __init__(self):
        load_dotenv(override=True)
        print("DEBUG: Loading environment variables...")
        self.text_provider: str = os.getenv("TEXT_PROVIDER", PROVIDER_OPENAI).lower()
        self.tts_provider: str = os.getenv("TTS_PROVIDER", PROVIDER_OPENAI).lower()
        self.text_model: Optional[str] = os.getenv("TEXT_MODEL")
        self.audio_model: Optional[str] = os.getenv("AUDIO_MODEL") # e.g., "tts-1", "tts-1-hd" (OpenAI), Google doesn't use model name here
        self.speaker_1_voice: Optional[str] = os.getenv("SPEAKER_1_VOICE")
        self.speaker_2_voice: Optional[str] = os.getenv("SPEAKER_2_VOICE")
        self.speaker_1_name: str = os.getenv("SPEAKER_1_NAME", DEFAULT_SPEAKER_1_NAME)
        self.speaker_2_name: str = os.getenv("SPEAKER_2_NAME", DEFAULT_SPEAKER_2_NAME)
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        # Google TTS uses ADC or GOOGLE_APPLICATION_CREDENTIALS

        # Set default text model if not specified
        if self.text_provider == PROVIDER_GOOGLE and not self.text_model:
            self.text_model = DEFAULT_GEMINI_MODEL
            print(f"DEBUG: No Google text model specified, using default: {self.text_model}")
        elif self.text_provider == PROVIDER_OPENAI and not self.text_model:
            # Set a default OpenAI model if needed, e.g.,
            # self.text_model = "gpt-4o"
            # print(f"DEBUG: No OpenAI text model specified, using default: {self.text_model}")
            pass # Or leave as None and let the API call fail if not set

        print(f"DEBUG: TEXT_PROVIDER = '{self.text_provider}'")
        print(f"DEBUG: TTS_PROVIDER = '{self.tts_provider}'")
        print(f"DEBUG: TEXT_MODEL = '{self.text_model}'")
        print(f"DEBUG: AUDIO_MODEL (for OpenAI TTS) = '{self.audio_model}'")
        print(f"DEBUG: SPEAKER_1_VOICE = '{self.speaker_1_voice}'")
        print(f"DEBUG: SPEAKER_2_VOICE = '{self.speaker_2_voice}'")

# --- Cost Tracking ---
class CostTracker:
    """Tracks API costs during a request."""
    def __init__(self):
        self.text_generation_cost: float = 0.0
        self.audio_generation_cost: float = 0.0
        self.details: List[Dict[str, Any]] = []

    def add_cost(self, cost_type: str, provider: str, cost: float, details: Dict[str, Any]):
        """Adds a cost entry."""
        if cost_type == "text_generation":
            self.text_generation_cost += cost
        elif cost_type == "audio_generation":
            self.audio_generation_cost += cost
        else:
            print(f"Warning: Unknown cost type '{cost_type}'")
            return

        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": cost_type,
            "provider": provider,
            "cost": cost,
            **details # Merge specific details
        }
        self.details.append(entry)
        print(f"Cost Added: Type={cost_type}, Provider={provider}, Cost=${cost:.6f}, Details={details}")


    @property
    def total_cost(self) -> float:
        """Calculates the total cost."""
        return self.text_generation_cost + self.audio_generation_cost

    def get_summary(self, text_provider: str, tts_provider: str, text_model: str,
                     s1_voice: str, s2_voice: str, jpy_rate: int = 150) -> str:
        """Generates a formatted cost summary string."""
        total_tts_chars = sum(d.get('character_count', 0) for d in self.details if d['type'] == 'audio_generation')
        jpy_cost = self.total_cost * jpy_rate

        return f"""
            ==== API 使用料 ====
            テキスト生成 ({text_provider}): ${self.text_generation_cost:.6f}
            音声生成 ({tts_provider}): ${self.audio_generation_cost:.6f} ({total_tts_chars:,} 文字)
            合計: ${self.total_cost:.6f}

            円換算: ¥{jpy_cost:,.0f} (1USD = ¥{jpy_rate}/USD)
            ====================

            使用料の詳細: {len(self.details)} API call(s)
            使用したテキストモデル: {text_model or 'N/A'}
            使用した音声ボイス1: {s1_voice or 'N/A'}
            使用した音声ボイス2: {s2_voice or 'N/A'}
            """

# --- Provider Handlers ---
class BaseProviderHandler:
    def __init__(self, config: AppConfig):
        self.config = config

    def _calculate_cost(self, cost_tracker: CostTracker, cost_type: str, provider: str, cost: float, details: Dict[str, Any]):
         cost_tracker.add_cost(cost_type, provider, cost, details)

class OpenAIHandler(BaseProviderHandler):
    """Handles OpenAI API interactions."""
    def __init__(self, config: AppConfig):
        super().__init__(config)
        self.client: Optional[OpenAI] = None
        if OpenAI and config.openai_api_key:
            try:
                self.client = OpenAI(api_key=config.openai_api_key)
                print("OpenAI Client initialized.")
            except Exception as e:
                print(f"Error initializing OpenAI Client: {e}")
        elif config.text_provider == PROVIDER_OPENAI or config.tts_provider == PROVIDER_OPENAI:
            print("Warning: OpenAI provider selected but API key is missing or library not installed.")

    def is_available(self) -> bool:
        return self.client is not None

    def generate_text(self, prompt: str, max_tokens: int, cost_tracker: CostTracker) -> str:
        if not self.is_available():
            raise RuntimeError("OpenAI client not available.")
        if not self.config.text_model:
             raise ValueError("OpenAI text model not configured (TEXT_MODEL env var).")

        try:
            response = self.client.chat.completions.create(
                model=self.config.text_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            dialogue_content = response.choices[0].message.content

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
                        "input_cost": input_cost, "output_cost": output_cost
                    })
                else:
                    print(f"Warning: Pricing not found for model '{self.config.text_model}'.")
                self._calculate_cost(cost_tracker, "text_generation", PROVIDER_OPENAI, cost, cost_details)
            else:
                print("Warning: OpenAI response missing usage data. Cost calculation skipped.")
                # Optionally estimate based on characters here if needed

            return dialogue_content.strip() if dialogue_content else ""

        except APIError as e:
            print(f"OpenAI API Error (Text Gen): Status={e.status_code}, Message={e.message}")
            raise # Re-raise for higher level handling
        except Exception as e:
            print(f"Error during OpenAI dialogue generation: {e}")
            traceback.print_exc()
            raise

    def synthesize_speech(self, text: str, voice: str, cost_tracker: CostTracker) -> Optional[bytes]:
        if not self.is_available():
            raise RuntimeError("OpenAI client not available.")
        if not self.config.audio_model:
             raise ValueError("OpenAI audio model not configured (AUDIO_MODEL env var, e.g., tts-1).")
        if not voice:
             raise ValueError("OpenAI TTS requires a voice.")

        # Calculate cost *before* API call
        cost = 0.0
        char_count = len(text)
        price = TTS_PRICES.get(self.config.audio_model)
        cost_details = {
            "model_identifier": self.config.audio_model,
            "voice": voice,
            "character_count": char_count
        }
        if price:
            cost = char_count * price
            cost_details["cost_per_char"] = price
        else:
            print(f"Warning: Pricing not found for OpenAI TTS model '{self.config.audio_model}'.")
        self._calculate_cost(cost_tracker, "audio_generation", PROVIDER_OPENAI, cost, cost_details)

        try:
            with self.client.audio.speech.with_streaming_response.create(
                model=self.config.audio_model, # e.g., "tts-1" or "tts-1-hd"
                voice=voice,
                input=text,
                response_format="mp3"
            ) as response:
                # Consider adding status check if supported by SDK version
                # if response.status_code != 200:
                #    raise APIError(...)
                audio_bytes = response.read()
                if not audio_bytes:
                    print("Warning: OpenAI TTS returned empty audio data.")
                    return None
                return audio_bytes

        except APIError as e:
            print(f"OpenAI TTS API Error: Status={e.status_code}, Message={e.message}")
            raise
        except Exception as e:
            print(f"Error during OpenAI speech synthesis: {e}")
            traceback.print_exc()
            raise


class GoogleHandler(BaseProviderHandler):
    """Handles Google AI (Gemini) and Google TTS interactions."""
    def __init__(self, config: AppConfig):
        super().__init__(config)
        self.gemini_model: Optional[Any] = None # Type hint depends on genai version
        self.tts_client: Optional[texttospeech.TextToSpeechClient] = None
        self._initialize_clients()

    def _initialize_clients(self):
        # Initialize Gemini
        if genai and self.config.gemini_api_key:
            if not self.config.text_model:
                 print(f"Warning: Google provider selected, but TEXT_MODEL not set. Using default {DEFAULT_GEMINI_MODEL}.")
                 self.config.text_model = DEFAULT_GEMINI_MODEL # Use default if needed

            try:
                if hasattr(genai, 'configure'):
                    genai.configure(api_key=self.config.gemini_api_key)

                if hasattr(genai, 'GenerativeModel'):
                    safety_settings = [ # Consider making these configurable
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    ]
                    self.gemini_model = genai.GenerativeModel(self.config.text_model, safety_settings=safety_settings)
                    print(f"Google Gemini model '{self.config.text_model}' initialized.")
                else:
                    print("Error: Could not find GenerativeModel in genai module. Check library version.")
            except Exception as e:
                print(f"Error initializing Gemini model '{self.config.text_model}': {e}")
        elif self.config.text_provider == PROVIDER_GOOGLE:
             print("Warning: Google text provider selected but API key is missing or library not installed.")

        # Initialize Google TTS
        if texttospeech:
            try:
                # Uses GOOGLE_APPLICATION_CREDENTIALS or ADC
                self.tts_client = texttospeech.TextToSpeechClient()
                print("Google TTS Client initialized.")
            except Exception as e:
                print(f"Error initializing Google TTS Client: {e}. Ensure credentials are set.")
                self.tts_client = None # Ensure client is None on failure
        elif self.config.tts_provider == PROVIDER_GOOGLE:
            print("Warning: Google TTS provider selected but library not installed or client init failed.")


    def is_text_available(self) -> bool:
        return self.gemini_model is not None

    def is_tts_available(self) -> bool:
        return self.tts_client is not None

    @staticmethod
    def _get_google_tts_tier(voice_name: Optional[str]) -> str:
        if voice_name:
            if "Wavenet" in voice_name: return "google-tts-wavenet"
            if "Chirp" in voice_name: return "google-tts-chirp3-hd" # Assuming Chirp is HD/Premium
        return "google-tts-standard" # Default

    def generate_text(self, prompt: str, max_words: int, cost_tracker: CostTracker) -> str:
        if not self.is_text_available():
             raise RuntimeError(f"Google Gemini model '{self.config.text_model}' not available.")

        try:
            # Estimate max tokens (adjust multiplier as needed)
            max_output_tokens = max_words * 3
            generation_config = {
                "max_output_tokens": max_output_tokens,
                "temperature": 0.7,
            }

            print(f"--- Sending request to Gemini ({self.config.text_model}) ---")
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=generation_config
            )
            print(f"--- Received response from Gemini ---")

            dialogue_content = ""
            # Extract text safely from response (handles different possible structures)
            try:
                if hasattr(response, 'text'): # Simplest case
                     dialogue_content = response.text
                elif hasattr(response, 'parts') and response.parts:
                     dialogue_content = "".join(part.text for part in response.parts if hasattr(part, 'text'))
                elif hasattr(response, 'candidates') and response.candidates:
                     first_candidate = response.candidates[0]
                     if hasattr(first_candidate, 'content') and hasattr(first_candidate.content, 'parts'):
                         dialogue_content = "".join(part.text for part in first_candidate.content.parts if hasattr(part, 'text'))
                     # Fallback for older/different structures if needed
            except (AttributeError, IndexError, TypeError) as e:
                 print(f"Error extracting text from Gemini response: {e}. Response: {response}")
                 # Leave dialogue_content as empty string

            # Cost Calculation (Using character count as fallback if token count unavailable)
            cost = 0.0
            cost_details = {"model": self.config.text_model}
            input_chars = len(prompt)
            output_chars = len(dialogue_content)
            prices = MODEL_PRICES.get(self.config.text_model)

            # Prefer usage metadata if available (check specific fields)
            # Example: if hasattr(response, 'usage_metadata') and response.usage_metadata:
            #    input_tokens = response.usage_metadata.prompt_token_count
            #    output_tokens = response.usage_metadata.candidates_token_count
            #    # ... calculate cost based on tokens ...
            # else:
            # Fallback to character count
            if prices:
                # Using character-based pricing as example
                input_cost = input_chars * prices["input"]
                output_cost = output_chars * prices["output"]
                cost = input_cost + output_cost
                cost_details.update({
                    "input_chars": input_chars, "output_chars": output_chars,
                    "input_cost": input_cost, "output_cost": output_cost
                })
            else:
                print(f"Warning: Pricing not found for model '{self.config.text_model}'.")
            self._calculate_cost(cost_tracker, "text_generation", PROVIDER_GOOGLE, cost, cost_details)

            # Safety Checks
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                reason = response.prompt_feedback.block_reason
                print(f"Warning: Prompt blocked by Gemini. Reason: {reason}")
                raise ValueError(f"Prompt blocked by Gemini safety filters: {reason}")
            if not dialogue_content and hasattr(response, 'candidates') and response.candidates:
                 finish_reason = getattr(response.candidates[0], 'finish_reason', 'UNKNOWN')
                 if finish_reason != 'STOP':
                     print(f"Warning: Gemini generation finished unexpectedly. Reason: {finish_reason}")
                     raise ValueError(f"Gemini generation failed or was blocked. Finish Reason: {finish_reason}")

            return dialogue_content.strip() if dialogue_content else ""

        except google_exceptions.GoogleAPICallError as e:
            print(f"Google API Call Error (Text Gen): {e}")
            if isinstance(e, google_exceptions.ResourceExhausted):
                print("Quota likely exceeded for Gemini API.")
            raise
        except Exception as e:
            print(f"Error during Gemini dialogue generation: {e}")
            traceback.print_exc()
            raise

    def synthesize_speech(self, text: str, voice_name: str, cost_tracker: CostTracker) -> Optional[bytes]:
        if not self.is_tts_available():
            raise RuntimeError("Google TTS client not available.")
        if not voice_name:
             raise ValueError("Google TTS requires a voice_name.")

        # Calculate cost *before* API call
        cost = 0.0
        char_count = len(text)
        tts_tier = self._get_google_tts_tier(voice_name)
        price = TTS_PRICES.get(tts_tier)
        cost_details = {
            "model_identifier": "google-tts", # Generic ID for Google TTS service
            "voice": voice_name,
            "character_count": char_count,
            "tts_tier": tts_tier
        }
        if price:
            cost = char_count * price
            cost_details["cost_per_char"] = price
        else:
            print(f"Warning: Pricing not found for Google TTS tier '{tts_tier}'.")
        self._calculate_cost(cost_tracker, "audio_generation", PROVIDER_GOOGLE, cost, cost_details)

        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US", # Make configurable if needed
                name=voice_name
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3 # Make configurable if needed
            )

            response = self.tts_client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            if not response.audio_content:
                print("Warning: Google TTS returned empty audio data.")
                return None
            return response.audio_content

        except google_exceptions.GoogleAPICallError as e:
            print(f"Google TTS API Call Error: {e}")
            if isinstance(e, google_exceptions.InvalidArgument):
                print(f"Likely issue with parameters like voice name ('{voice_name}') or language code.")
            raise
        except Exception as e:
            print(f"Error during Google speech synthesis: {e}")
            traceback.print_exc()
            raise

# --- Core Logic ---

# Dialogue Prompt Generation
DIALOGUE_PROMPT_TEMPLATE = """
Generate an engaging yet comfortably paced podcast dialogue for English language learners at CEFR level {cefr_level}.
Topic: '{topic}'.{additional_info}

Style: Sound like a clear, informative, and welcoming educational podcast episode. Maintain a natural, conversational flow between the hosts, similar to a relaxed NPR segment, but with a clear sense of podcast structure and progression. Use accessible language for the CEFR level. The tone should be friendly and approachable for the listener.
Speakers: Clearly identify turns for "{speaker1_name}" and "{speaker2_name}". Alternate turns naturally. It's okay if one speaker occasionally guides the conversation slightly more (e.g., introducing subtopics, wrapping up points) to enhance the podcast flow, but keep it balanced overall.
Format:
{speaker1_name}: [Dialogue text]
{speaker2_name}: [Dialogue text]

Content Requirements:
- Total Length: STRICTLY {word_count} words. Count carefully, fitting the content naturally within this limit.
- Structure:
    - Clear Podcast Intro: Welcome the listener, briefly introduce the hosts (if first time implied) and the episode's topic.
    - Structured Body: Discuss the main points/subtopics in a logical sequence. Use smooth transitions between points, perhaps with phrases like "So, moving on to..." or "That reminds me of...".
    - Clear Podcast Outro: Summarize the key takeaways conversationally, thank the listener for tuning in, and perhaps hint at the next episode or give a simple call to action (e.g., think about this, try this out).
- Educational Elements: Define specialized terms simply within the natural conversation. Highlight key points or facts as interesting discussion points, not dry facts.
- Engagement: Include interesting details or relatable anecdotes. Make the listener feel included and informed. Light humor is okay if it fits the topic and tone.
- Conversational Flow & Podcast Feel: Naturally incorporate features of spoken English and podcasting:
    - Occasional, appropriate fillers (e.g., 'well,' 'you know,' 'so,' 'right') used sparingly.
    - Natural pauses or hesitations (...).
    - Phrases for agreement, transition, and active listening ('Right,' 'That's a good point,' 'Okay, so let's talk about...').
    - Subtle awareness of the listener (without overdoing direct address).
    - **Rare instances of repeated words or phrases for simulate natural hesitation (e.g., "It's a a really good point," or "So, so what do you think?")**
- Technical: Avoid markdown or special characters within the dialogue text itself. **Crucially, do not include any formatting characters used in this prompt (like quotation marks) within the generated dialogue. Write plain text.**

Remember the primary goal: create an authentic-sounding and well-structured podcast episode that is engaging, educational, easy to follow, and comfortable for English learners at the specified level and EXACT word count. It should sound like a real podcast they would enjoy listening to.
Must use EXACTLY {cefr_level} level vocabulary and grammar.
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
    """Generates dialogue using the configured text provider."""
    additional_info_text = f"\nAdditional information or constraints: {additional_info}" if additional_info else ""
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
        if config.text_provider == PROVIDER_OPENAI and isinstance(text_handler, OpenAIHandler):
            print(f"Using OpenAI ({config.text_model}) for text generation...")
            # OpenAI needs max_tokens, estimate generously
            max_tokens_openai = min(4000, word_count * 3) # Adjust estimate as needed
            dialogue_content = text_handler.generate_text(prompt, max_tokens_openai, cost_tracker)
        elif config.text_provider == PROVIDER_GOOGLE and isinstance(text_handler, GoogleHandler):
            print(f"Using Google Gemini ({config.text_model}) for text generation...")
            dialogue_content = text_handler.generate_text(prompt, word_count, cost_tracker)
        else:
            raise ValueError(f"Unsupported or misconfigured text provider: {config.text_provider}")

        end_time = time.time()
        if dialogue_content:
            actual_word_count = len(dialogue_content.split())
            print(f"Dialogue generated ({config.text_provider}) in {end_time - start_time:.2f} seconds.")
            print(f"Requested word count: {word_count}, Actual word count: {actual_word_count}")
            # Optional: Add check for significant word count deviation
            # if abs(actual_word_count - word_count) > word_count * 0.1:
            #    print("Warning: Word count deviates significantly from request.")
        else:
            raise ValueError(f"No content returned from {config.text_provider} API.")

        return dialogue_content

    except Exception as e:
        print(f"Error in generate_dialogue: {e}")
        # Log more details if needed
        log_error_details("generate_dialogue", config.text_provider, config.text_model, e,
                          {"topic": topic, "cefr": cefr_level, "words": word_count})
        raise # Re-raise to be caught by the main Gradio function


def text_to_audio(
    config: AppConfig,
    tts_handler: Union[OpenAIHandler, GoogleHandler],
    cost_tracker: CostTracker,
    text: str,
) -> Optional[str]:
    """Processes dialogue text line by line and generates a concatenated audio file path."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_uuid = uuid.uuid4()
    file_path = os.path.join(TEMP_DIR, f"dialogue_{current_time}_{file_uuid}.mp3")
    log_path = os.path.join(LOG_DIR, f"log_{current_time}_{file_uuid}.txt")

    full_log = [f"--- Log Start: {current_time} ---"]
    full_log.append(f"Text Provider: {config.text_provider} ({config.text_model})")
    full_log.append(f"TTS Provider: {config.tts_provider} ({config.audio_model if config.tts_provider == PROVIDER_OPENAI else 'Google TTS'})")
    full_log.append(f"Speaker 1 ({config.speaker_1_name}): {config.speaker_1_voice}")
    full_log.append(f"Speaker 2 ({config.speaker_2_name}): {config.speaker_2_voice}")
    full_log.append(f"Output Audio File: {file_path}")
    full_log.append(f"--- Input Text ---")
    full_log.append(text)
    full_log.append(f"--- Processing Log ---")

    dialogue_lines = text.strip().split('\n')
    error_logs: List[str] = []
    success_count = 0
    skipped_count = 0
    error_count = 0

    audio_accumulator = io.BytesIO()

    print(f"--- Processing Audio Generation ({config.tts_provider}) ---")

    for line_idx, line in enumerate(dialogue_lines):
        line = line.strip()
        if not line:
            skipped_count += 1
            continue

        content = ""
        voice = None
        speaker_name = "Unknown"

        # Determine speaker and content
        if line.startswith(f"{config.speaker_1_name}:"):
            speaker_name = config.speaker_1_name
            voice = config.speaker_1_voice
            content = line[len(config.speaker_1_name)+1:].strip()
        elif line.startswith(f"{config.speaker_2_name}:"):
            speaker_name = config.speaker_2_name
            voice = config.speaker_2_voice
            content = line[len(config.speaker_2_name)+1:].strip()
        else:
            full_log.append(f"Line {line_idx+1}: SKIPPED - No speaker prefix found: '{line[:50]}...'")
            skipped_count += 1
            continue

        if not content:
            full_log.append(f"Line {line_idx+1}: SKIPPED - Empty content for speaker {speaker_name}.")
            skipped_count += 1
            continue

        if not voice:
            log_msg = f"Line {line_idx+1}: SKIPPED - Voice for speaker '{speaker_name}' is not configured."
            full_log.append(log_msg)
            error_logs.append(log_msg)
            skipped_count += 1 # Count as skipped because config is missing
            continue

        line_log_prefix = f"Line {line_idx+1} ({speaker_name}, Voice: {voice}): "
        full_log.append(f"{line_log_prefix}Processing \"{content[:50]}...\"")

        try:
            audio_bytes: Optional[bytes] = None
            if config.tts_provider == PROVIDER_OPENAI and isinstance(tts_handler, OpenAIHandler):
                 audio_bytes = tts_handler.synthesize_speech(content, voice, cost_tracker)
            elif config.tts_provider == PROVIDER_GOOGLE and isinstance(tts_handler, GoogleHandler):
                 audio_bytes = tts_handler.synthesize_speech(content, voice, cost_tracker)
            else:
                 raise ValueError(f"Unsupported or misconfigured TTS provider: {config.tts_provider}")

            if audio_bytes and len(audio_bytes) > 0:
                audio_accumulator.write(audio_bytes)
                success_msg = f"SUCCESS - Line {line_idx+1}: {speaker_name} (Voice: {voice})"
                full_log.append(f"{line_log_prefix}SUCCESS - Audio size: {len(audio_bytes)} bytes")
                print(success_msg)
                success_count += 1
            elif audio_bytes is None: # Explicit check for synthesis error signal
                error_msg = f"ERROR - Line {line_idx+1}: Synthesis failed for '{content[:30]}...' (function returned None)."
                full_log.append(f"{line_log_prefix}{error_msg}")
                error_logs.append(error_msg)
                error_count += 1
            else: # audio_bytes is empty but not None
                warn_msg = f"WARNING - Line {line_idx+1}: Received empty audio data (0 bytes) for '{content[:30]}...'"
                full_log.append(f"{line_log_prefix}{warn_msg}")
                error_logs.append(warn_msg)
                error_count += 1 # Treat as error for now

        except Exception as e:
            error_msg = f"CRITICAL ERROR during synthesis for line {line_idx+1}: {e}"
            full_log.append(f"{line_log_prefix}{error_msg}\n{traceback.format_exc()}")
            error_logs.append(f"Line {line_idx+1}: Critical synthesis error - {e}")
            error_count += 1
            # Continue processing next lines unless it's a fatal error (like auth)

    # Finalization
    full_log.append(f"--- Processing Summary ---")
    full_log.append(f"Successful lines: {success_count}")
    full_log.append(f"Skipped lines: {skipped_count}")
    full_log.append(f"Errored lines: {error_count}")

    final_audio_bytes = audio_accumulator.getvalue()
    output_path: Optional[str] = None
    if final_audio_bytes and len(final_audio_bytes) > 0:
        try:
            with open(file_path, 'wb') as f_out:
                f_out.write(final_audio_bytes)
            final_size = len(final_audio_bytes)
            full_log.append(f"Final audio file created: {file_path} ({final_size} bytes)")
            print(f"Final audio file size: {final_size} bytes. Path: {file_path}")
            output_path = file_path
        except IOError as e:
            error_msg = f"ERROR: Failed to write final audio file '{file_path}': {e}"
            full_log.append(error_msg)
            error_logs.append(error_msg)
            error_count +=1
    else:
        error_msg = f"ERROR: No audio data was successfully generated. Output file '{file_path}' not created."
        full_log.append(error_msg)
        if error_count == 0 and success_count == 0: # Add error only if no successes either
             error_logs.append("Failed to generate any audio content (check logs for details).")

    # Save the detailed log
    try:
        with open(log_path, 'w', encoding='utf-8') as log_file:
            log_file.write("\n".join(full_log))
        print(f"Detailed log saved to: {log_path}")
    except Exception as log_e:
        print(f"Error writing log file '{log_path}': {log_e}")

    if error_logs:
        print(f"--- ERRORS/WARNINGS ENCOUNTERED ({len(error_logs)}) ---")
        for err in error_logs[:10]: # Print first few errors
            print(err)
        if len(error_logs) > 10:
            print(f"... and {len(error_logs) - 10} more. See log file '{log_path}' for details.")
        print(f"-------------------------------------")

    # Return the path ONLY if successful, otherwise None
    return output_path


# --- Utility Functions ---
def log_error_details(function_name: str, provider: str, model: Optional[str],
                       error: Exception, params: Dict[str, Any]):
    """Logs detailed error information."""
    error_details = {
        "timestamp": datetime.datetime.now().isoformat(),
        "function": function_name,
        "provider": provider,
        "model": model or "N/A",
        "error_message": str(error),
        "error_type": type(error).__name__,
        "traceback": traceback.format_exc(),
        "parameters": params
    }
    print("--- ERROR DETAILS ---")
    try:
        print(json.dumps(error_details, indent=2))
    except TypeError: # Handle non-serializable params if any
         print(str(error_details)) # Fallback to string representation
    print("---------------------")
    # Consider writing this to a persistent error log file


# --- Global Instances (Initialized at Startup) ---
APP_CONFIG = AppConfig()
TEXT_HANDLER: Optional[Union[OpenAIHandler, GoogleHandler]] = None
TTS_HANDLER: Optional[Union[OpenAIHandler, GoogleHandler]] = None

def initialize_handlers():
    """Initializes the appropriate provider handlers based on config."""
    global TEXT_HANDLER, TTS_HANDLER

    # Text Handler
    if APP_CONFIG.text_provider == PROVIDER_OPENAI:
        TEXT_HANDLER = OpenAIHandler(APP_CONFIG)
        if not TEXT_HANDLER.is_available():
            print(f"Warning: OpenAI Text Handler initialization failed. Text generation with OpenAI will not work.")
    elif APP_CONFIG.text_provider == PROVIDER_GOOGLE:
        TEXT_HANDLER = GoogleHandler(APP_CONFIG)
        if not TEXT_HANDLER.is_text_available():
             print(f"Warning: Google Text Handler (Gemini) initialization failed. Text generation with Google will not work.")
    else:
        print(f"Error: Unsupported TEXT_PROVIDER: {APP_CONFIG.text_provider}")

    # TTS Handler
    if APP_CONFIG.tts_provider == PROVIDER_OPENAI:
        # Use existing OpenAI handler if text provider is also OpenAI
        if APP_CONFIG.text_provider == PROVIDER_OPENAI and isinstance(TEXT_HANDLER, OpenAIHandler):
             TTS_HANDLER = TEXT_HANDLER
        else:
            TTS_HANDLER = OpenAIHandler(APP_CONFIG) # Initialize separately if needed
        if not TTS_HANDLER or not TTS_HANDLER.is_available():
             print(f"Warning: OpenAI TTS Handler initialization failed. Audio generation with OpenAI will not work.")
    elif APP_CONFIG.tts_provider == PROVIDER_GOOGLE:
         # Use existing Google handler if text provider is also Google
        if APP_CONFIG.text_provider == PROVIDER_GOOGLE and isinstance(TEXT_HANDLER, GoogleHandler):
             TTS_HANDLER = TEXT_HANDLER
        else:
            TTS_HANDLER = GoogleHandler(APP_CONFIG) # Initialize separately if needed
        if not TTS_HANDLER or not TTS_HANDLER.is_tts_available():
             print(f"Warning: Google TTS Handler initialization failed. Audio generation with Google will not work.")
    else:
        print(f"Error: Unsupported TTS_PROVIDER: {APP_CONFIG.tts_provider}")


# --- Gradio Main Function ---
def generate_audio_dialogue_wrapper(
    topic: str,
    additional_info: str = "",
    cefr_level: str = "B1",
    word_count: int = 300,
    # Removed provider/model inputs - config is fixed at startup now
) -> Tuple[str, Optional[str], str]:
    """
    Gradio wrapper function. Orchestrates dialogue and audio generation
    using pre-initialized handlers based on AppConfig.
    """
    # Ensure handlers are initialized (should be done at startup, but check)
    if TEXT_HANDLER is None or TTS_HANDLER is None:
        return ("Error: Provider handlers not initialized. Check startup logs.",
                None,
                "Error: Initialization failed.")

    # Create a new cost tracker for this request
    cost_tracker = CostTracker()

    print(f"\n--- New Request ---")
    print(f"Topic: {topic}, CEFR: {cefr_level}, Words: {word_count}, AddInfo: '{additional_info}'")
    print(f"Using Text Handler: {type(TEXT_HANDLER).__name__} ({APP_CONFIG.text_provider}/{APP_CONFIG.text_model})")
    print(f"Using TTS Handler: {type(TTS_HANDLER).__name__} ({APP_CONFIG.tts_provider})")

    dialogue_text = "Error: Dialogue generation failed."
    audio_output_path: Optional[str] = None
    cost_summary = "Error: Cost calculation failed."

    try:
        # 1. Generate Dialogue Text
        if not TEXT_HANDLER: raise RuntimeError("Text handler not available.")
        dialogue_text = generate_dialogue(
            APP_CONFIG, TEXT_HANDLER, cost_tracker,
            topic, cefr_level, int(word_count), additional_info
        )

        # 2. Generate Audio from Text
        if not TTS_HANDLER: raise RuntimeError("TTS handler not available.")
        audio_output_path = text_to_audio(
            APP_CONFIG, TTS_HANDLER, cost_tracker, dialogue_text
        )

        # Check if audio generation succeeded
        if not audio_output_path:
             # Append warning to text if audio failed but text succeeded
             dialogue_text += "\n\n--- WARNING: Audio generation failed or produced no sound. See logs for details. ---"
             print(f"Audio generation failed or returned no path.")
        else:
             print(f"Audio generation successful: {audio_output_path}")

        # 3. Format Cost Summary
        cost_summary = cost_tracker.get_summary(
            APP_CONFIG.text_provider, APP_CONFIG.tts_provider, APP_CONFIG.text_model,
            APP_CONFIG.speaker_1_voice, APP_CONFIG.speaker_2_voice
        )

    except Exception as e:
        print(f"--- Error in generate_audio_dialogue_wrapper ---")
        # Log details using the utility function
        log_error_details("generate_audio_dialogue_wrapper", "N/A", "N/A", e,
                          {"topic": topic, "cefr": cefr_level, "words": word_count})
        print(f"--------------------------------------")
        # Update outputs to show error state
        dialogue_text = f"An error occurred during processing:\n{type(e).__name__}: {e}\n\nTraceback:\n{traceback.format_exc()}"
        audio_output_path = None # Ensure audio is None on error
        cost_summary = f"An error occurred: {e}"

    # Ensure outputs match Gradio component types
    return dialogue_text, audio_output_path, cost_summary


# --- Gradio UI ---
def create_gradio_interface() -> gr.Interface:
    """Creates and returns the Gradio interface."""
    ui_description = f"""
    トピックとCEFRレベルに基づいて、英語学習ポッドキャストの対話を生成します。
    現在の設定：
    テキスト: **{APP_CONFIG.text_provider.upper()} ({APP_CONFIG.text_model or 'Not Set'})**
    音声: **{APP_CONFIG.tts_provider.upper()} ({APP_CONFIG.audio_model if APP_CONFIG.tts_provider == PROVIDER_OPENAI else 'Google TTS'})**
    スピーカー1 ({APP_CONFIG.speaker_1_name}): **{APP_CONFIG.speaker_1_voice or 'Not Set'}**
    スピーカー2 ({APP_CONFIG.speaker_2_name}): **{APP_CONFIG.speaker_2_voice or 'Not Set'}**
    プロバイダーとモデルの設定は `.env` ファイルで行ってください。
    """

    return gr.Interface(
        fn=generate_audio_dialogue_wrapper,
        inputs=[
            gr.Textbox(label="トピック", placeholder="例：Indie Hacking"),
            gr.Textbox(label="追加情報／制約（任意）", lines=3, placeholder="例：二人の会話は少しユーモアを交えて、最後に簡単なクイズを入れてください。"),
            gr.Dropdown(choices=["A1", "A2", "B1", "B2", "C1", "C2"], label="CEFR レベル", value="B1", info="CEFRレベルは、英語学習者のレベルを示す指標です。"),
            gr.Slider(minimum=100, maximum=1500, value=300, step=50, label="目標単語数"),
            # Removed dynamic provider/model selection from UI inputs
        ],
        outputs=[
            gr.Textbox(label="対話スクリプト", lines=15),
            gr.Audio(label="ポッドキャスト音声", type="filepath"), # type="filepath" expects a path string
            gr.Textbox(label="API コスト見積もり")
        ],
        title="CEFR English Podcast ジェネレーター（OpenAI/Google）",
        description=ui_description,
        flagging_mode='never',
        # examples=[ # Add examples if desired
        #     ["Learning English with Podcasts", "", "B1", 250],
        #     ["The future of AI", "Discuss ethical implications", "C1", 500],
        # ]
    )

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Initializing Application ---")
    # Initialize configuration and handlers
    initialize_handlers()

    # Perform startup checks
    print("--- Startup Checks ---")
    initialization_ok = True
    if APP_CONFIG.text_provider == PROVIDER_OPENAI and (not TEXT_HANDLER or not TEXT_HANDLER.is_available()):
        print("ERROR: OpenAI Text Provider configured, but handler failed initialization.")
        initialization_ok = False
    elif APP_CONFIG.text_provider == PROVIDER_GOOGLE and (not TEXT_HANDLER or not TEXT_HANDLER.is_text_available()):
        print("ERROR: Google Text Provider configured, but handler failed initialization.")
        initialization_ok = False

    if APP_CONFIG.tts_provider == PROVIDER_OPENAI and (not TTS_HANDLER or not TTS_HANDLER.is_available()):
         print("ERROR: OpenAI TTS Provider configured, but handler failed initialization.")
         initialization_ok = False
    elif APP_CONFIG.tts_provider == PROVIDER_GOOGLE and (not TTS_HANDLER or not TTS_HANDLER.is_tts_available()):
         print("ERROR: Google TTS Provider configured, but handler failed initialization.")
         initialization_ok = False

    if not APP_CONFIG.speaker_1_voice or not APP_CONFIG.speaker_2_voice:
         print("WARNING: One or both speaker voices (SPEAKER_1_VOICE, SPEAKER_2_VOICE) are not set in .env. Audio generation might fail depending on the TTS provider.")
         # Google requires voice names, OpenAI has defaults but specific ones are better.

    print("----------------------")

    if not initialization_ok:
        print("Critical initialization errors found. Application might not function correctly.")
        # Optionally exit here if critical components failed
        # sys.exit(1)

    # Create and launch the Gradio UI
    print("Creating Gradio interface...")
    gradio_ui = create_gradio_interface()
    print("Launching Gradio interface...")
    gradio_ui.launch()
    print("Application finished.")
