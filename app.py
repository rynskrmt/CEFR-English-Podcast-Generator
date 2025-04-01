import gradio as gr
import io, os
import uuid
import time
import traceback
import datetime
from dotenv import load_dotenv

# --- Provider Specific Imports ---
try:
    from openai import OpenAI, APIError # Import APIError for better handling
except ImportError:
    print("OpenAI library not installed. Skipping OpenAI specific imports.")
    OpenAI = None
    APIError = None

try:
    import google.generativeai as genai
    from google.cloud import texttospeech
    from google.api_core import exceptions as google_exceptions # Import Google API exceptions
except ImportError:
    print("Google libraries (google-generativeai, google-cloud-texttospeech) not installed. Skipping Google specific imports.")
    genai = None
    texttospeech = None
    google_exceptions = None
# --- End Provider Specific Imports ---

# --- 環境変数の読み込み ---
load_dotenv(override=True)
print(f"DEBUG: 環境変数を読み込み中...")
TEXT_PROVIDER = os.getenv("TEXT_PROVIDER", "openai").lower()
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "openai").lower()
TEXT_MODEL = os.getenv("TEXT_MODEL")
AUDIO_MODEL = os.getenv("AUDIO_MODEL")
SPEAKER_1_VOICE = os.getenv("SPEAKER_1_VOICE")
SPEAKER_2_VOICE = os.getenv("SPEAKER_2_VOICE")
SPEAKER_1_NAME = os.getenv("SPEAKER_1_NAME", "Speaker 1")
SPEAKER_2_NAME = os.getenv("SPEAKER_2_NAME", "Speaker 2")

print(f"DEBUG: TEXT_PROVIDER = '{TEXT_PROVIDER}'")
print(f"DEBUG: TTS_PROVIDER = '{TTS_PROVIDER}'")
print(f"DEBUG: TEXT_MODEL = '{TEXT_MODEL}'")

# --- Client Initialization ---
openai_client = None
gemini_model = None
google_tts_client = None

def initialize_gemini_client(model_name=None):
    """Initializes or reinitializes the Gemini client with the specified model."""
    global gemini_model, TEXT_MODEL
    
    if model_name:
        TEXT_MODEL = model_name
    
    if not TEXT_MODEL:
        TEXT_MODEL = "gemini-1.5-flash-latest"  # デフォルトモデル
        print(f"No model specified, using default model: {TEXT_MODEL}")
    
    # APIキーの確認
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: No API key found for Gemini. Check GEMINI_API_KEY or GOOGLE_API_KEY in .env")
        return None
    
    # 古いバージョンのサポート
    if hasattr(genai, 'configure'):
        genai.configure(api_key=api_key)
    
    try:
        # 新しいバージョン vs. 古いバージョンのAPI
        if hasattr(genai, 'GenerativeModel'):
            # 新しいバージョン
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
            model = genai.GenerativeModel(TEXT_MODEL, safety_settings=safety_settings)
            print(f"Gemini model '{TEXT_MODEL}' initialized.")
            return model
        else:
            print("Error: Could not find GenerativeModel in genai module.")
            return None
    except Exception as e:
        print(f"Error initializing Gemini model '{TEXT_MODEL}': {e}")
        return None


if TEXT_PROVIDER == "openai" and OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: TEXT_PROVIDER is 'openai' but OPENAI_API_KEY is not set.")
    else:
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elif TEXT_PROVIDER == "google" and genai:
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        print("Warning: TEXT_PROVIDER is 'google' but neither GEMINI_API_KEY nor GOOGLE_API_KEY is set.")
    else:
        gemini_model = initialize_gemini_client()
        if not gemini_model:
            print("Warning: Failed to initialize Gemini model.")

if TTS_PROVIDER == "openai" and OpenAI:
    if not openai_client: # Initialize if not already done for text
         if not os.getenv("OPENAI_API_KEY"):
             print("Warning: TTS_PROVIDER is 'openai' but OPENAI_API_KEY is not set.")
         else:
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elif TTS_PROVIDER == "google" and texttospeech:
    try:
        # GOOGLE_APPLICATION_CREDENTIALS env var will be used automatically if set
        # Otherwise, it uses Application Default Credentials (ADC)
        google_tts_client = texttospeech.TextToSpeechClient()
        print("Google TTS Client initialized.")
    except Exception as e:
        print(f"Error initializing Google TTS Client: {e}. Ensure credentials (ADC or GOOGLE_APPLICATION_CREDENTIALS) are set up correctly.")
        google_tts_client = None
# --- End Client Initialization ---

# --- Pricing ---
# Add Gemini and Google TTS pricing (example values, update with current rates)
# Gemini pricing can be complex (per character/token, depends on model, free tier exists)
# Google TTS pricing (per million characters, standard vs WaveNet)
MODEL_PRICES = {
    # OpenAI
    "gpt-4o": {"input": 0.005 / 1000, "output": 0.015 / 1000}, # Price per token
    "gemini-1.5-flash-latest": {"input": 0.000125 / 1000, "output": 0.000375 / 1000}, # Example per 1k characters
    "gemini-1.5-pro-latest": {"input": 0.00125 / 1000, "output": 0.00375 / 1000},    # Example per 1k characters
    "gemini-pro": {"input": 0.000125 / 1000, "output": 0.000375 / 1000}, # Example per 1k chars (check if token based)
    "gemini-2.0-flash": {"input": 0.00010 / 1000, "output": 0.0004 / 1000},    # Example per 1k characters
}

TTS_PRICES = {
    # OpenAI (per 1k characters)
    "tts-1": 0.015 / 1000,
    "tts-1-hd": 0.030 / 1000,
    "google-tts-standard": 0.004 / 1000,
    "google-tts-wavenet": 0.030 / 1000,
    "google-tts-chirp3-hd": 0.030 / 1000,
}

# Helper to determine Google TTS price tier based on voice name
def get_google_tts_tier(voice_name):
    if voice_name and "Wavenet" in voice_name:
        return "google-tts-wavenet"
    elif "Chirp3-HD" in voice_name:
        return "google-tts-chirp3-hd"
    else:
        return "google-tts-standard"

# Global cost info
api_cost_info = {
    "text_generation": 0.0,
    "audio_generation": 0.0,
    "total": 0.0,
    "details": []
}

# --- Cost Calculation Functions ---
def calculate_text_cost(provider, model_name, input_data, output_data):
    """Calculates text generation cost for OpenAI or Google Gemini."""
    cost = 0.0
    detail = {
        "timestamp": datetime.datetime.now().isoformat(),
        "provider": provider,
        "model": model_name,
        "type": "text_generation"
    }

    if model_name not in MODEL_PRICES:
        print(f"Warning: Pricing not found for model '{model_name}'. Cost will be $0.00.")
        detail["cost"] = 0.0
        api_cost_info["details"].append(detail)
        return 0.0

    prices = MODEL_PRICES[model_name]

    if provider == "openai":
        # Assumes input_data is usage.prompt_tokens, output_data is usage.completion_tokens
        input_tokens = input_data
        output_tokens = output_data
        input_cost = input_tokens * prices["input"]
        output_cost = output_tokens * prices["output"]
        cost = input_cost + output_cost
        detail.update({
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "input_cost": input_cost, "output_cost": output_cost, "total_cost": cost
        })
    elif provider == "google":
         # Assumes input_data is prompt (string), output_data is response_text (string)
         # Using character count for Gemini pricing example
         # TODO: Update if Gemini provides token counts and prices are token-based
        input_chars = len(input_data)
        output_chars = len(output_data)
        # Using 'input' price for input chars, 'output' price for output chars
        # Check Gemini official pricing for the correct unit (token/char)
        input_cost = input_chars * prices["input"]
        output_cost = output_chars * prices["output"]
        cost = input_cost + output_cost
        detail.update({
            "input_chars": input_chars, "output_chars": output_chars,
            "input_cost": input_cost, "output_cost": output_cost, "total_cost": cost
        })
    else:
        print(f"Warning: Unknown text provider '{provider}'. Cost calculation skipped.")
        return 0.0

    api_cost_info["text_generation"] += cost
    api_cost_info["total"] += cost
    api_cost_info["details"].append(detail)
    print(f"Text Generation Cost ({provider}/{model_name}): ${cost:.6f}")
    return cost

def calculate_tts_cost(provider, model_identifier, voice_name, text):
    """Calculates TTS cost for OpenAI or Google."""
    cost = 0.0
    char_count = len(text)
    detail = {
        "timestamp": datetime.datetime.now().isoformat(),
        "provider": provider,
        "model_identifier": model_identifier, # e.g., "tts-1", "google-tts"
        "voice": voice_name,
        "character_count": char_count,
        "type": "audio_generation"
    }

    if provider == "openai":
        if model_identifier in TTS_PRICES:
            cost = char_count * TTS_PRICES[model_identifier]
            detail["cost_per_char"] = TTS_PRICES[model_identifier]
        else:
            print(f"Warning: Pricing not found for OpenAI TTS model '{model_identifier}'. Cost will be $0.00.")
            cost = 0.0
    elif provider == "google":
        tts_tier = get_google_tts_tier(voice_name)
        if tts_tier in TTS_PRICES:
            cost = char_count * TTS_PRICES[tts_tier]
            detail["cost_per_char"] = TTS_PRICES[tts_tier]
            detail["tts_tier"] = tts_tier # Add tier info
        else:
            # This case should ideally not happen if get_google_tts_tier is correct
            print(f"Warning: Pricing not found for Google TTS tier '{tts_tier}'. Cost will be $0.00.")
            cost = 0.0
    else:
        print(f"Warning: Unknown TTS provider '{provider}'. Cost calculation skipped.")
        return 0.0

    detail["cost"] = cost
    api_cost_info["audio_generation"] += cost
    api_cost_info["total"] += cost
    api_cost_info["details"].append(detail)
    print(f"TTS Cost ({provider}/{voice_name}): ${cost:.6f} for {char_count} characters")
    return cost
# --- End Cost Calculation Functions ---

# --- Core Logic Functions ---
def generate_dialogue_openai(prompt, text_model, max_tokens):
    """Generates dialogue using OpenAI API."""
    if not openai_client:
        raise ValueError("OpenAI client not initialized.")
    try:
        response = openai_client.chat.completions.create(
            model=text_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens, # Note: OpenAI uses max_tokens
            temperature=0.7,
        )
        dialogue_content = response.choices[0].message.content
        # Calculate cost using usage data
        if response.usage:
             calculate_text_cost("openai", text_model, response.usage.prompt_tokens, response.usage.completion_tokens)
        else:
             print("Warning: OpenAI response did not contain usage data. Cost calculation might be inaccurate.")
             # Estimate cost based on char counts if needed (less accurate)
             # calculate_text_cost("openai", text_model, len(prompt), len(dialogue_content))

        return dialogue_content.strip() if dialogue_content else ""
    except APIError as e:
        print(f"OpenAI API Error: Status={e.status_code}, Message={e.message}")
        raise  # Re-raise the error
    except Exception as e:
        print(f"Error during OpenAI dialogue generation: {e}")
        traceback.print_exc()
        raise

def generate_dialogue_gemini(prompt, text_model, max_words):
    """Generates dialogue using Google Gemini API."""
    if not gemini_model:
        raise ValueError(f"Gemini model '{TEXT_MODEL}' not initialized. Check API key and model name.")
    try:
        # Gemini uses max_output_tokens. Estimate based on words.
        max_output_tokens = max_words * 4 # Rough estimate: 4 tokens per word

        # generation_configの設定
        generation_config = {
            "max_output_tokens": max_output_tokens,
            "temperature": 0.7,
        }

        print(f"--- Sending request to Gemini ({text_model}) ---")
        print(f"Max Output Tokens: {max_output_tokens}")
        print(f"---------------------------------------------")

        # 古いバージョンのAPIでは直接テキストを渡す方法
        response = gemini_model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        print(f"--- Received response from Gemini ---")

        # 応答からテキストを抽出
        try:
            # 最新バージョンのレスポンス形式
            if hasattr(response, 'parts'):
                dialogue_content = "".join(part.text for part in response.parts)
            # 古いバージョンのレスポンス形式
            elif hasattr(response, 'text'):
                dialogue_content = response.text
            else:
                # どちらの形式でもない場合は候補から取得を試みる
                dialogue_content = response.candidates[0].content.parts[0].text
        except (AttributeError, IndexError) as e:
            print(f"Error extracting text from Gemini response: {e}")
            print("Full Response:", response)
            dialogue_content = None

        # コスト計算
        if hasattr(response, 'usage_metadata'):
            print("Gemini Usage Metadata:", response.usage_metadata)
            calculate_text_cost("google", text_model, prompt, dialogue_content or "")
        else:
            print("Warning: Gemini response did not contain usage_metadata. Estimating cost based on character count.")
            calculate_text_cost("google", text_model, prompt, dialogue_content or "")

        # 安全性チェック
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
            print(f"Warning: Prompt blocked by Gemini. Reason: {response.prompt_feedback.block_reason}")
            raise ValueError(f"Prompt blocked by Gemini safety filters: {response.prompt_feedback.block_reason}")

        if not dialogue_content:
            # 応答がない場合、エラーをチェック
            if hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0], 'finish_reason'):
                if response.candidates[0].finish_reason != 'STOP':
                    print(f"Warning: Gemini generation finished unexpectedly. Reason: {response.candidates[0].finish_reason}")
                    raise ValueError(f"Gemini generation failed or was blocked. Finish Reason: {response.candidates[0].finish_reason}")

        return dialogue_content.strip() if dialogue_content else ""

    except google_exceptions.GoogleAPICallError as e:
        print(f"Google API Call Error: {e}")
        if isinstance(e, google_exceptions.ResourceExhausted):
            print("Quota likely exceeded for Gemini API.")
        raise
    except Exception as e:
        print(f"Error during Gemini dialogue generation: {e}")
        traceback.print_exc()
        raise

def generate_dialogue(topic, cefr_level, word_count=300, additional_info=""):
    """Generates dialogue using the selected text provider."""
    word_count = int(word_count)
    additional_info_text = f"\nAdditional information or constraints: {additional_info}" if additional_info else ""

    # --- Unified Prompt ---
    # Keep the prompt structure mostly the same, as both models understand it.
    # Minor tweaks might be needed based on observed model behavior.
    prompt = f"""
        Generate an engaging yet **comfortably paced** podcast dialogue for English language learners at CEFR level {cefr_level}.
        Topic: '{topic}'.{additional_info_text}

        Style: Sound like a **clear, informative, and welcoming educational podcast episode**. Maintain a **natural, conversational flow** between the hosts, similar to a relaxed NPR segment, but with a clear sense of **podcast structure and progression**. Use accessible language for the CEFR level. The tone should be **friendly and approachable for the listener**.
        Speakers: Clearly identify turns for "{SPEAKER_1_NAME}" and "{SPEAKER_2_NAME}". Alternate turns naturally. It's okay if **one speaker occasionally guides the conversation slightly more** (e.g., introducing subtopics, wrapping up points) to enhance the podcast flow, but keep it balanced overall.
        Format:
        {SPEAKER_1_NAME}: [Dialogue text]
        {SPEAKER_2_NAME}: [Dialogue text]

        Content Requirements:
        - Total Length: STRICTLY {word_count} words. Count carefully, fitting the content naturally within this limit.
        - Structure:
            - **Clear Podcast Intro:** Welcome the listener, briefly introduce the hosts (if first time implied) and the episode's topic.
            - **Structured Body:** Discuss the main points/subtopics in a logical sequence. Use **smooth transitions** between points, perhaps with phrases like "So, moving on to..." or "That reminds me of...".
            - **Clear Podcast Outro:** Summarize the key takeaways conversationally, thank the listener for tuning in, and perhaps hint at the next episode or give a simple call to action (e.g., "think about this," "try this out").
        - Educational Elements: Define specialized terms simply *within the natural conversation*. Highlight key points or facts as interesting discussion points, not dry facts.
        - Engagement: Include interesting details or relatable anecdotes. Make the listener feel **included and informed**. Light humor is okay if it fits the topic and tone.
        - **Conversational Flow & Podcast Feel:** **Naturally incorporate** features of spoken English and podcasting:
            - Occasional, appropriate fillers (e.g., 'well,' 'you know,' 'so,' 'right') used sparingly.
            - Natural pauses or hesitations (...).
            - Phrases for agreement, transition, and active listening ('Right,' 'That's a good point,' 'Okay, so let's talk about...').
            - **Subtle awareness of the listener** (without overdoing direct address).
        - Technical: Avoid markdown or special characters *within the dialogue text itself*.

        Remember the primary goal: create an **authentic-sounding and well-structured podcast episode** that is **engaging, educational, easy to follow, and comfortable** for English learners at the specified level and EXACT word count. It should sound like a real podcast they would enjoy listening to.
        """
    # --- End Unified Prompt ---

    try:
        start_time = time.time()
        if TEXT_PROVIDER == "openai":
            print(f"Using OpenAI ({TEXT_MODEL}) for text generation...")
            # OpenAI needs max_tokens, estimate generously
            max_tokens_openai = min(4000, word_count * 3) # More generous token estimate
            dialogue_content = generate_dialogue_openai(prompt, TEXT_MODEL, max_tokens_openai)
        elif TEXT_PROVIDER == "google":
            print(f"Using Google Gemini ({TEXT_MODEL}) for text generation...")
            dialogue_content = generate_dialogue_gemini(prompt, TEXT_MODEL, word_count)
        else:
            raise ValueError(f"Unsupported TEXT_PROVIDER: {TEXT_PROVIDER}")
        end_time = time.time()

        if dialogue_content:
            actual_word_count = len(dialogue_content.split())
            print(f"Dialogue generated ({TEXT_PROVIDER}) in {end_time - start_time:.2f} seconds.")
            print(f"Requested word count: {word_count}, Actual word count: {actual_word_count}")
            # Optional: Add logic here to retry or trim/pad if word count is critical and off
            # if abs(actual_word_count - word_count) > word_count * 0.1: # e.g., if more than 10% off
            #    print("Warning: Word count deviates significantly from request.")
            return dialogue_content
        else:
            raise ValueError(f"No content returned from {TEXT_PROVIDER} API.")

    except Exception as e:
        print(f"Error in generate_dialogue: {e}")
        # Log the error details
        error_details = {
            "timestamp": datetime.datetime.now().isoformat(),
            "provider": TEXT_PROVIDER,
            "model": TEXT_MODEL,
            "error_message": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "parameters": { "topic": topic, "cefr": cefr_level, "words": word_count }
        }
        # Consider writing this to a log file
        print("--- ERROR DETAILS ---")
        import json
        print(json.dumps(error_details, indent=2))
        print("---------------------")
        # Re-raise or return a user-friendly error message
        raise ValueError(f"Failed to generate dialogue using {TEXT_PROVIDER}. Check logs for details. Error: {e}")


def synthesize_speech_openai(text, voice, audio_model_name):
    """Synthesizes speech using OpenAI TTS."""
    if not openai_client:
        raise ValueError("OpenAI client not initialized.")
    try:
        # Calculate cost *before* the API call
        calculate_tts_cost("openai", audio_model_name, voice, text)

        # Use streaming response
        with openai_client.audio.speech.with_streaming_response.create(
            model=audio_model_name, # Should be "tts-1" or "tts-1-hd"
            voice=voice,
            input=text,
            response_format="mp3" # Explicitly set format
        ) as response:
            # Check for non-200 status codes if possible (depends on SDK version)
            # if response.status_code != 200:
            #    raise APIError(f"OpenAI TTS API returned status {response.status_code}", response=response)

            # Read the streamed content into bytes
            audio_bytes = response.read() # Read all data from the stream
            if not audio_bytes:
                 print("Warning: OpenAI TTS returned empty audio data.")
                 return None # Indicate failure
            return audio_bytes

    except APIError as e:
        print(f"OpenAI TTS API Error: Status={e.status_code}, Message={e.message}")
        # Log details from e.response if available
        # error_body = e.response.text if hasattr(e.response, 'text') else "N/A"
        # print(f"Response Body: {error_body}")
        raise # Re-raise
    except Exception as e:
        print(f"Error during OpenAI speech synthesis: {e}")
        traceback.print_exc()
        raise

def synthesize_speech_google(text, voice_name, language_code="en-US", audio_encoding=texttospeech.AudioEncoding.MP3):
    """Synthesizes speech using Google Cloud TTS."""
    if not google_tts_client:
        raise ValueError("Google TTS client not initialized.")
    try:
        # Calculate cost *before* the API call
        calculate_tts_cost("google", AUDIO_MODEL, voice_name, text) # Use generic ID 'google-tts'

        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Ensure voice_name is provided
        if not voice_name:
             raise ValueError("Google TTS requires a valid voice_name.")

        # Select the voice
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code, # Adjust if supporting other languages
            name=voice_name
            # ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL # Or specify if needed
        )

        # Select the type of audio file
        audio_config = texttospeech.AudioConfig(
            audio_encoding=audio_encoding # MP3, LINEAR16, OGG_OPUS etc.
            # Optional: Adjust pitch, speaking rate
            # speaking_rate=1.0,
            # pitch=0.0
        )

        # Perform the text-to-speech request
        response = google_tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        if not response.audio_content:
             print("Warning: Google TTS returned empty audio data.")
             return None # Indicate failure
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

# Renamed get_mp3 to synthesize_speech for broader applicability
def synthesize_speech(text, voice, audio_model_identifier):
    """Synthesizes speech using the selected TTS provider."""
    try:
        if TTS_PROVIDER == "openai":
             # audio_model_identifier should be "tts-1" or "tts-1-hd"
            return synthesize_speech_openai(text, voice, audio_model_identifier)
        elif TTS_PROVIDER == "google":
            # 'voice' parameter contains the Google voice name (e.g., "en-US-Wavenet-D")
            # 'audio_model_identifier' is just "google-tts" for costing
             return synthesize_speech_google(text, voice) # Language code defaults to en-US
        else:
            raise ValueError(f"Unsupported TTS_PROVIDER: {TTS_PROVIDER}")
    except Exception as e:
        # Log the error with context
        print(f"--- TTS Synthesis Error ---")
        print(f"Provider: {TTS_PROVIDER}")
        print(f"Voice: {voice}")
        print(f"Text Snippet: {text[:80]}...")
        print(f"Error: {e}")
        print(f"Traceback:")
        traceback.print_exc()
        print(f"-------------------------")
        # Return None to indicate failure for this line
        return None


def text_to_audio(text, speaker_1_voice, speaker_2_voice, audio_model_id):
    """Processes dialogue text line by line and generates concatenated audio."""
    temporary_directory = "./gradio_cached_examples/tmp/"
    os.makedirs(temporary_directory, exist_ok=True)
    log_directory = "./logs/"
    os.makedirs(log_directory, exist_ok=True)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_uuid = uuid.uuid4()
    # Use NamedTemporaryFile for safer handling, will be cleaned up automatically if needed
    # Or stick to explicit path if Gradio needs a persistent path before returning
    file_path = os.path.join(temporary_directory, f"dialogue_{current_time}_{file_uuid}.mp3")
    log_path = os.path.join(log_directory, f"log_{current_time}_{file_uuid}.txt")

    full_log = [f"--- Log Start: {current_time} ---"]
    full_log.append(f"Text Provider: {TEXT_PROVIDER} ({TEXT_MODEL})")
    full_log.append(f"TTS Provider: {TTS_PROVIDER} ({audio_model_id})") # audio_model_id is like 'tts-1' or 'google-tts'
    full_log.append(f"Speaker 1 ({SPEAKER_1_NAME}): {speaker_1_voice}")
    full_log.append(f"Speaker 2 ({SPEAKER_2_NAME}): {speaker_2_voice}")
    full_log.append(f"Output Audio File: {file_path}")
    full_log.append(f"--- Input Text ---")
    full_log.append(text)
    full_log.append(f"--- Processing Log ---")

    dialogue_lines = text.strip().split('\n')
    error_logs = []
    success_count = 0
    skipped_count = 0
    error_count = 0

    # Use BytesIO to accumulate audio in memory first
    audio_accumulator = io.BytesIO()

    # 初期情報をターミナルにも出力
    print(f"--- Processing Audio Generation ---")
    print(f"TTS Provider: {TTS_PROVIDER} ({audio_model_id})")
    print(f"Speaker 1 ({SPEAKER_1_NAME}): {speaker_1_voice}")
    print(f"Speaker 2 ({SPEAKER_2_NAME}): {speaker_2_voice}")

    for line_idx, line in enumerate(dialogue_lines):
        line = line.strip()
        if not line:
            skipped_count += 1
            continue

        content = ""
        voice = None
        speaker_name = "Unknown"

        if line.startswith(f"{SPEAKER_1_NAME}:"):
            speaker_name = SPEAKER_1_NAME
            voice = speaker_1_voice
            content = line[len(SPEAKER_1_NAME)+1:].strip()
        elif line.startswith(f"{SPEAKER_2_NAME}:"):
            speaker_name = SPEAKER_2_NAME
            voice = speaker_2_voice
            content = line[len(SPEAKER_2_NAME)+1:].strip()
        else:
            full_log.append(f"Line {line_idx+1}: SKIPPED - No speaker prefix found.")
            skipped_count += 1
            continue

        if not content:
            full_log.append(f"Line {line_idx+1}: SKIPPED - Empty content after prefix.")
            skipped_count += 1
            continue

        if not voice:
            full_log.append(f"Line {line_idx+1}: SKIPPED - Voice for speaker '{speaker_name}' is not configured.")
            error_logs.append(f"Line {line_idx+1}: Missing voice for {speaker_name}")
            skipped_count += 1
            continue


        line_log_prefix = f"Line {line_idx+1} ({speaker_name}, Voice: {voice}): "
        full_log.append(f"{line_log_prefix}Processing \"{content[:50]}...\"")

        try:
            # --- Synthesize Speech ---
            # Retry logic can be added here if needed, similar to original code
            audio_bytes = synthesize_speech(content, voice, audio_model_id)
            # --- End Synthesize Speech ---

            if audio_bytes and len(audio_bytes) > 0:
                audio_accumulator.write(audio_bytes)
                success_msg = f"Line {line_idx+1}: SUCCESS - {speaker_name} (Voice: {voice})"
                full_log.append(f"{line_log_prefix}SUCCESS - Audio size: {len(audio_bytes)} bytes")
                print(success_msg)
                success_count += 1
            elif audio_bytes is None: # Explicit check for None indicating synthesis error
                 full_log.append(f"{line_log_prefix}ERROR - Synthesis function returned None (error occurred)")
                 error_logs.append(f"Line {line_idx+1}: Synthesis failed for '{content[:30]}...'")
                 error_count += 1
            else: # audio_bytes is empty but not None
                 full_log.append(f"{line_log_prefix}WARNING - Received empty audio data (0 bytes)")
                 # Treat as skippable or error? Let's count as error for now.
                 error_logs.append(f"Line {line_idx+1}: Received empty audio for '{content[:30]}...'")
                 error_count += 1

        except Exception as e:
            error_msg = f"CRITICAL ERROR during synthesis for line {line_idx+1}: {e}"
            full_log.append(f"{line_log_prefix}{error_msg}")
            full_log.append(traceback.format_exc()) # Add traceback to main log
            error_logs.append(f"Line {line_idx+1}: Critical error - {e}")
            error_count += 1
            # Decide whether to continue or stop processing on critical error
            # continue # Continue processing next lines

    # --- Finalization ---
    full_log.append(f"--- Processing Summary ---")
    full_log.append(f"Successful lines: {success_count}")
    full_log.append(f"Skipped lines: {skipped_count}")
    full_log.append(f"Errored lines: {error_count}")

    final_audio_bytes = audio_accumulator.getvalue()
    if final_audio_bytes and len(final_audio_bytes) > 0:
        with open(file_path, 'wb') as f_out:
            f_out.write(final_audio_bytes)
        final_size = len(final_audio_bytes)
        full_log.append(f"Final audio file created: {file_path} ({final_size} bytes)")
        print(f"Final audio file size: {final_size} bytes")
        output_path = file_path
    else:
        full_log.append(f"ERROR: No audio data was generated. Output file '{file_path}' will be empty or not created.")
        error_logs.append("Failed to generate any audio content.")
        output_path = None # Indicate failure

    # Save the detailed log
    try:
        with open(log_path, 'w', encoding='utf-8') as log_file:
            log_file.write("\n".join(full_log))
        print(f"Detailed log saved to: {log_path}")
    except Exception as log_e:
        print(f"Error writing log file '{log_path}': {log_e}")


    if error_logs:
        print(f"--- ERRORS ENCOUNTERED ({error_count}) ---")
        for err in error_logs[:10]: # Print first 10 errors
            print(err)
        if error_count > 10:
            print(f"... and {error_count - 10} more errors. See log file for details.")
        print(f"--------------------------")
        # Optionally raise an error or return specific error status if needed by Gradio

    # Return the path ONLY if successful
    return output_path if output_path else "Audio generation failed. Check logs."


def generate_audio_dialogue(topic, additional_info="", cefr_level="B1", word_count=300, 
                           text_provider=None, text_model=None, tts_provider=None):
    """Main Gradio function to generate dialogue and audio."""
    global api_cost_info, TEXT_PROVIDER, TTS_PROVIDER, TEXT_MODEL, gemini_model
    
    # 元の設定を保存
    original_text_provider = TEXT_PROVIDER
    original_tts_provider = TTS_PROVIDER
    original_text_model = TEXT_MODEL
    
    # UIから設定が変更された場合
    provider_changed = False
    model_changed = False
    
    if text_provider and text_provider.lower() != TEXT_PROVIDER:
        TEXT_PROVIDER = text_provider.lower()
        provider_changed = True
    
    if text_model and text_model != TEXT_MODEL:
        TEXT_MODEL = text_model
        model_changed = True
    
    if tts_provider and tts_provider.lower() != TTS_PROVIDER:
        TTS_PROVIDER = tts_provider.lower()
    
    # プロバイダーまたはモデルが変更された場合、クライアントを再初期化
    if TEXT_PROVIDER == "google" and (provider_changed or model_changed):
        gemini_model = initialize_gemini_client()
        if not gemini_model:
            print(f"Warning: Failed to initialize Gemini model '{TEXT_MODEL}'")
            # エラーメッセージを返す
            return f"Error: Failed to initialize Gemini model '{TEXT_MODEL}'. Check API key and model name.", None, "Error: Client initialization failed."
    
    # コスト情報のリセット
    api_cost_info = {
        "text_generation": 0.0,
        "audio_generation": 0.0,
        "total": 0.0,
        "details": []
    }
    
    print(f"\n--- New Request ---")
    print(f"Topic: {topic}, CEFR: {cefr_level}, Words: {word_count}, AddInfo: {additional_info}")
    print(f"Text Provider: {TEXT_PROVIDER}, Model: {TEXT_MODEL}")
    print(f"TTS Provider: {TTS_PROVIDER}, Model: {AUDIO_MODEL}")
    print(f"Speaker 1 ({SPEAKER_1_NAME}): {SPEAKER_1_VOICE}")
    print(f"Speaker 2 ({SPEAKER_2_NAME}): {SPEAKER_2_VOICE}")

    dialogue_text = "Error: Dialogue generation failed."
    audio_output = None # Use None for audio output on failure
    cost_summary = "Error: Cost calculation failed."

    try:
        # 1. Generate Dialogue Text
        dialogue_text = generate_dialogue(topic, cefr_level, word_count, additional_info)

        # 2. Generate Audio from Text
        # Pass the correct voice names based on config
        audio_file_path = text_to_audio(dialogue_text, SPEAKER_1_VOICE, SPEAKER_2_VOICE, AUDIO_MODEL)

        # Check if audio generation succeeded
        if audio_file_path and isinstance(audio_file_path, str) and os.path.exists(audio_file_path):
             audio_output = audio_file_path # Set the output path for Gradio Audio component
        else:
             # audio_file_path might contain an error message string or be None
             print(f"Audio generation failed. Reason: {audio_file_path}")
             dialogue_text += "\n\n--- WARNING: Audio generation failed. See logs for details. ---"
             audio_output = None # Ensure audio output is None

        # 3. Format Cost Summary
        total_cost = api_cost_info['total']
        # Rough JPY conversion (update rate as needed)
        jpy_rate = 155
        jpy_cost = total_cost * jpy_rate

        cost_summary = f"""
==== API Usage Cost ====
Text Generation ({TEXT_PROVIDER}): ${api_cost_info['text_generation']:.6f}
Audio Generation ({TTS_PROVIDER}): ${api_cost_info['audio_generation']:.6f}
Total: ${total_cost:.6f}

Estimated JPY: ¥{jpy_cost:,.0f} (at ¥{jpy_rate}/USD)
====================

Cost Details Logged: {len(api_cost_info['details'])} API call(s)
"""
        # Optionally add more details from api_cost_info['details'] if needed

    except Exception as e:
        print(f"--- Error in generate_audio_dialogue ---")
        print(f"Error: {e}")
        traceback.print_exc()
        print(f"--------------------------------------")
        # Update outputs to show error state
        dialogue_text = f"An error occurred: {e}\n\n{traceback.format_exc()}"
        audio_output = None
        cost_summary = f"An error occurred: {e}"

    # Ensure outputs match Gradio component types
    # Textbox expects string, Audio expects filepath string or None, Textbox expects string
    return dialogue_text, audio_output, cost_summary

# --- Gradio UI ---
ui = gr.Interface(
    fn=generate_audio_dialogue,
    inputs=[
        gr.Textbox(label="Topic", placeholder="e.g., The Future of Remote Work"),
        gr.Textbox(label="Additional Info / Constraints (Optional)", placeholder="e.g., Mention the challenges of time zones. Keep the tone optimistic.", lines=3),
        gr.Dropdown(choices=["A1", "A2", "B1", "B2", "C1", "C2"], label="CEFR Level", value="B1"),
        gr.Slider(minimum=100, maximum=2000, value=300, step=50, label="Target Word Count", info="The AI will try its best to match this count."),
    ],
    outputs=[
        gr.Textbox(label="Generated Dialogue Transcript", lines=15),
        gr.Audio(label="Generated Podcast Audio", type="filepath"), # type="filepath" is correct
        gr.Textbox(label="API Cost Estimate")
    ],
    title="AI English Podcast Generator (OpenAI/Google)",
    description=f"Generate English learning podcast dialogues using AI. \n"
                f"Currently configured for: Text: **{TEXT_PROVIDER.upper()} ({TEXT_MODEL})**, "
                f"Audio: **{TTS_PROVIDER.upper()} ({AUDIO_MODEL} - Spk1: {SPEAKER_1_VOICE}, Spk2: {SPEAKER_2_VOICE})**. \n"
                f"Configure providers and models in the `.env` file.",
    allow_flagging='never',
    # examples=[ # Add examples if desired
    #     ["Learning English Phrasal Verbs", "", "B2", 400],
    #     ["A Trip to London", "Focus on vocabulary for ordering food and asking directions.", "A2", 250]
    # ]
)

if __name__ == "__main__":
    print("--- Initializing Application ---")
    print(f"Text Provider: {TEXT_PROVIDER} ({TEXT_MODEL})")
    print(f"TTS Provider: {TTS_PROVIDER} ({AUDIO_MODEL})")
    print(f"Speaker 1 ({SPEAKER_1_NAME}): {SPEAKER_1_VOICE}")
    print(f"Speaker 2 ({SPEAKER_2_NAME}): {SPEAKER_2_VOICE}")
    
    if TEXT_PROVIDER == "google" and not gemini_model:
         print("WARNING: Google Text Provider selected, but Gemini model failed to initialize. Text generation will fail.")
    if TTS_PROVIDER == "google" and not google_tts_client:
         print("WARNING: Google TTS Provider selected, but TTS client failed to initialize. Audio generation will fail.")
    if TEXT_PROVIDER == "openai" and not openai_client and TTS_PROVIDER != "openai": # Only warn if OpenAI text needed but not available
         print("WARNING: OpenAI Text Provider selected, but client failed to initialize (check API Key?). Text generation will fail.")
    if TTS_PROVIDER == "openai" and not openai_client: # Warn if OpenAI TTS needed but not available
         print("WARNING: OpenAI TTS Provider selected, but client failed to initialize (check API Key?). Audio generation will fail.")
    print("-----------------------------")
    # Consider adding a check here to ensure at least one provider for text/TTS is working before launching
    ui.launch()

