import gradio as gr
from openai import OpenAI
import io, os
from dotenv import load_dotenv
import uuid
from tempfile import NamedTemporaryFile
from pathlib import Path
import time
import traceback
import datetime

load_dotenv()

# .envから設定を読み込む
TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt-4o")
AUDIO_MODEL = os.getenv("AUDIO_MODEL", "tts-1")
SPEAKER_1_VOICE = os.getenv("SPEAKER_1_VOICE", "alloy")
SPEAKER_2_VOICE = os.getenv("SPEAKER_2_VOICE", "echo")
SPEAKER_1_NAME = os.getenv("SPEAKER_1_NAME", "Speaker 1")
SPEAKER_2_NAME = os.getenv("SPEAKER_2_NAME", "Speaker 2")

# API料金計算用の定数（ドル単位 1000トークンあたり）
MODEL_PRICES = {
    "gpt-4.5-preview": {"input": 0.075, "output": 0.150},
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-audio-preview": {"input": 0.0025, "output": 0.010},
    "gpt-4o-realtime-preview": {"input": 0.0050, "output": 0.020},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.00060},
    "gpt-4o-mini-audio-preview": {"input": 0.00015, "output": 0.00060},
    "gpt-4o-mini-realtime-preview": {"input": 0.00060, "output": 0.00240},
    "gpt-4o-mini-search-preview": {"input": 0.00015, "output": 0.00060},
    "o1": {"input": 0.015, "output": 0.060},
    "o1-pro": {"input": 0.150, "output": 0.600},
    "o3-mini": {"input": 0.00110, "output": 0.00440},
    "o1-mini": {"input": 0.00110, "output": 0.00440}
}

# TTS料金計算用の定数（ドル単位 1000文字あたり）
TTS_PRICES = {
    "tts-1": 0.015,
    "tts-1-hd": 0.03
}

# 料金情報を保存するグローバル変数
api_cost_info = {
    "text_generation": 0.0,
    "audio_generation": 0.0,
    "total": 0.0,
    "details": []
}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def calculate_text_cost(response, model):
    """テキスト生成のコストを計算する関数"""
    if model not in MODEL_PRICES:
        # 未知のモデルの場合は推定できないため0を返す
        return 0.0
    
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    
    input_cost = (input_tokens / 1000) * MODEL_PRICES[model]["input"]
    output_cost = (output_tokens / 1000) * MODEL_PRICES[model]["output"]
    total_cost = input_cost + output_cost
    
    # コスト情報を記録
    cost_detail = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "type": "text_generation"
    }
    
    api_cost_info["text_generation"] += total_cost
    api_cost_info["total"] += total_cost
    api_cost_info["details"].append(cost_detail)
    
    return total_cost

def calculate_tts_cost(text, model):
    """TTS（テキスト読み上げ）のコストを計算する関数"""
    if model not in TTS_PRICES:
        # 未知のモデルの場合は推定できないため0を返す
        return 0.0
    
    # OpenAIのTTSは文字数ベースで課金
    char_count = len(text)
    cost = (char_count / 1000) * TTS_PRICES[model]
    
    # コスト情報を記録
    cost_detail = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model,
        "character_count": char_count,
        "cost": cost,
        "type": "audio_generation"
    }
    
    api_cost_info["audio_generation"] += cost
    api_cost_info["total"] += cost
    api_cost_info["details"].append(cost_detail)
    
    return cost

def generate_dialogue(topic, cefr_level, word_count=300, additional_info=""):
    # TEXT_MODELはもう引数として渡さず、環境変数から取得
    text_model = TEXT_MODEL
    
    # 追加情報があれば、プロンプトに含める
    additional_info_text = f"\nAdditional information: {additional_info}" if additional_info else ""
    
    prompt = f"""
            You are tasked with generating an engaging, educational and fun podcast dialogue designed specifically for English language learners at CEFR level {cefr_level}.
            The conversation should focus on the topic: '{topic}'.{additional_info_text}
            Craft a natural, lively, and informative discussion between two speakers (clearly identified as Speaker 1 and Speaker 2). The dialogue should emulate the engaging, conversational style typical of NPR podcasts, using accessible language appropriate for the specified CEFR level.
            Your dialogue must:
                - Be around {word_count} words.
                - Clearly alternate turns between Speaker 1 and Speaker 2.
                - Avoid special characters or markdown formatting.
                - Define any specialized terms clearly and simply, suitable for a broad, learner-oriented audience.
                - Highlight key points, interesting facts, and relevant definitions within the conversation naturally.
                - Include a few jokes and interesting facts.
                - Make it fun and interesting.
            Use the following format:
                {SPEAKER_1_NAME}: sentence
                {SPEAKER_2_NAME}: sentence
            Remember, your primary goal is to deliver an authentic, stimulating, and educational listening experience tailored for learners of English.
            """

    response = client.chat.completions.create(
        model=text_model,
        messages=[{"role": "user", "content": prompt}],
    )

    # APIコストを計算
    calculate_text_cost(response, text_model)

    dialogue_content = response.choices[0].message.content
    if dialogue_content:
        return dialogue_content.strip()
    else:
        raise ValueError("No content was returned from OpenAI.")

def get_mp3(text, voice, audio_model):
    try:
        # APIコストを計算（TTSの場合は事前に計算）
        calculate_tts_cost(text, audio_model)
        
        with client.audio.speech.with_streaming_response.create(
            model=audio_model,
            voice=voice,
            input=text,
        ) as response:
            with io.BytesIO() as file:
                for chunk in response.iter_bytes():
                    file.write(chunk)
                return file.getvalue()
    except Exception as e:
        # APIエラーの詳細をテキストとして出力
        error_details = {
            "timestamp": datetime.datetime.now().isoformat(),
            "error_message": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "parameters": {
                "model": audio_model,
                "voice": voice,
                "input_text_length": len(text),
                "input_text_preview": text[:50] + ("..." if len(text) > 50 else "")
            }
        }
        
        # エラー情報を整形して出力
        print(f"===== OpenAI API エラー詳細 =====")
        print(f"発生時刻: {error_details['timestamp']}")
        print(f"エラータイプ: {error_details['error_type']}")
        print(f"エラーメッセージ: {error_details['error_message']}")
        print(f"使用モデル: {audio_model}")
        print(f"使用ボイス: {voice}")
        
        if hasattr(e, 'status_code'):
            error_details["status_code"] = e.status_code
            print(f"ステータスコード: {e.status_code}")
            
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            error_details["response_text"] = e.response.text
            print(f"APIレスポンス: {e.response.text}")
            
        print(f"スタックトレース:")
        print(error_details["traceback"])
        print("================================")
        
        # 詳細なエラー情報を辞書として返し、呼び出し元でログに記録できるようにする
        raise Exception(str(error_details))

def text_to_audio(text, speaker_1_voice, speaker_2_voice, audio_model):
    # 一時ディレクトリを作成
    temporary_directory = "./gradio_cached_examples/tmp/"
    os.makedirs(temporary_directory, exist_ok=True)
    
    # ログディレクトリを作成
    log_directory = "./logs/"
    os.makedirs(log_directory, exist_ok=True)
    
    # 現在の日時を取得してログファイル名に含める
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ファイル名をUUIDで生成
    file_uuid = uuid.uuid4()
    file_path = os.path.join(temporary_directory, f"{file_uuid}.mp3")
    error_log_path = os.path.join(log_directory, f"error_log_{current_time}_{file_uuid}.txt")
    
    # 入力テキストをログに記録
    full_log = [f"処理開始時刻: {current_time}"]
    full_log.append(f"入力テキスト:\n{text}\n")
    full_log.append(f"音声モデル: {audio_model}")
    full_log.append(f"話者1ボイス: {speaker_1_voice}")
    full_log.append(f"話者2ボイス: {speaker_2_voice}\n")
    
    dialogue_lines = text.split('\n')
    
    # 成功フラグ
    success = False
    # エラーログを保存するリスト
    error_logs = []
    
    # ファイルをバイナリ書き込みモードで開く
    with open(file_path, 'wb') as f:
        for line_idx, line in enumerate(dialogue_lines):
            if not line.strip():
                continue
                
            line_info = {
                "line_number": line_idx + 1,
                "original_text": line,
                "processed_content": None,
                "voice": None,
                "status": "skipped"
            }
            
            # スピーカー名に基づいて処理
            if line.startswith(f"{SPEAKER_1_NAME}:"):
                voice = speaker_1_voice
                content = line.replace(f"{SPEAKER_1_NAME}:", "").strip()
                line_info["voice"] = speaker_1_voice
                line_info["processed_content"] = content
                line_info["speaker"] = SPEAKER_1_NAME
            elif line.startswith(f"{SPEAKER_2_NAME}:"):
                voice = speaker_2_voice
                content = line.replace(f"{SPEAKER_2_NAME}:", "").strip()
                line_info["voice"] = speaker_2_voice
                line_info["processed_content"] = content
                line_info["speaker"] = SPEAKER_2_NAME
            else:
                full_log.append(f"行 {line_idx+1} をスキップ: スピーカー識別子なし")
                continue
            
            if not content:
                full_log.append(f"行 {line_idx+1} をスキップ: 内容なし")
                continue
                
            # 処理開始をログに記録
            process_start_time = datetime.datetime.now()
            full_log.append(f"\n--- 行 {line_idx+1} 処理開始 [{process_start_time.strftime('%H:%M:%S')}] ---")
            full_log.append(f"スピーカー: {line_info['speaker']}")
            full_log.append(f"ボイス: {voice}")
            full_log.append(f"テキスト: \"{content}\"")
            
            try:
                # APIリクエストを最大3回試行
                retry_count = 0
                max_retries = 3
                while retry_count < max_retries:
                    try:
                        retry_start_time = datetime.datetime.now()
                        full_log.append(f"試行 {retry_count+1}/{max_retries} 開始 [{retry_start_time.strftime('%H:%M:%S')}]")
                        
                        # オーディオデータを取得
                        audio_bytes = get_mp3(content, voice, audio_model)
                        
                        # データサイズをチェック
                        if audio_bytes and len(audio_bytes) > 0:
                            f.write(audio_bytes)
                            line_info["status"] = "success"
                            line_info["audio_size_bytes"] = len(audio_bytes)
                            
                            full_log.append(f"成功: オーディオデータサイズ {len(audio_bytes)} バイト")
                            success = True
                            break
                        else:
                            error_msg = f"警告: '{content[:30]}...'のオーディオデータが空です"
                            line_info["status"] = "empty_response"
                            
                            full_log.append(error_msg)
                            error_logs.append(error_msg)
                            retry_count += 1
                    except Exception as e:
                        error_msg = f"リトライ {retry_count+1}/{max_retries}: {str(e)}"
                        line_info["status"] = "error"
                        line_info["error"] = str(e)
                        
                        full_log.append(error_msg)
                        error_logs.append(error_msg)
                        retry_count += 1
                        time.sleep(3)  # リトライ間隔
                        
                process_end_time = datetime.datetime.now()
                process_duration = (process_end_time - process_start_time).total_seconds()
                full_log.append(f"--- 行 {line_idx+1} 処理完了 [{process_end_time.strftime('%H:%M:%S')}] 所要時間: {process_duration:.2f}秒 ---\n")
                
            except Exception as e:
                error_msg = f"重大なエラー: 行 {line_idx+1} 処理中にキャッチされない例外が発生しました\n"
                error_msg += f"内容: {content[:50]}...\nエラー詳細: {str(e)}\n"
                error_msg += traceback.format_exc()
                
                line_info["status"] = "critical_error"
                line_info["error"] = str(e)
                line_info["traceback"] = traceback.format_exc()
                
                full_log.append(error_msg)
                error_logs.append(error_msg)
    
    # ファイルサイズの確認
    file_size = os.path.getsize(file_path)
    full_log.append(f"\n最終処理結果:")
    
    if file_size == 0:
        msg = f"警告: 生成されたオーディオファイルのサイズが0バイトです"
        full_log.append(msg)
        print(msg)
        
        msg = f"オーディオの生成に失敗しました。詳細はログファイルを確認してください: {error_log_path}"
        full_log.append(msg)
        error_logs.append(msg)
        print(msg)
    else:
        msg = f"最終オーディオファイルサイズ: {file_size} バイト"
        full_log.append(msg)
        print(msg)
    
    # 常に詳細なログを保存（成功した場合も含む）
    with open(error_log_path, 'w', encoding='utf-8') as log_file:
        log_file.write("\n".join(full_log))
    
    # エラーが発生した場合はコンソールにログファイルの場所を表示
    if error_logs:
        print(f"詳細なエラーログをファイルに保存しました: {error_log_path}")
    
    return file_path

def generate_audio_dialogue(topic, cefr_level, word_count=300, additional_info=""):
    # 料金情報をリセット
    global api_cost_info
    api_cost_info = {
        "text_generation": 0.0,
        "audio_generation": 0.0,
        "total": 0.0,
        "details": []
    }
    
    # 環境変数から設定を取得
    audio_model = AUDIO_MODEL
    speaker_1_voice = SPEAKER_1_VOICE
    speaker_2_voice = SPEAKER_2_VOICE
    
    # 追加情報を渡す
    dialogue_text = generate_dialogue(topic, cefr_level, word_count, additional_info)
    audio_file_path = text_to_audio(dialogue_text, speaker_1_voice, speaker_2_voice, audio_model)
    
    # 料金情報を整形
    cost_summary = f"""
==== API利用料金 ====
テキスト生成: ${api_cost_info['text_generation']:.4f}
音声生成: ${api_cost_info['audio_generation']:.4f}
合計: ${api_cost_info['total']:.4f}

※日本円換算: {api_cost_info['total'] * 150:.0f}円 (1ドル=150円として計算)
===================
    """
    
    return dialogue_text, audio_file_path, cost_summary

ui = gr.Interface(
    fn=generate_audio_dialogue,
    inputs=[
        gr.Textbox(label="トピック", placeholder="e.g., Indie Hacking"),
        gr.Textbox(label="追加情報（オプション）", placeholder="", lines=3),
        gr.Dropdown(choices=["A1", "A2", "B1", "B2", "C1", "C2"], label="CEFRレベル", value="B1", info="CEFRレベルは目安であり、実際の難易度は内容により変動することがあります。"),
        gr.Slider(minimum=100, maximum=1000, value=300, step=50, label="単語数", info=""),
    ],
    outputs=[
        gr.Textbox(label="トランスクリプト"),
        gr.Audio(label="音声ファイル", type="filepath"),
        gr.Textbox(label="API利用料金")
    ],
    title="CEFR English Podcast ジェネレーター",
    description="トピックとCEFRレベルに基づいて英語学習者向け会話音声教材を生成します。単語数や追加情報を入力して、ニーズに合わせた会話を生成できます。"
)

if __name__ == "__main__":
    ui.launch()
