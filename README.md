## 🗺️概要
### 🎯背景

グローバル化が進む中で、英語でのコミュニケーション力はますます重要になっています。特に、英語を自分で学んでいる人や語学の先生にとっては、「自分や生徒の興味に合っていて、しかもレベルに合った音声教材」を見つけるのは意外と難しいものです。

市販の教材は多く出回っていますが、多くは内容が固定されていて、学習者一人ひとりの関心やニーズに合わせて柔軟に使えるものは限られています。さらに、CEFR（ヨーロッパ言語共通参照枠）に準拠しながら、適切な難易度を保った音声教材を自作するのは、かなりの手間がかかります。

そこで、このアプリではOpenAIやGoogle CloudのAPIを使用し、ユーザーが自由にトピックと英語レベル（CEFR）を指定することで、自分にぴったりの英会話コンテンツを自動で生成・音声化できるようにしました。興味やレベルに合った学習素材を手軽に作れることで、より効果的な英語学習をサポートすることを目指しています。


### 🚀主な機能
このアプリケーションでは、以下のことができます

✅テキスト生成と音声合成の**プロバイダーを選択可能** (OpenAI / Google Cloud)
✅トピックとCEFRレベル（A1〜C2）を指定して、英会話のスクリプトを自動生成
✅生成されたスクリプトを、選択した**Text to Speech API**で音声（MP3）に変換
✅会話のテキスト（トランスクリプト）も一緒に取得可能
✅使用したAPIの料金を自動で計算して表示

## 🎥デモ動画
https://github.com/user-attachments/assets/02feffa6-176d-461a-98ad-95bffadc1f09

## 📦インストール方法

```bash
# リポジトリをクローン
git clone https://github.com/rynskrmt/CEFR-English-Podcast-Generator.git
cd cefr-english-podcast-generator

# 必要なパッケージをインストール
pip install -r requirements.txt
# (openai, google-generativeai, google-cloud-texttospeech 等のライブラリが含まれます)
```


## 🛠️使用方法
### 必要な環境変数

`.env`ファイルを作成し、以下の環境変数を設定してください。**使用するプロバイダー（OpenAIまたはGoogle）に応じて、必要なキーや設定を行います。**

```dotenv
# Google TTS & Gemini
TEXT_PROVIDER="google"
TTS_PROVIDER="google"

# Text Generation Model
TEXT_MODEL=gemini-2.0-flash

# Audio Generation Model
AUDIO_MODEL=google-tts-chirp3-hd    

# Voice settings
SPEAKER_1_VOICE=en-US-Chirp3-HD-Charon
SPEAKER_1_NAME="Alex"

SPEAKER_2_VOICE=en-US-Chirp3-HD-Leda
SPEAKER_2_NAME="Sarah"

# Google Cloud Application Credentials
GOOGLE_APPLICATION_CREDENTIALS= <your_credentials_file.json> # CredentialをGoogle Cloudで取得し設定する
GEMINI_API_KEY= <your_gemini_api_key> ＃ GeminiAPIKeyをGoogle AI Studio等で取得し設定する


# OpenAIのTTSやLLMのAPIを利用する場合は、上記をコメントアウトし、下記を利用する


# # OpenAI
# TEXT_PROVIDER="openai"
# TTS_PROVIDER="openai"

# TEXT_MODEL=gpt-4o-mini

# # Audio Generation Model
# AUDIO_MODEL=tts-1

# # Voice settings
# SPEAKER_1_VOICE=alloy
# SPEAKER_1_NAME="Alex"

# SPEAKER_2_VOICE=echo
# SPEAKER_2_NAME="Sarah"

# # OpenAI API Key
# OPENAI_API_KEY=
```

**主な環境変数の説明:**

*   `TEXT_PROVIDER`: テキスト生成に使用するプロバイダー (`openai` または `google`)。
*   `TTS_PROVIDER`: 音声合成に使用するプロバイダー (`openai` または `google`)。
*   `OPENAI_API_KEY`: OpenAI APIキー（`TEXT_PROVIDER`または`TTS_PROVIDER`が`openai`の場合に必要）。
*   `GEMINI_API_KEY`: Google Gemini APIキー（`TEXT_PROVIDER`が`google`の場合に必要）。`GOOGLE_API_KEY`でも可。
*   `GOOGLE_APPLICATION_CREDENTIALS`: Google Cloud TTSを使用する場合のサービスアカウントキーファイルへのパス（推奨）。設定しない場合はADCが試行されます。
*   `TEXT_MODEL`: 使用するテキスト生成モデル名（例: `gpt-4o`, `gemini-1.5-flash-latest`）。プロバイダーに合わせて指定。
*   `AUDIO_MODEL`: OpenAI TTSを使用する場合のモデル名 (`tts-1`, `tts-1-hd`)。Google TTSでは直接使用しません。
*   `SPEAKER_1_VOICE`: スピーカー1の音声名。プロバイダーによって指定方法が異なります（例: OpenAI:`alloy`, Google:`en-US-Wavenet-D`）。
*   `SPEAKER_2_VOICE`: スピーカー2の音声名。プロバイダーによって指定方法が異なります（例: OpenAI:`echo`, Google:`en-GB-News-K`）。
*   `SPEAKER_1_NAME`, `SPEAKER_2_NAME`: スクリプト内で使用されるスピーカー名。

### 実行
```bash
python app.py
```

ブラウザで自動的に開かれるGradioインターフェースから、以下の情報を入力してください

1.  **トピック**: 会話のテーマ（例：Indie Hacking、Climate Change、Food Cultureなど）
2.  **追加情報（オプション）**: 会話に含めたい特定の情報や指示
3.  **CEFRレベル**: 英語の難易度（A1、A2、B1、B2、C1、C2）
4.  **単語数**: 生成する会話の長さ（100〜1000単語）

「Submit」ボタンをクリックすると、以下の出力が得られます
- 生成された会話のテキスト
- 会話の音声ファイル（再生可能）
- API利用料金の概算


## 🐛トラブルシューティング

音声生成やテキスト生成に問題が発生した場合
- `logs/`ディレクトリにエラーログが保存されます
- APIキーや認証情報が正しく設定されているか確認してください (`.env`ファイルと、Google Cloudの場合はADCやサービスアカウントキー)

## 💰料金について

このアプリケーションは**OpenAIまたはGoogle CloudのAPI**を使用しており、使用量に応じて料金が発生します
- テキスト生成: GPTモデルまたはGeminiモデルによる会話作成
- 音声生成: OpenAI TTSモデルまたはGoogle Cloud TTSによる音声合成

料金の詳細は出力画面に表示されます（日本円換算を含む）。**（使用したプロバイダーとモデルに応じた料金が計算されます）**

## 📄ライセンス
Apache-2.0 License

## ⚠️利用上の注意点
### CEFRレベルについて
CEFRレベルは目安であり、実際の難易度は内容により変動することがあります。生成されるコンテンツは指定されたレベルに近づけるよう最適化されていますが、トピックの専門性や生成される表現によって、実際の難易度が前後する場合があることをご了承ください。
### 正確性について
生成される英会話は完璧な言語的正確性や教育的妥当性を保証するものではありません。

## 🛑免責事項
本アプリケーションの使用によって生じたいかなる損害についても、開発者は責任を負いません。また、OpenAI APIおよびGoogle Cloud APIの使用料金はユーザー自身の負担となります。

## 🤝Contribution
バグ報告や機能改善の提案は、GitHubのIssueやPull Requestを通じてお願いします。
