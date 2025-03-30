---
title: CEFR English Podcast Generator
emoji: 🎧
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
---

## 🗺️概要
### 🎯背景

グローバル化が進む中で、英語でのコミュニケーション力はますます重要になっています。特に、英語を自分で学んでいる人や語学の先生にとっては、「自分や生徒の興味に合っていて、しかもレベルに合った音声教材」を見つけるのは意外と難しいものです。

市販の教材は多く出回っていますが、多くは内容が固定されていて、学習者一人ひとりの関心やニーズに合わせて柔軟に使えるものは限られています。さらに、CEFR（ヨーロッパ言語共通参照枠）に準拠しながら、適切な難易度を保った音声教材を自作するのは、かなりの手間がかかります。

そこで、このアプリではOpenAIのTTS APIを使って、ユーザーが自由にトピックと英語レベル（CEFR）を指定することで、自分にぴったりの英会話コンテンツを自動で生成・音声化できるようにしました。興味やレベルに合った学習素材を手軽に作れることで、より効果的な英語学習をサポートすることを目指しています。


### 🚀主な機能
このアプリケーションでは、以下のことができます

✅トピックとCEFRレベル（A1〜C2）を指定して、英会話のスクリプトを自動生成  
✅生成されたスクリプトを、OpenAIのTTS APIで音声（MP3）に変換  
✅会話のテキスト（トランスクリプト）も一緒に取得可能  
✅使用したAPIの料金を自動で計算して表示  

## 🎥デモ動画
<blockquote class="twitter-tweet" data-media-max-width="560"><p lang="ja" dir="ltr">英語学習アプリをOSSで公開🚀<br>自分の興味 × 英語レベルに合わせて、英会話教材を自動生成できるツールを作りました。<br><br>✅ トピックとCEFRレベル（A1 - C2）を入力<br>✅ 英語ポッドキャスト音声（MP3）を生成<br>✅ トランスクリプトを自動生成<br><br>興味のあるテーマだから、毎日少しずつでも続けられます🎧 <a href="https://t.co/h1yDWMBmY8">pic.twitter.com/h1yDWMBmY8</a></p>&mdash; rynskrmt（りゅうのすけ） (@rynskrmt) <a href="https://twitter.com/rynskrmt/status/1905931286314905982?ref_src=twsrc%5Etfw">March 29, 2025</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

## 📦インストール方法

```bash
# リポジトリをクローン
git clone https://github.com/rynskrmt/CEFR-English-Podcast-Generator.git
cd cefr-english-podcast-generator

# 必要なパッケージをインストール
pip install -r requirements.txt

```


## 🛠️使用方法
### 必要な環境変数

`.env`ファイルを作成し、以下の変数を設定してください

```
OPENAI_API_KEY=your_openai_api_key_here
TEXT_MODEL=gpt-4o
AUDIO_MODEL=tts-1
SPEAKER_1_VOICE=alloy
SPEAKER_2_VOICE=echo
SPEAKER_1_NAME=Speaker 1
SPEAKER_2_NAME=Speaker 2
```
### 実行
```bash
python app.py
```

ブラウザで自動的に開かれるGradioインターフェースから、以下の情報を入力してください

1. **トピック**: 会話のテーマ（例：Indie Hacking、Climate Change、Food Cultureなど）
2. **追加情報（オプション）**: 会話に含めたい特定の情報や指示
3. **CEFRレベル**: 英語の難易度（A1、A2、B1、B2、C1、C2）
4. **単語数**: 生成する会話の長さ（100〜1000単語）

「Submit」ボタンをクリックすると、以下の出力が得られます
- 生成された会話のテキスト
- 会話の音声ファイル（再生可能）
- API利用料金の概算


## 🐛トラブルシューティング

音声生成に問題が発生した場合
- `logs/`ディレクトリにエラーログが保存されます
- APIキーが正しく設定されているか確認してください

## 💰料金について

このアプリケーションはOpenAIのAPIを使用しており、使用量に応じて料金が発生します
- テキスト生成: GPT-4モデルによる会話作成
- 音声生成: TTSモデルによる音声合成

料金の詳細は出力画面に表示されます（日本円換算を含む）。

## 📄ライセンス
Apache-2.0 License

## ⚠️利用上の注意点
### CEFRレベルについて
CEFRレベルは目安であり、実際の難易度は内容により変動することがあります。生成されるコンテンツは指定されたレベルに近づけるよう最適化されていますが、トピックの専門性や生成される表現によって、実際の難易度が前後する場合があることをご了承ください。  
### 正確性について
生成される英会話は完璧な言語的正確性や教育的妥当性を保証するものではありません。

## 🛑免責事項
本アプリケーションの使用によって生じたいかなる損害についても、開発者は責任を負いません。また、OpenAI APIの使用料金はユーザー自身の負担となります。  

## 🤝Contribution
バグ報告や機能改善の提案は、GitHubのIssueやPull Requestを通じてお願いします。
