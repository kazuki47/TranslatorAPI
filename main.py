import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app) 

MODEL_NAME = "facebook/nllb-200-distilled-600M"
print(f"サーバーを起動しています。NLLBモデル ({MODEL_NAME}) を読み込んでいます...")
try:
    # 英語 -> 日本語モデル
    # pipelineの引数でソース言語とターゲット言語を指定する
    en_to_ja_translator = pipeline(
        'translation', 
        model=MODEL_NAME, 
        src_lang='eng_Latn', 
        tgt_lang='jpn_Jpan'
    )
    print("✅ 英語->日本語 (NLLB) モデルの読み込み完了。")
    
    # 日本語 -> 英語モデル
    ja_to_en_translator = pipeline(
        'translation',
        model=MODEL_NAME,
        src_lang='jpn_Jpan',
        tgt_lang='eng_Latn'
    )
    print("✅ 日本語->英語 (NLLB) モデルの読み込み完了。")
    
except Exception as e:
    print(f"モデルの読み込み中にエラーが発生しました: {e}")
    en_to_ja_translator = None
    ja_to_en_translator = None


@app.route("/")
def index():
    return "双方向翻訳APIサーバー (NLLBモデル) が起動しています。"

@app.route("/api/translate/en-to-ja", methods=['POST'])
def translate_en_to_ja():
    if en_to_ja_translator is None:
        return jsonify({"error": "英語->日本語モデルの準備ができていません。"}), 500
    data = request.get_json()
    text = data.get('text', '')
    result = en_to_ja_translator(text)
    return jsonify({"translated_text": result[0]['translation_text']})

@app.route("/api/translate/ja-to-en", methods=['POST'])
def translate_ja_to_en():
    if ja_to_en_translator is None:
        return jsonify({"error": "日本語->英語モデルの準備ができていません。"}), 500
    data = request.get_json()
    text = data.get('text', '')
    result = ja_to_en_translator(text)
    return jsonify({"translated_text": result[0]['translation_text']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
