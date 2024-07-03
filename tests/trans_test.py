import torch
from transformers import MarianMTModel, MarianTokenizer

# 加載已保存的模型和分詞器
model_path = 'models/trained_model'
model = MarianMTModel.from_pretrained(model_path)
tokenizer = MarianTokenizer.from_pretrained(model_path)

# 檢查 CUDA 是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# 測試翻譯函數
def translate(jp_text, en_text):
    try:
        text = jp_text + " " + en_text
        print(f"Combined text: {text}")  # 調試信息：輸入的日文和英文結合後的文本
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        print(f"Tokenized inputs: {inputs}")  # 調試信息：分詞後的輸入
        translated = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
        print(f"Translated tokens: {translated}")  # 調試信息：翻譯後的標記
        translation = tokenizer.batch_decode(translated, skip_special_tokens=True)
        print(f"Decoded translation: {translation}")  # 調試信息：解碼後的翻譯
        return translation[0]
    except Exception as e:
        print(f"Error during translation: {e}")
        return None


# 示例翻譯
jp_text = "こんにちは"
en_text = "Hello"
translation = translate(jp_text, en_text)
print(f"Translation: {translation}")

# 添加更多的測試例子
test_cases = [
    ("アルフィノ", "Alphinaud"),
    ("アリゼー", "Alisaie"),
    ("ヤ・シュトラ", "Y'shtola"),
    ("はい", "Yes"),
    ("いいえ", "No")
]

for jp, en in test_cases:
    print(f"\nTesting with: Japanese='{jp}' and English='{en}'")
    translation = translate(jp, en)
    print(f"Translation: {translation}")
