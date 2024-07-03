import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset, DatasetDict
import evaluate
import os


def read_and_chunk_data(file_path, chunk_size=10000):
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
        chunk = chunk.astype(str).dropna().reset_index(drop=True)
        yield chunk


def process_chunk(chunk):
    jp_en_tw_pairs = chunk[['jp_text1', 'en_text1', 'tw_text1']].drop_duplicates()
    jp_en_tw_pairs.columns = ['jp_text', 'en_text', 'tw_text']
    jp_en_tw_pairs = jp_en_tw_pairs.dropna()
    jp_en_tw_pairs.reset_index(drop=True, inplace=True)
    dataset = Dataset.from_pandas(jp_en_tw_pairs)
    return dataset


def train_chunk(dataset, model, tokenizer, training_args, checkpoint_dir):
    # 檢查數據集的大小，確保有足夠的樣本進行訓練和驗證
    if len(dataset) < 5:
        print(f"Skipping chunk with {len(dataset)} samples due to insufficient data")
        return

    # 拆分數據集為訓練集和驗證集
    train_test_split = dataset.train_test_split(test_size=0.2)
    datasets = DatasetDict({'train': train_test_split['train'], 'test': train_test_split['test']})

    # 編碼函數
    def preprocess_function(examples):
        inputs = [jp + " " + en for jp, en in zip(examples['jp_text'], examples['en_text'])]
        targets = examples['tw_text']
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
        labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # 預處理數據
    tokenized_datasets = datasets.map(preprocess_function, batched=True)

    # 加載評估指標
    metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        labels = [[label] for label in decoded_labels]
        result = metric.compute(predictions=decoded_preds, references=labels)
        return result

    # 初始化訓練器
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 繼續訓練
    if checkpoint_dir and os.path.isdir(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        trainer.train()

    # 保存模型檢查點
    trainer.save_model(checkpoint_dir)


# 初始化模型和分詞器
model_name = 'Helsinki-NLP/opus-mt-ja-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# 訓練參數設置
training_args = Seq2SeqTrainingArguments(
    output_dir='models/trained_model',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True
)

# 檢查 CUDA 是否可用
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA is available. Using GPU.")
else:
    device = torch.device('cpu')
    print("CUDA is not available. Using CPU.")
model.to(device)

# 讀取和處理數據的每個塊，並訓練模型
file_path = 'data/languages/merged.csv'
chunk_size = 10000  # 可以根據記憶體大小調整塊大小

# 保存檢查點的目錄
checkpoint_dir = 'models/trained_model/checkpoints'

# 找到最新的檢查點
latest_checkpoint = None
if os.path.isdir(checkpoint_dir):
    checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
    else:
        latest_checkpoint = checkpoint_dir

for i, chunk in enumerate(read_and_chunk_data(file_path, chunk_size)):
    dataset = process_chunk(chunk)
    train_chunk(dataset, model, tokenizer, training_args, latest_checkpoint)
    print(f"Chunk {i} training complete")

# 保存最終訓練好的模型
model.save_pretrained('models/trained_model')
tokenizer.save_pretrained('models/trained_model')

print("Model training complete and saved to 'models/trained_model'")
