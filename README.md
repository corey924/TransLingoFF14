## FF14 Translation Model Training
此專案旨在訓練 FF14 網路遊戲的文本翻譯模型，使用已存在的日文、英文、繁體中文翻譯文件進行訓練。
※ 因為對於語言模型的訓練研究時間不足，訓練出來的模型未達預期，目前仍無法正確轉譯，僅供參考，期待有高人指點！

### 專案結構
```
.
├── data
│ └── languages
│ └── merged.csv
├── models
│ └── trained_model
├── src
│ ├── data_preprocessing.py
│ ├── train_model.py
│ └── trans_test.py
└── README.md
```


### 前置需求

1. Python 3.8 及以上版本
2. 安裝所需的套件：
```
pip install -r requirements.txt
```


### 使用方法

1. 資料預處理
執行 `data_preprocessing.py` 來解壓縮語言文件並合併翻譯資料。
```
python src/data_preprocessing.py
```

2. 模型訓練
執行 `train_model.py` 來訓練翻譯模型。
```
python src/train_model.py
```

3. 測試翻譯
執行 `trans_test.py` 來測試訓練好的模型。
```
python src/trans_test.py
```


### 說明
* `data_preprocessing.py`: 負責解壓縮及合併翻譯文件。
* `train_model.py`: 負責訓練翻譯模型，包含斷點續訓功能。
* `trans_test.py`: 測試訓練完成的模型翻譯效果。


### 注意事項
訓練過程中若記憶體不足，可調整批次大小。
訓練過程若中斷，可利用斷點續訓功能繼續訓練。
