import zipfile
import pandas as pd
import os
import re

def extract_files(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'<.*?>', '', text)  # 移除HTML標記
    text = re.sub(r'\[.*?\]', '', text)  # 移除中括號內的內容
    text = re.sub(r'\(.*?\)', '', text)  # 移除括號內的內容
    text = re.sub(r'ObjectParameter\(\d+\)', '', text)  # 移除 ObjectParameter
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # 在小寫字母和大寫字母之間添加空格
    text = re.sub(r'\s+', ' ', text)  # 移除多餘的空格
    text = re.sub(r'nan', '', text)  # 移除 'nan'
    text = re.sub(r'[^\w\s]', '', text)  # 移除所有非字母數字和空格字符
    text = text.strip()  # 去除首尾空白字符
    return text

def read_csv_files(directory):
    dataframes = []
    for root, dirs, files in os.walk(directory):
        print(f"Searching in directory: {root}")
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    df.insert(0, 'ID', range(1, 1 + len(df)))
                    # 忽略前兩行
                    df = df.drop([0, 1])
                    # 清洗數據
                    for col in df.columns:
                        df[col] = df[col].map(clean_text)
                    dataframes.append(df)
                    print(f"Successfully read {file_path} with {len(df)} rows")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return dataframes

def ensure_columns(df, column_names):
    for col in column_names:
        if col not in df.columns:
            df[col] = ''
    return df[column_names]

def merge_translation_files(jp_dfs, en_dfs, tw_dfs, batch_size=10):
    batch_count = 0
    column_names = ['ID', 'jp_text1', 'jp_text2', 'en_text1', 'en_text2', 'tw_text1', 'tw_text2']

    with open('data/languages/merged.csv', 'w', encoding='utf-8-sig', newline='') as output_file:
        for i in range(0, len(jp_dfs), batch_size):
            jp_batch = jp_dfs[i:i + batch_size]
            en_batch = en_dfs[i:i + batch_size]
            tw_batch = tw_dfs[i:i + batch_size]

            for jp_df, en_df, tw_df in zip(jp_batch, en_batch, tw_batch):
                try:
                    # 檢查每個數據框的列數是否足夠
                    if len(jp_df.columns) > 3:
                        jp_df = ensure_columns(jp_df, ['ID', jp_df.columns[2], jp_df.columns[3]])
                    else:
                        jp_df = ensure_columns(jp_df, ['ID', jp_df.columns[2], ''])

                    if len(en_df.columns) > 3:
                        en_df = ensure_columns(en_df, ['ID', en_df.columns[2], en_df.columns[3]])
                    else:
                        en_df = ensure_columns(en_df, ['ID', en_df.columns[2], ''])

                    if len(tw_df.columns) > 3:
                        tw_df = ensure_columns(tw_df, ['ID', tw_df.columns[2], tw_df.columns[3]])
                    else:
                        tw_df = pd.DataFrame(
                            {'ID': jp_df['ID'], 'tw_text1': [''] * len(jp_df), 'tw_text2': [''] * len(jp_df)})

                    jp_df.columns = ['ID', 'jp_text1', 'jp_text2']
                    en_df.columns = ['ID', 'en_text1', 'en_text2']
                    tw_df.columns = ['ID', 'tw_text1', 'tw_text2']

                    merged_df = jp_df.merge(en_df, on='ID').merge(tw_df, on='ID', how='left')
                    valid_rows = merged_df.dropna(subset=['jp_text1', 'en_text1'])

                    # 移除無效數據
                    valid_rows = valid_rows[(valid_rows['jp_text1'] != '') & (valid_rows['en_text1'] != '')]

                    if not valid_rows.empty:
                        valid_rows.to_csv(output_file, index=False, header=batch_count == 0, encoding='utf-8-sig',
                                          mode='a')
                        batch_count += 1
                        print(f"Processed batch {batch_count}")

                except KeyError as e:
                    print(f"Merge error: {e}")

    return batch_count

def concatenate_batches(batch_count):
    all_batch_dfs = []
    for i in range(batch_count):
        batch_df = pd.read_csv(f'data/languages/merged_batch_{i}.csv', low_memory=False)
        all_batch_dfs.append(batch_df)
    if all_batch_dfs:
        all_df = pd.concat(all_batch_dfs, ignore_index=True)
        return all_df
    else:
        return pd.DataFrame()  # 返回空的DataFrame

# 示例使用
if __name__ == "__main__":
    zip_path = 'data/languages.zip'
    extract_path = 'data/languages'

    # extract_files(zip_path, extract_path)

    jp_path = os.path.join(extract_path, 'jp')
    en_path = os.path.join(extract_path, 'en')
    tw_path = os.path.join(extract_path, 'tw')

    jp_dfs = read_csv_files(jp_path)
    en_dfs = read_csv_files(en_path)
    tw_dfs = read_csv_files(tw_path)

    print(f"Read {len(jp_dfs)} Japanese files, {len(en_dfs)} English files, {len(tw_dfs)} Chinese files")

    if jp_dfs and en_dfs:
        batch_count = merge_translation_files(jp_dfs, en_dfs, tw_dfs)
        print(f"合併的數據已保存到 data/languages/merged.csv")
    else:
        print("Japanese and English lists are empty, cannot merge files")
