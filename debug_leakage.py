import pandas as pd
from sklearn.model_selection import train_test_split

# === fine_bert.ipynb approach ===
data1 = pd.read_csv('resources/cn7050data.csv', encoding='latin-1', names=['sentiment', 'text'])
data1 = data1.drop_duplicates().reset_index(drop=True)
data1['sentiment'] = data1['sentiment'].str.lower()
train1, temp1 = train_test_split(data1, test_size=0.2, random_state=13, stratify=data1['sentiment'])
_, test1 = train_test_split(temp1, test_size=0.5, random_state=13, stratify=temp1['sentiment'])
print('fine_bert test size:', len(test1))

# === gptoss.ipynb approach (the cell that saves to CSV uses random_state=42) ===
data2 = pd.read_csv('resources/cn7050data.csv', encoding='latin-1', names=['sentiment', 'text'])
data2 = data2.drop_duplicates().reset_index(drop=True).dropna()
data2['sentiment'] = data2['sentiment'].str.lower()
data2['text'] = data2['text'].str.replace(r'[^\x00-\x7F]+', '[UNK]', regex=True)
train2, temp2 = train_test_split(data2, test_size=0.20, random_state=42, stratify=data2['sentiment'])
_, test2 = train_test_split(temp2, test_size=0.50, random_state=42, stratify=temp2['sentiment'])
gpt_80 = test2.sample(n=80, random_state=13)
print('gptoss test size:', len(test2))

# Check: How many of the 80 GPT test samples leaked into BERT's TRAINING set?
bert_train_texts = set(train1['text'].tolist())
bert_test_texts = set(test1['text'].tolist())
gpt_test_texts = set(gpt_80['text'].tolist())
leaked = gpt_test_texts & bert_train_texts
print(f'GPT 80 in BERT TRAIN set (DATA LEAKAGE): {len(leaked)} / 80')
print(f'GPT 80 in BERT TEST set (clean overlap): {len(gpt_test_texts & bert_test_texts)} / 80')

# Also check the actual saved CSV
gpt_csv = pd.read_csv('resources/gpt_oss_results.csv')
csv_texts = set(gpt_csv['text'].tolist())
leaked_csv = csv_texts & bert_train_texts
print(f'\ngpt_oss_results.csv in BERT TRAIN set (DATA LEAKAGE): {len(leaked_csv)} / {len(gpt_csv)}')
print(f'gpt_oss_results.csv in BERT TEST set: {len(csv_texts & bert_test_texts)} / {len(gpt_csv)}')

# Check how many NaN rows exist
data_raw = pd.read_csv('resources/cn7050data.csv', encoding='latin-1', names=['sentiment', 'text'])
print(f'\nRaw CSV rows: {len(data_raw)}')
print(f'After drop_duplicates only: {len(data_raw.drop_duplicates())}')
print(f'After drop_duplicates + dropna: {len(data_raw.drop_duplicates().dropna())}')
