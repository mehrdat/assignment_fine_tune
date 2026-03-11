import pandas as pd
from sklearn.model_selection import train_test_split

gpt_csv = pd.read_csv('resources/gpt_oss_results.csv')

data = pd.read_csv('resources/cn7050data.csv', encoding='latin-1', names=['sentiment', 'text'])
data = data.drop_duplicates().reset_index(drop=True).dropna()
data['sentiment'] = data['sentiment'].str.lower()
data['text'] = data['text'].str.replace(r'[^\x00-\x7F]+', '[UNK]', regex=True)

train13, temp13 = train_test_split(data, test_size=0.20, random_state=13, stratify=data['sentiment'])
_, test13 = train_test_split(temp13, test_size=0.50, random_state=13, stratify=temp13['sentiment'])
gpt80_13 = test13.sample(n=80, random_state=13)

train42, temp42 = train_test_split(data, test_size=0.20, random_state=42, stratify=data['sentiment'])
_, test42 = train_test_split(temp42, test_size=0.50, random_state=42, stratify=temp42['sentiment'])
gpt80_42 = test42.sample(n=80, random_state=13)

csv_texts = set(gpt_csv['text'].tolist())
texts_13 = set(gpt80_13['text'].tolist())
texts_42 = set(gpt80_42['text'].tolist())

print(f"CSV match with state=13: {len(csv_texts & texts_13)}/80")
print(f"CSV match with state=42: {len(csv_texts & texts_42)}/80")

# Also check leakage for state=13 scenario
bert_train_13 = set(train13['text'].tolist())
print(f"\nWith state=13: GPT 80 in BERT train set: {len(texts_13 & bert_train_13)}/80")
print(f"With state=42: GPT 80 in BERT train set (state=13): {len(texts_42 & bert_train_13)}/80")
