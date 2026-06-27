import re
import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return ' '.join(word for word in text.split() if word not in stop_words)

def load_dataset(true_path='data/True.csv', fake_path='data/Fake.csv'):
    try:
        true_df = pd.read_csv(true_path, encoding='latin1')
        fake_df = pd.read_csv(fake_path, encoding='latin1')
        true_df['label'] = 1
        fake_df['label'] = 0
        df = pd.concat([true_df, fake_df], ignore_index=True)
        df['text'] = (df['title'] + ' ' + df['text']).apply(clean_text)
        return df[['text', 'label']]
    except FileNotFoundError:
        print("Dataset files not found in 'data/' folder.")
        return None

if __name__ == "__main__":
    data = load_dataset()