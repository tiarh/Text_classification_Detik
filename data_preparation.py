import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re


def prepare_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df

def preprocess_data(df):
    df['text'] = df['isi'].apply(preprocess_text)
    return df

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca
    text = re.sub(r'\d', '', text)  # Hapus angka
    text = ' '.join(text.split())  # Satukan teks
    return text

def extract_tfidf_features(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text'])
    return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
