import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import torch

def load_newsgroups_docs():
    news = fetch_20newsgroups(subset='all')
    vectorizer = CountVectorizer(max_df=0.5, min_df=20, stop_words='english')
    docs = torch.from_numpy(vectorizer.fit_transform(news['data']).toarray())
    vocab = pd.DataFrame(columns=['word', 'index'])
    vocab['word'] = vectorizer.get_feature_names_out()
    vocab['index'] = vocab.index
    return docs, vocab


