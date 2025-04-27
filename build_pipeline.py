from custom_transformers import *
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from symspellpy import SymSpell
import spacy
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import importlib.resources
import swifter
from nltk.sentiment import SentimentIntensityAnalyzer  
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.preprocessing import StandardScaler, FunctionTransformer, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from joblib import dump

# Load SymSpell
dictionary_path = importlib.resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary(str(dictionary_path), term_index=0, count_index=1)

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Stopwords
stop_words = set(stopwords.words('english'))

linear_model = LogisticRegression(
        penalty="l1",
        C=100,
        solver='saga',
        max_iter=5000,
        random_state=42,
        n_jobs=-1
    )

xg_model = XGBClassifier(eval_metric='mlogloss', random_state=42)

ensemble_model = VotingClassifier(
    estimators=[('lr', linear_model), ('xgb', xg_model)],
    voting='soft',
    weights=[0.9, 0.1],
    n_jobs=-1
)

tfidf_pca_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=3000)),
    ('scaler', StandardScaler(with_mean=False)),
    ('to_dense', FunctionTransformer(func=to_dense, accept_sparse=True)),
    ('pca', PCA(n_components=0.90)),
    ('min_max', MinMaxScaler())
])


feature_engineering = ColumnTransformer(transformers=[
    ('tfidf_pca', tfidf_pca_pipeline, 'Review'),
    ('sentiment_passthrough', 'passthrough', ['Sentiment'])
])

data_processing = Pipeline([
    ('data_cleaning', DataCleaner()),                  
    ('full_preprocessing', FullTextPreprocessor(sym_spell=sym_spell, nlp=nlp, stop_words=stop_words)),
    ('drop_empty_text', DropEmptyText()),
    ('add_sentiment', RatingSentimentTransformer()),
    ('tfidf_pca', feature_engineering)
    ])

dump(data_processing, 'data_processing.joblib')

ensemble = Pipeline([
    ('ensemble_model', ensemble_model)
])

dump(ensemble, 'ensemble.joblib')