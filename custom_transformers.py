from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import emoji
import re
from symspellpy import SymSpell
import swifter
import spacy
from langdetect import detect
from nltk.sentiment import SentimentIntensityAnalyzer  
from nltk.tokenize import word_tokenize

class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_cleaned = X.drop_duplicates(subset='Review Id', keep='first').dropna()[["Review"]]
        return X_cleaned

class FullTextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, sym_spell, nlp, stop_words):
        self.sym_spell = sym_spell
        self.nlp = nlp
        self.stop_words = stop_words

    def is_english(self, text):
        try:
            return detect(text) == "en"
        except:
            return False

    def convert_emojis_to_text(self, text):
        return emoji.demojize(text, delimiters=(" ", " "))

    def to_lower(self, text):
        return text.lower()

    def correct_text(self, text):
        if isinstance(text, str):
            suggestion = self.sym_spell.lookup_compound(text, max_edit_distance=2)
            return suggestion[0].term if suggestion else text
        return text

    def remove_stopwords(self, text):
        words = word_tokenize(text)
        words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(words)

    def lemmatize_text(self, text):
        doc = self.nlp(text)
        return ' '.join([token.lemma_ for token in doc])

    def remove_punctuation(self, text):
        return re.sub(r'[^\w\s]', '', text)

    def preprocess(self, text):
        if not isinstance(text, str) or not self.is_english(text):
            return ""
        text = self.convert_emojis_to_text(text)
        text = self.to_lower(text)
        text = self.correct_text(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize_text(text)
        text = self.remove_punctuation(text)
        return text

    def transform(self, X, y=None):
        X["Review"] = X["Review"].swifter.apply(self.preprocess)
        return X

    def fit(self, X, y=None):
        return self

class DropEmptyText(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[X["Review"].str.strip().astype(bool)].reset_index(drop=True)


class RatingSentimentTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sentiments = X["Review"].apply(lambda x: self.analyzer.polarity_scores(str(x))['compound']).apply(self.to_discrete)
        print(type(sentiments))
        df = pd.concat([X, sentiments], axis = 1)
        df.columns = ["Review", "Sentiment"]
        return df

    def to_discrete(self, score):
        if score >= 0.05:
            return 2
        elif score <= -0.05:
            return 0
        else:
            return 1

def to_dense(x):
    return x.toarray() if hasattr(x, 'toarray') else x