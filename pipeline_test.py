from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import pandas as pd
import emoji
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from symspellpy import SymSpell, Verbosity
from sklearn.model_selection import train_test_split
import swifter
import spacy
from spacy.cli import download
from langdetect import detect
import importlib.resources
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

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

    def correct_text(self, text):
        if isinstance(text, str):
            suggestion = self.sym_spell.lookup_compound(text, max_edit_distance=2)
            return suggestion[0].term if suggestion else text
        return text

    def remove_punctuation(self, text):
        return re.sub(r'[^\w\s]', '', text)

    def remove_stopwords(self, text):
        words = word_tokenize(text)
        words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(words)

    def lemmatize_text(self, text):
        doc = self.nlp(text)
        return ' '.join([token.lemma_ for token in doc])

    def preprocess(self, text):
        if not isinstance(text, str) or not self.is_english(text):
            return ""
        text = self.convert_emojis_to_text(text)
        text = text.lower()
        text = self.correct_text(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize_text(text)
        text = self.remove_punctuation(text)
        return text

    def transform(self, X, y=None):
        return X.swifter.apply(self.preprocess)

    def fit(self, X, y=None):
        return self

class DropEmptyText(BaseEstimator, TransformerMixin):
    def transform(self, X, y=None):
        # Remove empty or whitespace-only strings
        return X[X.str.strip().astype(bool)].reset_index(drop=True)

    def fit(self, X, y=None):
        return self

from sklearn.pipeline import Pipeline
import pandas as pd
from symspellpy import SymSpell
import importlib.resources
import spacy
from nltk.corpus import stopwords
import nltk
import swifter


# Load SymSpell
dictionary_path = importlib.resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary(str(dictionary_path), term_index=0, count_index=1)

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Stopwords
stop_words = set(stopwords.words('english'))

# Sample DataFrame
df = pd.DataFrame({'Review': ["I loooove this soooo much!!! 🤩😱🥶", "C'est magnifique", "Thiss is awfull.. but ok👽☠️"]})

# Define pipeline
text_pipeline = Pipeline([
    ('full_preprocessing', FullTextPreprocessor(sym_spell=sym_spell, nlp=nlp, stop_words=stop_words)),
    ('drop_empty', DropEmptyText())
])

# Run pipeline
processed_text = text_pipeline.fit_transform(df['Review'])
print(processed_text)
