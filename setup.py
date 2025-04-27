import nltk
from spacy.cli import download

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
download("en_core_web_sm")