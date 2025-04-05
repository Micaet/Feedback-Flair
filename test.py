import emoji
from sklearn.preprocessing import FunctionTransformer 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import nltk
from nltk.corpus import stopwords

def convert_emojis_to_text(text):
    return emoji.demojize(text, delimiters=(" ", " "))

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

transformer = FunctionTransformer(convert_emojis_to_text)

transformer2 = FunctionTransformer(remove_stopwords)

preprocessor = ColumnTransformer(
    transformers=[
        ("remove_emojis", transformer, "Review"),  # Apply transformer to specific column
        ("remove_stop_words", transformer2, "Review")  # Apply transformer to another column
    ]
)

pipe = Pipeline([
    ("preprocessor", preprocessor)
])

df = pd.read_csv("ChatGPT_Reviews.csv")
pipe.fit(df)
print(df.head())