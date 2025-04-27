from joblib import load, dump
from sklearn.model_selection import train_test_split
import pandas as pd
from joblib import load
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_score, recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import warnings

warnings.filterwarnings("ignore")

data_processing = load("data_processing.joblib")
ensemble = load("ensemble.joblib")

df = pd.read_csv("ChatGPT_Reviews.csv")

df = data_processing.fit_transform(df)
train_df, test_df = train_test_split(df, test_size = 0.3, random_state = 2092)
X_train = train_df[:,:-1]
y_train = train_df[:,-1]
X_test = test_df[:,:-1]
y_test = test_df[:,-1]

ensemble.fit(X_train, y_train)

# Open results file
with open('results.txt', 'w') as f:

    # Test set evaluation
    preds = ensemble.predict(X_test)
    prob_preds = ensemble.predict_proba(X_test)
    
    f.write("=== TEST SET RESULTS ===\n")
    f.write(f"F-beta score per class on test: {fbeta_score(y_test, preds, beta=2, average=None)}\n")
    f.write(f"Precision per class on test: {precision_score(y_test, preds, average=None)}\n")
    f.write(f"Recall per class on test: {recall_score(y_test, preds, average=None)}\n")
    f.write(f"AUC-ROC per class on test: {roc_auc_score(y_test, prob_preds, average=None, multi_class='ovr')}\n")
    f.write(f"Confusion matrix (test):\n{confusion_matrix(y_test, preds)}\n\n")
    
    # Train set evaluation
    preds_train = ensemble.predict(X_train)
    prob_preds_train = ensemble.predict_proba(X_train)
    
    f.write("=== TRAIN SET RESULTS ===\n")
    f.write(f"F-beta score per class on train: {fbeta_score(y_train, preds_train, beta=2, average=None)}\n")
    f.write(f"Precision per class on train: {precision_score(y_train, preds_train, average=None)}\n")
    f.write(f"Recall per class on train: {recall_score(y_train, preds_train, average=None)}\n")
    f.write(f"AUC-ROC per class on train: {roc_auc_score(y_train, prob_preds_train, average=None, multi_class='ovr')}\n")
    f.write(f"Confusion matrix (train):\n{confusion_matrix(y_train, preds_train)}\n")

print("Results saved to results.txt")