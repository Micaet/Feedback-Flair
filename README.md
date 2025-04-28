# Feedback Flair

## Project Overview
The goal of this project is to develop a machine learning model capable of correctly classifying user ratings (positive, neutral, negative) given to ChatGPT based on the accompanying comment. 

## Business Objective
While English ChatGPT reviews are used as a sample dataset, the model can be generalised to some
arbitrary product $P$. The model could be used to analyse reviews and comments left by $P$ users in order to 
assess the sentiment regarding new updates, features, changes, etc. Primarily, the model could be used
to identify negative comments, hence highlighting areas in need of improvement
and helping businesses enhance their products and customer satisfaction.

## Assessment
The $F_{2}$ score has been chosen as the evaluation metric for our project because the main goal of the model is to accurately identify negative reviews. In this context, missing a negative review is considered more detrimental than incorrectly flagging a non-negative review as negative. The $F_{2}$ score emphasizes recall more than precision, making it a suitable choice for this use case. The results of the model ($F_{2}$ along other supplementary metrics) can be found in [**results.txt**](results.txt).

## Stages
The project is divided into three stages, each of which is elaborately described in the corresponding notebook:
1. **Data Preprocessing, Cleaning and Exploration**
2. **Feature Engineering and Preliminary Modeling**
3. **Final Models, Presentation of Results, and Report**

Each stage has been validated by an independent group, and their feedback has been incorporated as improvements.

The specific tools, libraries, and frameworks used throughout the project are documented in a separate file called [**requirements.txt**](requirements.txt).

## Notebooks

The repository contains notebooks outlining our reasoning. To reproduce our process, run them in the following order:

**Stage 1**
1. [**budowa_krok_1_6_preprocessing.ipynb**](budowa_krok_1_6_preprocessing.ipynb)
2. [**budowa_krok_1_6_eda.ipynb**](budowa_krok_1_6_eda.ipynb)

**Stage 2**
1. [**budowa_krok_2_6_feature_engineering.ipynb**](budowa_krok_2_6_feature_engineering.ipynb)
2. [**budowa_krok_2_6_preliminary_modelling.ipynb**](budowa_krok_2_6_preliminary_modelling.ipynb)

**Stage 3**
1. [**budowa_krok_3_6_fine_tuning.ipynb**](budowa_krok_3_6_fine_tuning.ipynb)

## Pipelines

The repository contains two pipelines:

1. **data_processing.joblib** <br>
which utilises [**custom_transformers.py**](custom_transformers.py) to preprocess and clean the data, and engineer the labels
using an NLP model.
2. **ensemble.joblib** <br>
which applies an ensemble model.

Both pipelines are created in [**build_pipeline.py**](build_pipeline.py).

## Model Training and Assessment

The model is trained and assessed in [**train_model.py**](train_model.py), which loads the above pipelines and then saves
the results to [**results.txt**](results.txt).

## Python version

The model was created using Python 3.12.2.

## Setup using *nix and Conda

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Micaet/Feedback-Flair
   cd Feedback-Flair
   ```
2. **Install dependencies**
   ```bash
   conda create -n feedback-flair python=3.12.2
   conda activate feedback-flair
   pip install -r requirements.txt
   ```
3. **Run set-up script (needed only once) to download necessary resources for NLP**
   ```bash
   python setup.py
   ```

   The script includes:
   ```python
   import nltk
   from spacy.cli import download

   nltk.download("stopwords")
   nltk.download("punkt")
   nltk.download("punkt_tab")
   download("en_core_web_sm")
   ```
4. **Train and assess the model**
   ```bash
   python train_model.py
   cat results.txt
   ```

5. **Or feel free to explore the notebooks:** <br>

+ [**budowa_krok_1_6_preprocessing.ipynb**](budowa_krok_1_6_preprocessing.ipynb)
+ [**budowa_krok_1_6_eda.ipynb**](budowa_krok_1_6_eda.ipynb)
+ [**budowa_krok_2_6_feature_engineering.ipynb**](budowa_krok_2_6_feature_engineering.ipynb)
+ [**budowa_krok_2_6_preliminary_modelling.ipynb**](budowa_krok_2_6_preliminary_modelling.ipynb)
+ [**budowa_krok_3_6_fine_tuning.ipynb**](budowa_krok_3_6_fine_tuning.ipynb)

## Dataset
The dataset used for this project is a collection of ChatGPT user reviews and ratings. It can be found at the following link:
[ChatGPT User Reviews Dataset](https://www.kaggle.com/datasets/anandshaw2001/chatgpt-users-reviews)

## Authors
- Antoni Rakowski
- Michał Syrkiewicz
