# if corupusTrain.csv exists, then it will be loaded
# if not, then it will be created

import os
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import RegexpStemmer
from nltk.stem import RSLPStemmer
from nltk.stem import ISRIStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')


def load_data():
    if os.path.exists('corpusTrain.csv'):
        return pd.read_csv('corpusTrain.csv')
    else:
        return None


def save_data(data):
    data.to_csv('corpusTrain.csv', index=False)


def get_data():

