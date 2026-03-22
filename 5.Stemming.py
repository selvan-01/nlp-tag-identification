# -----------------------------------------------
# 📌 Project: Tag Identification using NLP
# 📁 File: Stemming.py
# 📖 Description:
# This script performs:
# 1. Data loading
# 2. Removing punctuation
# 3. Tokenization
# 4. Removing stopwords
# 5. Stemming using PorterStemmer
# -----------------------------------------------

# Import required libraries
import pandas as pd
import nltk
import string
import re

# -----------------------------------------------
# 📌 Step 0: Download required resources (run once)
# -----------------------------------------------
# nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# -----------------------------------------------
# 📌 Step 1: Load the dataset
# -----------------------------------------------
data = pd.read_csv(
    'SMSSpamCollection.tsv',
    sep='\t',
    names=['label', 'body_text'],
    header=None
)

print("📊 Original Data:\n")
print(data.head())

# -----------------------------------------------
# 📌 Step 2: Remove punctuation
# -----------------------------------------------
def remove_punct(text):
    """
    Removes punctuation from text.
    """
    return "".join([char for char in text if char not in string.punctuation])

data['body_text_clean'] = data['body_text'].apply(remove_punct)

print("\n🧹 After Removing Punctuation:\n")
print(data.head())

# -----------------------------------------------
# 📌 Step 3: Tokenization
# -----------------------------------------------
def tokenize(text):
    """
    Splits text into tokens using regex.
    """
    return re.split(r'\W+', text)

data['body_text_tokenized'] = data['body_text_clean'].apply(
    lambda x: tokenize(x.lower())
)

print("\n🔍 After Tokenization:\n")
print(data.head())

# -----------------------------------------------
# 📌 Step 4: Remove Stopwords
# -----------------------------------------------
stop_words = set(stopwords.words('english'))

def remove_stopwords(tokenized_list):
    """
    Removes stopwords from tokenized text.
    """
    return [word for word in tokenized_list if word not in stop_words and word != ""]

data['body_text_nostop'] = data['body_text_tokenized'].apply(remove_stopwords)

print("\n🚫 After Removing Stopwords:\n")
print(data.head())

# -----------------------------------------------
# 📌 Step 5: Stemming
# -----------------------------------------------
ps = PorterStemmer()

def stemming(tokenized_text):
    """
    Applies stemming to each word using PorterStemmer.
    Example: running → run
    """
    return [ps.stem(word) for word in tokenized_text]

data['body_text_stemmed'] = data['body_text_nostop'].apply(stemming)

print("\n🌱 After Stemming:\n")
print(data.head())

# -----------------------------------------------
# 📌 Optional (Raw Data View)
# -----------------------------------------------
# rawdata = open("SMSSpamCollection.tsv").read()
# print(rawdata[:250])