# -----------------------------------------------
# 📌 Project: Tag Identification using NLP
# 📁 File: Remove Stopwords.py
# 📖 Description:
# This script performs:
# 1. Data loading
# 2. Removing punctuation
# 3. Tokenization
# 4. Removing stopwords
# -----------------------------------------------

# Import required libraries
import pandas as pd        # Data handling
import nltk                # NLP library
import string              # Punctuation handling
import re                  # Regex for tokenization

# -----------------------------------------------
# 📌 Step 0: Download stopwords (run once)
# -----------------------------------------------
# Uncomment this line if stopwords are not downloaded
# nltk.download('stopwords')

from nltk.corpus import stopwords

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
# Load English stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(tokenized_list):
    """
    Removes common English stopwords from tokenized text.
    """
    return [word for word in tokenized_list if word not in stop_words and word != ""]

data['body_text_nostop'] = data['body_text_tokenized'].apply(remove_stopwords)

print("\n🚫 After Removing Stopwords:\n")
print(data.head())

# -----------------------------------------------
# 📌 Optional (Raw Data View)
# -----------------------------------------------
# rawdata = open("SMSSpamCollection.tsv").read()
# print(rawdata[:250])