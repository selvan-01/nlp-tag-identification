# -----------------------------------------------
# 📌 Project: Tag Identification using NLP
# 📁 File: Tokenization.py
# 📖 Description:
# This script performs:
# 1. Data loading
# 2. Removing punctuation
# 3. Tokenizing text into words
# -----------------------------------------------

# Import required libraries
import pandas as pd       # Data handling
import nltk               # NLP library (optional)
import string             # Punctuation handling
import re                 # Regular expressions for tokenization

# -----------------------------------------------
# 📌 Step 1: Load the dataset
# -----------------------------------------------
data = pd.read_csv(
    'SMSSpamCollection.tsv',     # Dataset file
    sep='\t',                    # Tab-separated values
    names=['label', 'body_text'],  # Column names
    header=None                 # No header in dataset
)

print("📊 Original Data:\n")
print(data.head())

# -----------------------------------------------
# 📌 Step 2: Remove punctuation
# -----------------------------------------------
def remove_punct(text):
    """
    Removes punctuation from the input text.
    """
    text_nopunct = "".join(
        [char for char in text if char not in string.punctuation]
    )
    return text_nopunct

# Apply punctuation removal
data['body_text_clean'] = data['body_text'].apply(remove_punct)

print("\n🧹 After Removing Punctuation:\n")
print(data.head())

# -----------------------------------------------
# 📌 Step 3: Tokenization
# -----------------------------------------------
def tokenize(text):
    """
    Splits text into tokens (words) using regex.
    \W+ splits on non-word characters.
    """
    tokens = re.split(r'\W+', text)
    return tokens

# Convert to lowercase and tokenize
data['body_text_tokenized'] = data['body_text_clean'].apply(
    lambda x: tokenize(x.lower())
)

print("\n🔍 After Tokenization:\n")
print(data.head())

# -----------------------------------------------
# 📌 Optional (Raw Data View)
# -----------------------------------------------
# rawdata = open("SMSSpamCollection.tsv").read()
# print(rawdata[:250])