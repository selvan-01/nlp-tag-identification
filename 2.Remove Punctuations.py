# -----------------------------------------------
# 📌 Project: Tag Identification using NLP
# 📁 File: Remove Punctuations.py
# 📖 Description:
# This script removes punctuation from the text
# data in the SMS Spam dataset.
# -----------------------------------------------

# Import required libraries
import pandas as pd      # For data handling
import nltk              # NLP library (optional here)
import string            # Contains punctuation symbols

# -----------------------------------------------
# 📌 Step 1: Load the dataset
# -----------------------------------------------
data = pd.read_csv(
    'SMSSpamCollection.tsv',   # Dataset file
    sep='\t',                  # Tab-separated values
    names=['label', 'body_text'],  # Column names
    header=None               # No header in dataset
)

# Display initial data
print("📊 Original Data:\n")
print(data.head())

# -----------------------------------------------
# 📌 Step 2: Remove punctuation
# -----------------------------------------------
# string.punctuation contains all punctuation symbols
# Example: ! " # $ % & ' ( ) * + , - . / : ; < = > ? @ ...

def remove_punct(text):
    """
    This function removes punctuation from a given text.
    """
    text_nopunct = "".join(
        [char for char in text if char not in string.punctuation]
    )
    return text_nopunct

# Apply function to 'body_text' column
data['body_text_clean'] = data['body_text'].apply(lambda x: remove_punct(x))

# -----------------------------------------------
# 📌 Step 3: Display cleaned data
# -----------------------------------------------
print("\n🧹 Data after removing punctuation:\n")
print(data.head())

# -----------------------------------------------
# 📌 Optional (Debug / Understanding)
# -----------------------------------------------
# Print all punctuation characters
# print(string.punctuation)