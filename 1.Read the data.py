# -----------------------------------------------
# 📌 Project: Tag Identification using NLP
# 📁 File: Read the data.py
# 📖 Description:
# This script loads the SMS Spam dataset and 
# displays the first few rows for inspection.
# -----------------------------------------------

# Import required libraries
import pandas as pd      # For data handling and analysis
import nltk              # Natural Language Toolkit (for NLP tasks)

# -----------------------------------------------
# 📌 Step 1: Load the dataset
# -----------------------------------------------
# The dataset is in TSV format (Tab-Separated Values)
# It contains two columns:
#   1. label (spam/ham)
#   2. body_text (message content)

data = pd.read_csv(
    'SMSSpamCollection.tsv',   # File path
    sep='\t',                  # Tab separator
    names=['label', 'body_text'],  # Column names
    header=None               # No header in original file
)

# -----------------------------------------------
# 📌 Step 2: Display the dataset
# -----------------------------------------------
# Print the first 5 rows to understand the structure

print("📊 First 5 rows of the dataset:\n")
print(data.head())

# -----------------------------------------------
# 📌 Optional (Commented Code)
# -----------------------------------------------
# You can also read raw data manually like this:

# rawdata = open("SMSSpamCollection.tsv").read()
# print(rawdata[0:250])  # Print first 250 characters