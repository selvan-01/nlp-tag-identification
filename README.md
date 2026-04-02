# 📌 Tag Identification using NLP

## 🚀 Project Overview
This project demonstrates a complete **Natural Language Processing (NLP) pipeline** using the SMS Spam dataset. It converts raw text into clean, structured data for machine learning applications.

---

## 📂 Dataset
- Dataset: `SMSSpamCollection.tsv`
- Columns:
  - `label` → Spam / Ham
  - `body_text` → SMS message content

---

## ⚙️ Features
- Data Loading  
- Remove Punctuation  
- Tokenization  
- Stopwords Removal  
- Stemming  
- Lemmatization  

---

## 🔄 NLP Pipeline
1. Read Dataset  
2. Remove Punctuation  
3. Convert to Lowercase  
4. Tokenize Text  
5. Remove Stopwords  
6. Apply Stemming  
7. Apply Lemmatization  

---

## 🛠️ Technologies Used
- Python  
- Pandas  
- NLTK  
- Regular Expressions (re)  

---

## 📁 Project Structure
NLP-Tag-Identification/
│── Read the data.py  
│── Remove Punctuations.py  
│── Tokenization.py  
│── Remove Stopwords.py  
│── Stemming.py  
│── Lemmatization.py  
│── SMSSpamCollection.tsv  
│── README.md  
│── requirements.txt  

---

## ▶️ How to Run

### 1. Clone the Repository
git clone https://github.com/selvan-01/nlp-tag-identification.git  
cd nlp-tag-identification  

### 2. Install Requirements
pip install -r requirements.txt  

### 3. Run the Project
python lemmatization.py  

---

## ⚠️ Setup (Run Once)
import nltk  
nltk.download('stopwords')  
nltk.download('wordnet')  
nltk.download('omw-1.4')  

---

## 📌 Output
- Cleaned Text  
- Tokenized Words  
- Stopwords Removed  
- Stemmed Words  
- Lemmatized Words  

---

## 🎯 Learning Outcomes
- Understanding NLP preprocessing pipeline  
- Hands-on experience with NLTK  
- Text cleaning & normalization  
- Preparing data for ML models  

---

## 🚀 Future Improvements
- Add TF-IDF / CountVectorizer  
- Train ML models (Naive Bayes / Logistic Regression)  
- Build Web App (Flask / Streamlit)  

---
## 🔗 Links

- 💼 [LinkedIn](https://www.linkedin.com/in/senthamil45)
- 🌍 [Portfolio](https://senthamill.vercel.app/)
- 💻 [GitHub](https://github.com/selvan-01/nlp-tag-identification.git)

---
## 💡 Conclusion
This project builds a strong foundation in NLP preprocessing and prepares data for intelligent text classification systems.

---

## ⭐ Support
If you like this project, give it a ⭐ on GitHub!
