
# 🎬 Sentiment Analysis of Movie Reviews

This project implements a sentiment analysis model that classifies movie reviews as **positive** or **negative** using a vectorization approach. 

---

## 📌 Project Overview

The project consists of:

- **Text preprocessing** using tokenization, stopword removal, and lemmatization
- **Vectorization** using TF-IDF to convert text into numerical features
- **Model training** using Logistic Regression
- **Review testing** using a separate dataset of 1,000 reviews from a forked repo

The trained model can classify unseen movie reviews into positive or negative sentiment with reasonably high accuracy.

---

## 📂 Project Structure

```
AI-and-ML/
├── testFile.py      # Main module with training and testing functions
├── accuracyScript.py      # Script to evaluate model accuracy on test dataset
├── training_model.pkl         # Serialized trained model (Logistic Regression)
├── vectorizer.pkl             # Serialized TF-IDF vectorizer
├── test_reviews.json          # 1,000 IMDb reviews from the forked repo
└── README.md                  # Project documentation
```

---

## 🔧 How It Works

### Preprocessing

1. Lowercases and removes punctuation
2. Tokenizes text into words
3. Removes stopwords (e.g., "the", "and")
4. Lemmatizes remaining words (e.g., "running" → "run")

### Training (`calcSentiment_train`)
- Loads a labeled `.json` training file
- Applies preprocessing
- Vectorizes the data using `TfidfVectorizer`
- Trains a `LogisticRegression` model
- Saves both the model and vectorizer to disk

### Testing (`calcSentiment_test`)
- Loads trained model and vectorizer
- Preprocesses a single review
- Predicts the sentiment (returns `True` for positive, `False` for negative)

---

## 🧪 Dataset

For evaluation, 1,000 movie reviews from the [nas5w/imdb-data](https://github.com/nas5w/imdb-data) repository were used. These reviews were extracted from a JSON file and tested using the model developed in this project.

---

## 🚀 How to Run

1. Clone the repo:
```bash
git clone https://github.com/joshmaha/Sentiment-Analysis.git
cd Sentiment-Analysis
```

2. Train the model:
```python
from testFile import calcSentiment_train
calcSentiment_train("train.json")
```

3. Test a single review:
```python
from TestFile import calcSentiment_test
print(calcSentiment_test("This movie was absolutely wonderful!"))
```

---

## 📦 Dependencies

- `nltk`
- `scikit-learn`
- `joblib`

Also make sure to download NLTK resources if not already present:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

