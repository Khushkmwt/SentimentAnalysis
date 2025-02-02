# 🐦 Twitter Sentiment Analysis

## 📌 Project Overview
This project performs sentiment analysis on Twitter data using the **Sentiment140 dataset**. A **Logistic Regression model** is trained with **TF-IDF vectorization** to classify tweets as either **positive** or **negative**.

---
## 📂 Dataset Information
🔹 **Source**: [Sentiment140 Kaggle Dataset](https://www.kaggle.com/kazanova/sentiment140)  
🔹 **Description**:
- **0** → Negative sentiment 😠
- **4** → Positive sentiment 😊  

🔹 **Preprocessing Steps**:
✅ Remove special characters & non-alphabetic content  
✅ Convert text to lowercase  
✅ Remove stopwords  
✅ Apply stemming using the **Porter Stemmer** algorithm  

---
## 🛠 Installation & Setup
### 1️⃣ Install Required Dependencies:
```bash
pip install kaggle pandas numpy scikit-learn nltk matplotlib
```
### 2️⃣ Download & Extract the Dataset:
```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download kazanova/sentiment140
from zipfile import ZipFile
with ZipFile("sentiment140.zip", 'r') as zip_ref:
    zip_ref.extractall()
```

---
## 🧹 Data Preprocessing
```python
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')

# Load dataset
twitter_data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1')

# Preprocessing function
port_stem = PorterStemmer()
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    text = [port_stem.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

twitter_data['processed_text'] = twitter_data['text'].apply(preprocess_text)
```

---
## 🚀 Model Training
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Prepare dataset
X = twitter_data['processed_text'].values
y = twitter_data['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Feature extraction
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

---
## 📊 Model Evaluation
```python
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
```
✅ **Training Accuracy:** 🎯 **0.7987 (79.87%)**  
✅ **Testing Accuracy:** 🎯 **0.7767 (77.67%)**  

---
## 💾 Save & Load the Model
```python
import pickle

# Save the model
filename = 'sentiment_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Load the model
loaded_model = pickle.load(open('sentiment_model.sav', 'rb'))
```

---
## 📝 Sentiment Prediction
```python
X_sample = X_test[200]
y_actual = y_test[200]
prediction = loaded_model.predict(X_sample)
print("Actual Sentiment:", "Positive" if y_actual == 4 else "Negative")
print("Predicted Sentiment:", "Positive" if prediction[0] == 4 else "Negative")
```

---
## 🎯 Conclusion
This project successfully demonstrates **sentiment analysis** on Twitter data using **Logistic Regression** and **TF-IDF vectorization**. The model achieves:
- ✅ **79.87% Training Accuracy** 🏆
- ✅ **77.67% Testing Accuracy** 📊

A great balance between **accuracy** and **efficiency**, making it suitable for **large-scale text classification tasks**. 🚀

