# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import the dataset and preprocess the text data — remove punctuation, convert to lowercase, and split into words.

2.Convert the processed text into numerical form using CountVectorizer or TfidfVectorizer.

3.Split the data into training and testing sets, then train an SVM (Support Vector Machine) classifier on the training data.

4.Predict the labels for the test set and evaluate performance using confusion matrix, accuracy score, and visualization (matplotlib/seaborn).

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: R.Sairam
RegisterNumber:  25000694
*/
```
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import seaborn as sns

import matplotlib.pyplot as plt

data = pd.read_csv("spam.csv",encoding='latin-1')

data = data[['v1', 'v2']]

data.columns = ['label', 'message']

data['label'] = data['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer()

X_train_features = vectorizer.fit_transform(X_train)

X_test_features = vectorizer.transform(X_test)

model = SVC(kernel='linear')

model.fit(X_train_features, y_train)

y_pred = model.predict(X_test_features)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))

sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])

plt.title("SVM Spam Mail Detection - Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()



## Output:
<img width="508" height="338" alt="image" src="https://github.com/user-attachments/assets/b1543a0c-d6df-450b-b889-a69804ad24bc" />

<img width="816" height="585" alt="image" src="https://github.com/user-attachments/assets/823d4ff3-d01f-4f6f-8610-f481ad88d20d" />




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
