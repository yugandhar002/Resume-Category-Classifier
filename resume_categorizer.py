import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import re

def clean_text(text):
    # Remove URLs
    clean = re.sub('http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions and hashtags
    clean = re.sub(r'\@\w+|\#','', clean)
    # Remove digits
    clean = re.sub(r'\d+', '', clean)
    # Remove punctuation
    clean = re.sub(r'[^\w\s]', '', clean)
    # Remove all special characters and numbers
    clean = re.sub(r'[^a-zA-Z\s]', '', clean)
    # Remove extra spaces and newlines
    clean = re.sub(r'\s+', ' ', clean)
    clean = clean.strip()
    return clean

# Load the dataset
df = pd.read_csv('UpdatedResumeDataSet.csv')

# Clean the resume text
df['Resume'] = df['Resume'].apply(clean_text)

# Encode the categories
le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(df['Resume'])
y = df['Category']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print results
print('Accuracy:', accuracy_score(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Print category mapping
print('\nCategory Mapping:')
for i, category in enumerate(le.classes_):
    print(f'{category}: {i}')
