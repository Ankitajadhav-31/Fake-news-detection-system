import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
from flask import Flask, request, render_template, jsonify
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import datetime

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

app = Flask(__name__)

# Initialize the stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

fake_words = ['fake', 'hoax', 'conspiracy', 'scam', 'fraud', 'misleading', 'unverified', 'clickbait']

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove stopwords and stem
    words = nltk.word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

def train_model():
    # Load the datasets
    true_df = pd.read_csv('True.csv')
    fake_df = pd.read_csv('Fake.csv')
    
    # Add labels
    true_df['label'] = 1
    fake_df['label'] = 0
    
    # Combine datasets
    df = pd.concat([true_df, fake_df])
    
    # Preprocess the text
    df['text'] = df['text'].apply(preprocess_text)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )
    
    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)
    
    # Initialize and train the model
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)
    
    # Save the model and vectorizer
    with open('model.pkl', 'wb') as f:
        pickle.dump(pac, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    
    # Evaluate the model
    y_pred = pac.predict(tfidf_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy:.2f}')
    
    return accuracy

# Load the model and vectorizer
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except:
    print("Training new model...")
    train_model()
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

def analyze_date(date_str):
    try:
        date = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        today = datetime.datetime.now()
        if date > today:
            return 0  # Future date
        if (today - date).days > 365:
            return 0  # Older than a year
        return 1  # Valid date
    except:
        return 0  # Invalid date

def analyze_source(subject):
    reputable_sources = ['politics', 'worldnews', 'news']  # Example list
    return 1 if subject.lower() in reputable_sources else 0

def highlight_fake_words(text):
    words = text.lower().split()
    highlighted_words = [word for word in words if word in fake_words]
    return highlighted_words

@app.route('/predict', methods=['POST'])
def predict():
    text = request.json['text']
    date = request.json.get('date', '')
    subject = request.json.get('subject', '')
    
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]
    probability = model.decision_function(vectorized_text)[0]
    
    date_score = analyze_date(date)
    source_score = analyze_source(subject)
    highlighted_words = highlight_fake_words(text)
    
    result = {
        'prediction': 'REAL' if prediction == 1 else 'FAKE',
        'confidence': float(abs(probability)),
        'date_score': date_score,
        'source_score': source_score,
        'highlighted_words': highlighted_words
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
