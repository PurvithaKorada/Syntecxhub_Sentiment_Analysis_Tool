import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
# Preprocessing text data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
# Loading dataset
try:
    data = pd.read_csv("dataset.csv")
except FileNotFoundError:
    print("ERROR: dataset.csv file not found.")
    print("Make sure dataset.csv is in the same folder.")
    exit()
# Clean all text data
data["text"] = data["text"].apply(clean_text)
data = data.dropna()
# Input & output 
X = data["text"]
y = data["sentiment"]
# Convert text into numbers
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2)
)
X_vectorized = vectorizer.fit_transform(X)
# Splitting dataset for training/testing
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)
# Training  model
model = MultinomialNB()
model.fit(X_train, y_train)
# Model predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label="positive")
print("\n====================================")
print("   SENTIMENT ANALYSIS TOOL")
print("====================================")
print(f"\nModel Accuracy : {accuracy:.2f}")
print(f"F1 Score       : {f1:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\n====================================")
print(" Real-Time Sentiment Prediction")
print("====================================")
print("\nType a sentence to analyze sentiment.")
print("Type 'quit' to exit the program.\n")
while True:
    user_input = input("Enter Text: ")
    # Exit condition
    if user_input.lower() == "quit":
        print("\nExiting Sentiment Analysis Tool...")
        print("Thank you for using the program!")
        break
    # Empty input check
    if user_input.strip() == "":
        print("Please enter some text.\n")
        continue
    cleaned_input = clean_text(user_input)
    # Convert input txt
    user_vector = vectorizer.transform([cleaned_input])
    # Predict sentiment
    prediction = model.predict(user_vector)[0]
    probabilities = model.predict_proba(user_vector)[0]
    positive_prob = probabilities[list(model.classes_).index("positive")]
    negative_prob = probabilities[list(model.classes_).index("negative")]
    print("\nPredicted Sentiment:", prediction)
    print(f"Positive Confidence : {positive_prob:.2f}")
    print(f"Negative Confidence : {negative_prob:.2f}")

    if prediction == "positive":
        print("Emotion Detected: 😊 Positive")
    else:
        print("Emotion Detected: 😠 Negative")
    print()