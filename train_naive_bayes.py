import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import load_dataset

# Load and prepare data
df = load_dataset()
if df is None:
    exit()

X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Naive Bayes
print("Training Naive Bayes...")
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

y_pred = nb_model.predict(X_test_tfidf)
print("\nNaive Bayes Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# Save model
with open('naive_bayes_model.pkl', 'wb') as f:
    pickle.dump({'model': nb_model, 'vectorizer': tfidf}, f)
print("Naive Bayes model saved to naive_bayes_model.pkl")