import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import joblib

# nltk.download("stopwords")

df = pd.read_csv("spam.csv", sep="\t", header=None, names=["label", "message"])
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))
vectorizer = TfidfVectorizer()


def clean_text(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    filtered = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(filtered)


df["cleaned_message"] = df["message"].apply(clean_text)

X = vectorizer.fit_transform(df["cleaned_message"])
Y = df["label_num"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, Y_train)


def predict_message(message):
    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return "SPAM" if prediction == 1 else "HAM"


# Y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(Y_test, Y_pred))
# print("\nClassification Report:\n", classification_report(Y_test, Y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(Y_test, Y_pred))

joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
