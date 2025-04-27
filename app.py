import spacy
from flask import Flask, render_template, request
import joblib
import string
from unidecode import unidecode

model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


slang_dict = {
    "wkly": "weekly",
    "2": "to",
    "tkts": "tickets",
    "freemsg": "free message",
    "rcv": "receive",
    "u": "you",
    "r": "are",
    "txt": "text",
    "4": "for",
    "ur": "your",
    "mob": "mobile",
    "ansr": "answer",
    "msg": "message",
    "msgs": "messages",
    "yr": "your",
    "luv": "love",
    "plz": "please",
    "pls": "please",
    "knw": "know",
    "askd": "asked"
}
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


# Updated clean text
def clean_text(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    text = unidecode(text)
    words = text.split()
    for index, word in enumerate(words):
        if word in slang_dict:
            words[index] = slang_dict[word]
    cleaned = " ".join(words)
    doc = nlp(cleaned)
    lemmas = [token.lemma_.lower().strip() for token in doc]
    return " ".join(lemmas)


def predict_message(message):
    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]
    return "SPAM" if prediction == 1 else "NOT SPAM", max(probability)


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    prob = 0
    if request.method == "POST":
        message = request.form["message"]
        result, prob = predict_message(message)
    return render_template("index.html", result=result, prob=prob)


if __name__ == "__main__":
    app.run(debug=True)
