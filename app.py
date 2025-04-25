from flask import Flask, render_template, request
import joblib
import string


model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


def clean_text(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    return text


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
