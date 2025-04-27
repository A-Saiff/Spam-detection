from flask import Flask, render_template, request
import joblib
from unidecode import unidecode
from sentence_transformers import SentenceTransformer

model = joblib.load("spam_classifier_model.pkl")
bert = SentenceTransformer("bert_model")


def predict_message(message):
    cleaned = unidecode(message)
    vector = bert.encode([cleaned])
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
