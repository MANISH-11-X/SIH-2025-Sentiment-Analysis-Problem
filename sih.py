from flask import Flask, request, render_template
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

model = LogisticRegression()

x_data =[
    "I love this movie",
    "This movie is amazing",
    "This movie is okay",
    "It was an average movie",
    "I hate this movie",
    "This movie is terrible",
    "I don't like this movie"
]

x_sentiments = [
    "Positive",
    "Positive",
    "Neutral",
    "Neutral",
    "Negative",
    "Negative",
    "Negative"
]

x_train = vectorizer.fit_transform(x_data)

model.fit(x_train,x_sentiments)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods = ["POST"])
def predict():

    if request.method == "POST":

        user = request.form["user"]

        x_comment = vectorizer.transform([user])

        prediction = model.predict(x_comment)[0]

    return render_template('index.html', prediction = prediction, comment = user)

app.run(debug=True)