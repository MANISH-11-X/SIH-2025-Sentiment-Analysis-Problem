from flask import Flask, request, render_template
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

vectorizer = TfidfVectorizer()

model = LogisticRegression()

x_data = [
    # Positive
    "The government scheme will greatly benefit rural citizens",
    "This policy is a very good step for the development of our village",
    "I fully support this proposal",
    "The new education scheme will help many students",
    "The government has done a good job with this initiative",
    "This policy will improve public transportation",
    "The proposed healthcare scheme is very useful",
    "This initiative will create better employment opportunities",
    "The new digital services will make government services easier",
    "I appreciate the government's effort to improve public facilities",
    "This scheme will help farmers significantly",
    "The proposal is beneficial for ordinary citizens",
    "This is a positive step toward rural development",
    "The new roads will improve connectivity",
    "I support the proposed environmental protection measures",

    # Neutral
    "I need more information about this proposal",
    "The policy seems reasonable but more details are required",
    "I have no strong opinion about this scheme",
    "The proposal can be considered after further discussion",
    "I would like to know how this policy will be implemented",
    "The scheme appears acceptable at this stage",
    "More information about the eligibility criteria is needed",
    "I am waiting for the final details of the proposal",
    "The impact of this policy should be evaluated later",
    "The proposal requires further clarification",
    "I have read the policy but cannot decide yet",
    "The implementation process is not clear",
    "Further public discussion may be required",
    "The proposal can be reviewed before implementation",
    "I would like to understand the benefits of this scheme",

    # Negative
    "This policy will create problems for ordinary citizens",
    "I strongly oppose this proposal",
    "The government scheme is not useful for people in our area",
    "This policy will increase the burden on citizens",
    "I am not satisfied with the proposed changes",
    "The scheme does not address the actual problems of people",
    "This proposal will negatively affect poor families",
    "The implementation of this policy will be very difficult",
    "I disagree with this government decision",
    "The public transport system is still very poor",
    "The proposed changes are unfair to citizens",
    "This scheme will create unnecessary difficulties",
    "The government should reconsider this proposal",
    "The policy ignores the concerns of rural citizens",
    "I do not support this initiative"
]

x_sentiments = [
    # Positive - 15
    "Positive", "Positive", "Positive", "Positive", "Positive",
    "Positive", "Positive", "Positive", "Positive", "Positive",
    "Positive", "Positive", "Positive", "Positive", "Positive",

    # Neutral - 15
    "Neutral", "Neutral", "Neutral", "Neutral", "Neutral",
    "Neutral", "Neutral", "Neutral", "Neutral", "Neutral",
    "Neutral", "Neutral", "Neutral", "Neutral", "Neutral",

    # Negative - 15
    "Negative", "Negative", "Negative", "Negative", "Negative",
    "Negative", "Negative", "Negative", "Negative", "Negative",
    "Negative", "Negative", "Negative", "Negative", "Negative"
]

x_train = vectorizer.fit_transform(x_data)

model.fit(x_train,x_sentiments)

app = Flask(__name__)

predictions = []

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods = ["POST"])
def predict():

    if request.method == "POST":

        user = request.form["user"]

        x_comment = vectorizer.transform([user])

        prediction = model.predict(x_comment)[0]

        predictions.append(prediction)

    return render_template('index.html', prediction = prediction, comment = user)

@app.route("/dashboard")
def dashboard():

    count = Counter(predictions)

    total = len(predictions)

    if total > 0:

        positive = (count["Positive"] / total) * 100
        neutral = (count["Neutral"] / total) * 100
        negative = (count["Negative"] / total) * 100

    else:
        positive = 0
        neutral = 0
        negative = 0

    return render_template(
        'dashboard.html',

        total = total,

        positive = round(positive, 2),
        neutral = round(neutral, 2),
        negative  = round(negative, 2),

        positive_c = count["positive"],
        neutral_c = ["neutral"],
        negative_c = ["negative"]
    )

app.run(debug=True)
