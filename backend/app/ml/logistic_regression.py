from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def get_spam_dataset():
    messages = [
        "Congratulations you won a free prize click now",
        "Free entry in a prize draw claim your reward",
        "Win money now by clicking this link",
        "Urgent your account has been compromised click here",
        "Limited offer buy now and get discount",
        "You have been selected for a cash reward",
        "Exclusive deal just for you claim now",
        "Earn dollars from home with this simple trick",
        "Free coupon available click to claim",
        "Winner claim your free gift today",

        "Hey are we still meeting today",
        "Can you send me the homework file",
        "Please review the meeting notes",
        "Are you coming to class today",
        "Do not forget the project deadline",
        "I will call you after work",
        "Let's have lunch tomorrow",
        "Can we reschedule the meeting",
        "Please send the report when you can",
        "The lecture starts at ten tomorrow"
    ]

    labels = [
        "spam", "spam", "spam", "spam", "spam",
        "spam", "spam", "spam", "spam", "spam",
        "ham", "ham", "ham", "ham", "ham",
        "ham", "ham", "ham", "ham", "ham"
    ]

    return messages, labels


def train_logistic_regression_model():
    messages, labels = get_spam_dataset()

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english"
        )),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(messages, labels)

    return model, messages, labels


def get_logistic_regression_info():
    model, messages, labels = train_logistic_regression_model()

    training_examples = []

    for message, label in zip(messages, labels):
        training_examples.append({
            "message": message,
            "label": label
        })

    return {
        "category": "Supervised Learning → Classification",
        "algorithm": "Logistic Regression",
        "description": "Logistic Regression, classification problemlerinde kullanılan supervised learning algoritmasıdır.",
        "formula_linear": "z = wx + b",
        "formula_sigmoid": "p = 1 / (1 + e^(-z))",
        "classes": ["ham", "spam"],
        "training_examples": training_examples
    }


def predict_message_class(message: str):
    model, _, _ = train_logistic_regression_model()

    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]

    class_names = model.named_steps["classifier"].classes_

    probability_result = {
        class_names[index]: float(probabilities[index])
        for index in range(len(class_names))
    }

    return {
        "message": message,
        "prediction": prediction,
        "probabilities": probability_result
    }