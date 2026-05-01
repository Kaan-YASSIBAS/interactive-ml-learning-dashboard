import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_student_score_dataset():
    hours_studied = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
    exam_scores = np.array([35, 40, 50, 55, 60, 65, 72, 78, 85, 92])

    return hours_studied, exam_scores


def train_linear_regression_model():
    X, y = get_student_score_dataset()

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)

    return model, X, y, predictions, mae, mse, r2


def get_linear_regression_info():
    model, X, y, predictions, mae, mse, r2 = train_linear_regression_model()

    data_points = []

    for index in range(len(X)):
        data_points.append({
            "hours_studied": float(X[index][0]),
            "actual_score": float(y[index]),
            "predicted_score": float(predictions[index])
        })

    return {
        "category": "Supervised Learning → Regression",
        "algorithm": "Linear Regression",
        "description": "Linear Regression, sayısal bir değeri tahmin etmek için kullanılan supervised learning algoritmasıdır.",
        "formula": "y = wx + b",
        "weight": float(model.coef_[0]),
        "bias": float(model.intercept_),
        "mae": float(mae),
        "mse": float(mse),
        "r2_score": float(r2),
        "data_points": data_points
    }


def predict_exam_score(hours_studied: float):
    model, _, _, _, _, _, _ = train_linear_regression_model()

    prediction = model.predict([[hours_studied]])[0]

    return {
        "hours_studied": hours_studied,
        "predicted_score": float(prediction)
    }