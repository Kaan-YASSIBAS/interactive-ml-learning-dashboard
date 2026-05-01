from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def get_iris_data():
    iris = load_iris()

    return {
        "data": iris.data,
        "target": iris.target,
        "feature_names": iris.feature_names,
        "target_names": iris.target_names
    }


def evaluate_knn(
    k: int = 5,
    sepal_length: float = 5.1,
    sepal_width: float = 3.5,
    petal_length: float = 1.4,
    petal_width: float = 0.2
):
    iris = get_iris_data()

    X = iris["data"]
    y = iris["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)

    sample = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    sample_scaled = scaler.transform(sample)

    sample_prediction = model.predict(sample_scaled)[0]
    sample_probabilities = model.predict_proba(sample_scaled)[0]

    probability_result = {
        iris["target_names"][index]: float(sample_probabilities[index])
        for index in range(len(iris["target_names"]))
    }

    full_scaler = StandardScaler()
    X_scaled = full_scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    sample_scaled_for_pca = full_scaler.transform(sample)
    sample_pca = pca.transform(sample_scaled_for_pca)[0]

    points = []

    for index, point in enumerate(X_pca):
        points.append({
            "pc1": float(point[0]),
            "pc2": float(point[1]),
            "target": int(y[index]),
            "target_name": str(iris["target_names"][y[index]])
        })

    return {
        "category": "Supervised Learning → Classification",
        "algorithm": "K-Nearest Neighbors",
        "description": "KNN, yeni bir veriyi en yakın K komşusuna bakarak sınıflandıran supervised learning algoritmasıdır.",
        "formula": "d = sqrt((x2 - x1)^2 + (y2 - y1)^2)",
        "k": k,
        "accuracy": float(accuracy),
        "sample": {
            "features": {
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width
            },
            "prediction": str(iris["target_names"][sample_prediction]),
            "probabilities": probability_result,
            "pca_point": {
                "pc1": float(sample_pca[0]),
                "pc2": float(sample_pca[1])
            }
        },
        "points": points
    }