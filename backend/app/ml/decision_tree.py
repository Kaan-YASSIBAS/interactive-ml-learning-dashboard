from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def get_iris_data():
    iris = load_iris()

    return {
        "data": iris.data,
        "target": iris.target,
        "feature_names": iris.feature_names,
        "target_names": iris.target_names
    }


def evaluate_decision_tree(
    max_depth: int = 3,
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

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    sample = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    sample_prediction = model.predict(sample)[0]
    sample_probabilities = model.predict_proba(sample)[0]

    probabilities = {
        str(iris["target_names"][index]): float(sample_probabilities[index])
        for index in range(len(iris["target_names"]))
    }

    tree_rules = extract_tree_rules(
        model,
        iris["feature_names"],
        iris["target_names"]
    )

    tree_structure = extract_tree_structure(
    model,
    iris["feature_names"],
    iris["target_names"]
    )

    return {
        "category": "Supervised Learning → Classification",
        "algorithm": "Decision Tree",
        "description": "Decision Tree, veriyi karar kurallarıyla dallara ayırarak sınıflandırma yapan supervised learning algoritmasıdır.",
        "formula": "Gini = 1 - Σ p_i²",
        "max_depth": max_depth,
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "gap": float(train_accuracy - test_accuracy),
        "sample": {
            "features": {
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width
            },
            "prediction": str(iris["target_names"][sample_prediction]),
            "probabilities": probabilities
        },
        "tree_rules": tree_rules,
        "tree_structure": tree_structure
    }


def extract_tree_rules(model, feature_names, target_names):
    tree = model.tree_
    rules = []

    def traverse(node_id, depth):
        if tree.feature[node_id] != -2:
            feature_name = feature_names[tree.feature[node_id]]
            threshold = tree.threshold[node_id]

            rules.append({
                "depth": depth,
                "type": "decision",
                "rule": f"{feature_name} <= {threshold:.2f}"
            })

            traverse(tree.children_left[node_id], depth + 1)
            traverse(tree.children_right[node_id], depth + 1)
        else:
            class_id = tree.value[node_id][0].argmax()
            class_name = target_names[class_id]

            rules.append({
                "depth": depth,
                "type": "leaf",
                "rule": f"Predict: {class_name}"
            })

    traverse(0, 0)

    return rules

def extract_tree_structure(model, feature_names, target_names):
    tree = model.tree_

    def build_node(node_id):
        if tree.feature[node_id] != -2:
            feature_name = feature_names[tree.feature[node_id]]
            threshold = tree.threshold[node_id]

            return {
                "type": "decision",
                "rule": f"{feature_name} <= {threshold:.2f}",
                "left": build_node(tree.children_left[node_id]),
                "right": build_node(tree.children_right[node_id])
            }

        class_id = tree.value[node_id][0].argmax()
        class_name = target_names[class_id]

        return {
            "type": "leaf",
            "prediction": str(class_name)
        }

    return build_node(0)