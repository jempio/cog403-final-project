import pandas as pd
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score


def compare_text_models(X_train, y_train, X_test, y_test, configs, random_state, max_iter=5000):
    """
    Train and compare multiple logistic regression text models.

    """
    results = []
    predictions_dict = {}

    for config in configs:
        model = LogisticRegression(
            penalty=config["penalty"],
            solver=config["solver"],
            C=config["C"],
            max_iter=max_iter,
            random_state=random_state
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        predictions_dict[config["name"]] = preds

        results.append({
            "Model": config["name"],
            "Macro-F1": f1_score(y_test, preds, average="macro"),
            "Weighted-F1": f1_score(y_test, preds, average="weighted"),
            "Accuracy": accuracy_score(y_test, preds),
        })

    results_df = pd.DataFrame(results).sort_values("Macro-F1", ascending=False).round(4)
    return results_df, predictions_dict


def compare_fusion_models(X_train_tfidf, X_test_tfidf, X_train_app_sparse, X_test_app_sparse, y_train, y_test, configs, random_state, max_iter=5000):
    """
    Train and compare multiple fusion models that combine TF-IDF text features with appraisal features.
    """
    results = []
    predictions_dict = {}

    for config in configs:
        X_train_fused = hstack(
            [X_train_tfidf * config["text_weight"], X_train_app_sparse],
            format="csr"
        )
        X_test_fused = hstack(
            [X_test_tfidf * config["text_weight"], X_test_app_sparse],
            format="csr"
        )

        model = LogisticRegression(
            penalty=config["penalty"],
            solver="saga",
            C=config["C"],
            max_iter=max_iter,
            random_state=random_state
        )

        model.fit(X_train_fused, y_train)
        preds = model.predict(X_test_fused)
        predictions_dict[config["name"]] = preds

        results.append({
            "Model": config["name"],
            "Penalty": config["penalty"].upper(),
            "C": config["C"],
            "Text Weight": config["text_weight"],
            "Macro-F1": f1_score(y_test, preds, average="macro"),
            "Weighted-F1": f1_score(y_test, preds, average="weighted"),
            "Accuracy": accuracy_score(y_test, preds),
        })

    results_df = pd.DataFrame(results).sort_values("Macro-F1", ascending=False).round(4)
    return results_df, predictions_dict