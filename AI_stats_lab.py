"""
AIstats_lab.py

Student starter file for:
1. Naive Bayes spam classification
2. K-Nearest Neighbors on Iris
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def accuracy_score(y_true, y_pred):
    """
    Compute classification accuracy.
    """
    return float(np.mean(y_true == y_pred))


# =========================
# Q1 Naive Bayes
# =========================

def naive_bayes_mle_spam():
    """
    Implement Naive Bayes spam classification using simple MLE.

    Use the dataset below:

    texts = [
        "win money now",
        "limited offer win cash",
        "cheap meds available",
        "win big prize now",
        "exclusive offer buy now",
        "cheap pills buy cheap meds",
        "win lottery claim prize",
        "urgent offer win money",
        "free cash bonus now",
        "buy meds online cheap",
        "meeting schedule tomorrow",
        "project discussion meeting",
        "please review the report",
        "team meeting agenda today",
        "project deadline discussion",
        "review the project document",
        "schedule a meeting tomorrow",
        "please send the report",
        "discussion on project update",
        "team sync meeting notes"
    ]

    labels = np.array([
        1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0
    ])

    Predict the class of:
        test_email = "win cash prize now"

    Returns
    -------
    priors : dict
    word_probs : dict
    prediction : int
    """
    texts = [
        "win money now",
        "limited offer win cash",
        "cheap meds available",
        "win big prize now",
        "exclusive offer buy now",
        "cheap pills buy cheap meds",
        "win lottery claim prize",
        "urgent offer win money",
        "free cash bonus now",
        "buy meds online cheap",
        "meeting schedule tomorrow",
        "project discussion meeting",
        "please review the report",
        "team meeting agenda today",
        "project deadline discussion",
        "review the project document",
        "schedule a meeting tomorrow",
        "please send the report",
        "discussion on project update",
        "team sync meeting notes"
    ]

    labels = np.array([
        1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0
    ])

    test_email = "win cash prize now"

    tokenized = [text.split() for text in texts]

    vocabulary = set(word for doc in tokenized for word in doc)

    classes = np.unique(labels)
    n_total = len(labels)
    priors = {c: float(np.sum(labels == c)) / n_total for c in classes}

    word_probs = {c: {} for c in classes}
    for c in classes:
        class_docs = [tokenized[i] for i in range(len(labels)) if labels[i] == c]
        all_words = [word for doc in class_docs for word in doc]
        total_words = len(all_words)
        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        for word in vocabulary:
            word_probs[c][word] = word_counts.get(word, 0) / total_words if total_words > 0 else 0.0

    test_tokens = test_email.split()
    log_posteriors = {}
    for c in classes:
        log_posterior = np.log(priors[c])
        for word in test_tokens:
            if word in word_probs[c] and word_probs[c][word] > 0:
                log_posterior += np.log(word_probs[c][word])
            else:
                log_posterior += float('-inf')
        log_posteriors[c] = log_posterior

    prediction = int(max(log_posteriors, key=log_posteriors.get))

    return priors, word_probs, prediction


# =========================
# Q2 KNN
# =========================

def knn_iris(k=3, test_size=0.2, seed=0):
    """
    Implement KNN from scratch on the Iris dataset.

    Steps:
    1. Load Iris data
    2. Split into train/test
    3. Compute Euclidean distance
    4. Predict with majority voting
    5. Return train accuracy, test accuracy, and test predictions

    Returns
    -------
    train_accuracy : float
    test_accuracy : float
    predictions : np.ndarray
    """
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2, axis=1))

    def predict_single(x):
        distances = euclidean_distance(X_train, x)
        k_indices = np.argsort(distances)[:k]
        k_labels = y_train[k_indices]
        counts = np.bincount(k_labels)
        return int(np.argmax(counts))

    def predict_all(X):
        return np.array([predict_single(x) for x in X])

    train_predictions = predict_all(X_train)
    predictions = predict_all(X_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, predictions)

    return train_accuracy, test_accuracy, predictions
