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

    # TODO: tokenize the texts

    # TODO: build vocabulary

    # TODO: compute class priors

    # TODO: compute word probabilities using simple MLE (no smoothing)

    # TODO: predict the class of test_email

    raise NotImplementedError


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
    # TODO: load iris dataset

    # TODO: split into train and test

    # TODO: implement Euclidean distance

    # TODO: implement prediction using k nearest neighbors

    # TODO: compute train predictions and test predictions

    # TODO: compute accuracies

    raise NotImplementedError
