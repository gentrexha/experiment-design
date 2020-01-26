# -*- coding: utf-8 -*-
import logging
import json
import random

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
    cross_validate,
    cross_val_predict,
)
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
    precision_recall_fscore_support,
)


def load_data():
    """
    Load and merge datasets.
    :return: 
    """
    # Dev Data; label column: class
    df_metadata_dev = pd.read_csv(eval_dir / "metadata_dev.csv")
    df_user_ratings = pd.read_csv(eval_dir / "user_ratings_dev.csv")
    df_metadata_user_ratings_dev = pd.read_csv(
        eval_dir / "metadata_user_rating_dev.csv"
    )
    df_audio_dev = pd.read_csv(eval_dir / "audio_dev.csv")
    df_visual_dev = pd.read_csv(eval_dir / "visual_dev.csv")
    df_text_dev = pd.read_csv(eval_dir / "text_dev.csv")

    # Test Data; label column: class
    df_metadata_test = pd.read_csv(eval_dir / "metadata_test.csv")
    df_user_ratings_test = pd.read_csv(eval_dir / "user_ratings_test.csv")
    df_metadata_user_ratings_test = pd.read_csv(
        eval_dir / "metadata_user_rating_test.csv"
    )
    df_audio_test = pd.read_csv(eval_dir / "audio_test.csv")
    df_visual_test = pd.read_csv(eval_dir / "visual_test.csv")
    df_text_test = pd.read_csv(eval_dir / "text_test.csv")

    dev_set = {
        "User rating": df_user_ratings,
        "Visual": df_visual_dev,
        "Metadata": df_metadata_dev,
        "Metadata + user rating": df_metadata_user_ratings_dev,
        "Audio": df_audio_dev,
        "Text": df_text_dev,
    }

    test_set = {
        "User rating": df_user_ratings_test,
        "Visual": df_visual_test,
        "Metadata": df_metadata_test,
        "Metadata + user rating": df_metadata_user_ratings_test,
    }
    df_train = pd.concat(dev_set, axis=1, sort=False)
    df_test = pd.concat(test_set, axis=1, sort=False)

    df_train.columns = df_train.columns.droplevel()
    df_test.columns = df_test.columns.droplevel()

    return df_train, df_test, dev_set, test_set


def las_vegas(train_x, train_y, valid_x, valid_y, model, scorer, n_iterations=1000):
    best_score = -np.inf
    best_features = list(train_x.columns)

    for ii in range(n_iterations):
        n_features = random.randint(1, train_x.shape[1])
        features = random.sample(list(train_x.columns), n_features)

        model.fit(
            train_x[features], train_y,
        )

        score = scorer(model, valid_x[features], valid_y)

        if score > best_score:
            best_score = score
            best_features = features

    return best_score, features


def main():
    """ Runs evaluation methods for generating outputs for tables.

    """
    logger = logging.getLogger(__name__)
    logger.info("Loading data...")
    df_train, df_test, dev_set, test_set = load_data()
    logger.info("Data loaded!")
    logger.info("Started evaluating Las Vegas Wrapper...")
    eval_las_vegas(dev_set)


def eval_las_vegas(set):
    logger = logging.getLogger(__name__)
    las_vegas_results = []
    n_runs = len(set) * len(classifiers)
    ii = 0
    for train_set_name, train_set_df in set.items():
        for classifier_name, classifier in classifiers.items():
            ii += 1
            progress_percent = ii / n_runs * 100
            logger.info(f'Evaluating {classifier_name} on {train_set_name}  ({progress_percent:.1f}%)')

            cur_x = train_set_df.drop('class', axis=1)
            cur_y = train_set_df['class']
            cur_train_x, cur_val_x, cur_train_y, cur_val_y = train_test_split(cur_x, cur_y)

            cur_best_score, cur_best_features = las_vegas(
                cur_train_x,
                cur_train_y,
                cur_val_x,
                cur_val_y,
                classifier,
                make_scorer(f1_score)
            )

            las_vegas_results.append((
                train_set_name,
                classifier_name,
                cur_best_features
            ))
    with (results_dir / 'las_vegas.json').open('w') as ofile:
        json.dump(las_vegas_results, ofile)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # eval path
    eval_dir = project_dir / "data/evaluation"
    # results path
    results_dir = project_dir / "data/results"

    classifiers = {
        "nearest neighbor": KNeighborsClassifier(),
        "nearest mean": NearestCentroid(),
        "decision tree": DecisionTreeClassifier(),
        "logistic regression": LogisticRegression(solver="liblinear"),
        "SVM,": SVC(gamma="scale"),
        "bagging": BaggingClassifier(),
        "random forest": RandomForestClassifier(n_estimators=100),
        "AdaBoost": AdaBoostClassifier(),
        "gradient boosting": GradientBoostingClassifier(),
        "naive Bayes": GaussianNB(),
    }

    main()
