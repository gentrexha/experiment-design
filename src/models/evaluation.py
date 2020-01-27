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
        "Audio": df_audio_test,
        "Text": df_text_test,
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
            train_x[features], train_y
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
    logger.info("Finished evaluating Las Vegas Wrapper!")
    logger.info("Started evaluating cross validation...")
    cross_validation_evaluation(dev_set)
    logger.info("Finished evaluating cross validation!")
    logger.info("Started evaluating Majority Voting...")
    majority_voting(dev_set, test_set)
    logger.info("Finished evaluating Majority Voting!")
    logger.info("Started evaluating Label Stacking...")
    label_stacking(dev_set, test_set)
    logger.info("Finished evaluating Label Stacking!")
    logger.info("Started evaluating Label Attribute Stacking...")
    label_attr_stacking(dev_set, test_set, df_train, df_test)
    logger.info("Finished evaluating Label Attribute Stacking!")


def eval_las_vegas(set):
    """
    Calculates Las Vegas Wrapper
    :param set: dictionary of training dataframes
    """
    logger = logging.getLogger(__name__)
    las_vegas_results = []
    n_runs = len(set) * len(classifiers)
    ii = 0
    for train_set_name, train_set_df in set.items():
        for classifier_name, classifier in classifiers.items():
            ii += 1
            progress_percent = ii / n_runs * 100
            logger.info(
                f"Evaluating {classifier_name} on {train_set_name} ({progress_percent:.1f}%)"
            )

            cur_x = train_set_df.drop("class", axis=1)
            cur_y = train_set_df["class"]
            cur_train_x, cur_val_x, cur_train_y, cur_val_y = train_test_split(
                cur_x, cur_y
            )

            cur_best_score, cur_best_features = las_vegas(
                cur_train_x,
                cur_train_y,
                cur_val_x,
                cur_val_y,
                classifier,
                make_scorer(f1_score),
            )

            las_vegas_results.append(
                (train_set_name, classifier_name, cur_best_features)
            )
    with (results_dir / "las_vegas.json").open("w") as ofile:
        json.dump(las_vegas_results, ofile)


def cross_validation_evaluation(dev_set):
    with (results_dir / "las_vegas.json").open() as f:
        data = json.load(f)

    scoring = {"f1": "f1", "precision": "precision", "recall": "recall"}

    cv_results = []
    new_data = []
    for result in data:
        dataset = dev_set[result[0]]
        model = classifiers[result[1]]

        scores = cross_validate(
            model, dataset[result[2]], dataset["class"], scoring=scoring, cv=10,
        )

        if (
            scores["test_f1"].mean() >= 0.5
            and scores["test_precision"].mean() >= 0.5
            and scores["test_recall"].mean() >= 0.5
        ):
            new_data.append(result)
            cv_results.append(
                [
                    result[1],
                    result[0],
                    scores["test_f1"].mean(),
                    scores["test_precision"].mean(),
                    scores["test_recall"].mean(),
                ]
            )

    # JSON Output for further usage
    with (results_dir / "cv_results.json").open("w") as ofile:
        json.dump(new_data, ofile)

    # CSV Output of scores
    df = pd.DataFrame(
        cv_results, columns=["estimator", "dataset", "f1", "precision", "recall"]
    )
    df.to_csv(results_dir / "cv_results.csv", index=False)


def majority_voting(dev_set, test_set):
    """
    Does Majority Voting with Las Vegas Wrapper results.
    :param dev_set: training dictionary with dataframes
    :param test_set: test dictionary with dataframes
    """
    with (results_dir / "cv_results.json").open() as f:
        data = json.load(f)

    train_scores = []
    test_scores = []

    for result in data:
        data_train = dev_set[result[0]]
        data_test = test_set[result[0]]
        model = classifiers[result[1]]

        train_predictions = cross_val_predict(
            model, data_train[result[2]], data_train["class"], cv=10, n_jobs=-1
        )

        model.fit(data_train[result[2]], data_train["class"])
        test_predictions = model.predict(data_test[result[2]])

        train_scores.append(train_predictions)
        test_scores.append(test_predictions)

    maj_train_predictions = list(pd.DataFrame(train_scores).mode(axis=0).iloc[0])
    maj_test_predictions = list(pd.DataFrame(test_scores).mode(axis=0).iloc[0])

    save_results(
        data_test,
        data_train,
        maj_test_predictions,
        maj_train_predictions,
        "majority.json",
    )


def label_stacking(dev_set, test_set):
    with (results_dir / "cv_results.json").open() as f:
        data = json.load(f)

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    for idx, result in enumerate(data):
        data_train = dev_set[result[0]]
        data_test = test_set[result[0]]
        model = classifiers[result[1]]

        train_predictions = cross_val_predict(
            model, data_train[result[2]], data_train["class"], cv=10, n_jobs=-1
        )

        model.fit(data_train[result[2]], data_train["class"])
        test_predictions = model.predict(data_test[result[2]])

        df_train[f"est_{idx}"] = train_predictions
        df_test[f"est_{idx}"] = test_predictions

    # evaluate new dataframe
    stacked_estimator = LogisticRegression()
    train_predictions_stacked = cross_val_predict(
        stacked_estimator, df_train, data_train["class"], cv=10, n_jobs=-1
    )
    stacked_estimator.fit(df_train, data_train["class"])
    stacked_test_predictions = stacked_estimator.predict(df_test)

    save_results(
        data_test,
        data_train,
        stacked_test_predictions,
        train_predictions_stacked,
        "label_stacking.json",
    )


def save_results(
    df_test, data_train, stacked_test_predictions, train_predictions_stacked, name
):
    """
    Saves results to `name` file.
    :param df_test:
    :param data_train:
    :param stacked_test_predictions:
    :param train_predictions_stacked:
    :param name:
    """
    final_train_scores = precision_recall_fscore_support(
        data_train["class"], train_predictions_stacked, average="micro"
    )
    final_test_scores = precision_recall_fscore_support(
        df_test["class"], stacked_test_predictions, average="micro"
    )
    label_stacking_results = {
        "train_f1": final_train_scores[2],
        "train_precision": final_train_scores[0],
        "train_recall": final_train_scores[1],
        "test_f1": final_test_scores[2],
        "test_precision": final_test_scores[0],
        "test_recall": final_test_scores[1],
    }
    with (results_dir / name).open("w") as ofile:
        json.dump(label_stacking_results, ofile)


def label_attr_stacking(dev_set, test_set, full_train, full_test):
    with (results_dir / "cv_results.json").open() as f:
        data = json.load(f)

    scoring = {"f1": "f1", "precision": "precision", "recall": "recall"}

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    features = []
    for idx, result in enumerate(data):
        data_train = dev_set[result[0]]
        data_test = test_set[result[0]]
        model = classifiers[result[1]]

        train_predictions = cross_val_predict(
            model, data_train[result[2]], data_train["class"], cv=10, n_jobs=-1
        )

        model.fit(data_train[result[2]], data_train["class"])
        test_predictions = model.predict(data_test[result[2]])

        df_train[f"est_{idx}"] = train_predictions
        df_test[f"est_{idx}"] = test_predictions
        features.extend(list(set(result[2]) - set(features)))

    # evaluate new dataframe
    df_train = pd.concat((df_train, full_train[features]), axis=1, sort=False)
    df_test = pd.concat((df_test, full_test[features]), axis=1, sort=False)

    stacked_estimator = LogisticRegression()
    stacked_train_pred = cross_val_predict(
        stacked_estimator, df_train, data_train["class"], cv=10
    )
    stacked_estimator.fit(df_train, data_train["class"])
    stacked_test_pred = stacked_estimator.predict(df_test)

    save_results(
        data_test,
        data_train,
        stacked_test_pred,
        stacked_train_pred,
        "label_attr_stacking.json",
    )


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
