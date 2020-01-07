# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB


# @click.command()
# @click.argument("input_filepath", type=click.Path(exists=True))
# @click.argument("output_filepath", type=click.Path())
def main():  # input_filepath, output_filepath
    """ Runs data processing scripts to turn processed data from
        (../processed) into cleaned data ready to be analyzed
        (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Making final data set from processed data")
    logger.info("Loading datasets")
    df_audio_dev, df_metadata_dev, df_text_dev, df_visual_dev = load_dev_data()
    df_audio_test, df_metadata_test, df_text_test, df_visual_test = load_test_data()

    dev_sets = {
        "Metadata": df_metadata_dev,
        "Visual": df_visual_dev,
        "Audio": df_audio_dev,
        "Text": df_text_dev,
    }
    test_sets = {
        "Metadata": df_metadata_test,
        "Visual": df_visual_test,
        "Audio": df_audio_test,
        "Text": df_text_test,
    }

    logger.info("Merging loaded dataframes")
    df_dev = pd.concat(dev_sets, axis=1, sort=False)
    df_test = pd.concat(test_sets, axis=1, sort=False)

    df_dev.columns = df_dev.columns.droplevel()
    df_test.columns = df_test.columns.droplevel()

    print(df_dev.head())
    print(df_dev.describe())

    logger.info("Finished describing dataframe")


def load_test_data():
    """
    Load test dataframes from feature extracted test data.
    :return: pd.DataFrame()
    """
    df_metadata_test = pd.read_csv(processed_dir / "metadata_descriptor_test.csv")
    # df_user_ratings_test = pd.read_csv(processed_dir / "user_ratings_test.csv")
    # df_metadata_user_ratings_test = pd.read_csv(processed_dir / "metadata_user_rating_test.csv")
    df_audio_test = pd.read_csv(processed_dir / "audio_descriptor_test.csv")
    df_visual_test = pd.read_csv(processed_dir / "visual_descriptor_test.csv")
    df_text_test = pd.read_csv(processed_dir / "text_descriptor_test.csv")
    return df_audio_test, df_metadata_test, df_text_test, df_visual_test


def load_dev_data():
    """
    Load dev dataframes from feature extracted dev data.
    :return: pd.DataFrame()
    """
    df_metadata_dev = pd.read_csv(processed_dir / "metadata_descriptor_dev.csv")
    # df_user_ratings_dev = pd.read_csv(processed_dir / "user_ratings.csv")
    # df_metadata_user_ratings_dev = pd.read_csv(processed_dir / "metadata_user_rating.csv")
    df_audio_dev = pd.read_csv(processed_dir / "audio_descriptor_dev.csv")
    df_visual_dev = pd.read_csv(processed_dir / "visual_descriptor_dev.csv")
    df_text_dev = pd.read_csv(processed_dir / "text_descriptor_dev.csv")
    return df_audio_dev, df_metadata_dev, df_text_dev, df_visual_dev


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # data path
    processed_dir = project_dir / "data/processed"

    # classifiers
    # TODO(Discuss with @PrincMullatahiri if you want to move this another file.)
    classifiers = {
        "KNN": KNeighborsClassifier(),
        "Nearest Mean": NearestCentroid(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(solver="liblinear"),
        "SVM,": SVC(gamma="scale"),
        "Bagging": BaggingClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "AdaBoost": AdaBoostClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Naive Bayes": GaussianNB(),
    }

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
