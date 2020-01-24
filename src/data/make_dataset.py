# -*- coding: utf-8 -*-
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from typing import List, Tuple


def load_test_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load test dataframes from feature extracted test data.
    :return: pd.DataFrame
    """
    df_metadata_test = pd.read_csv(
        processed_dir / "metadata_descriptor_test.csv", index_col="name"
    ).sort_index()
    df_audio_test = pd.read_csv(
        processed_dir / "audio_descriptor_test.csv", index_col="name"
    ).sort_index()
    df_visual_test = pd.read_csv(
        processed_dir / "visual_descriptor_test.csv", index_col="name"
    ).sort_index()
    df_text_test = pd.read_csv(
        processed_dir / "text_descriptor_test.csv", index_col="name"
    ).sort_index()
    return df_audio_test, df_metadata_test, df_text_test, df_visual_test


def load_dev_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load dev dataframes from feature extracted dev data.
    :return: pd.DataFrame
    """
    df_metadata_dev = pd.read_csv(
        processed_dir / "metadata_descriptor_dev.csv", index_col="name"
    ).sort_index()
    df_audio_dev = pd.read_csv(
        processed_dir / "audio_descriptor_dev.csv", index_col="name"
    ).sort_index()
    df_visual_dev = pd.read_csv(
        processed_dir / "visual_descriptor_dev.csv", index_col="name"
    ).sort_index()
    df_text_dev = pd.read_csv(
        processed_dir / "text_descriptor_dev.csv", index_col="name"
    ).sort_index()
    return df_audio_dev, df_metadata_dev, df_text_dev, df_visual_dev


def join_train_test_datasets(
    df_dev: pd.DataFrame, df_test: pd.DataFrame
) -> pd.DataFrame:
    """
    Joins dev and test datasets and assigns new 'train' column.
    :param df_dev: pd.DataFrame
    :param df_test: pd.DataFrame
    :return: pd.DataFrame
    """
    df_dev["train"] = True
    df_test["train"] = False
    return pd.concat([df_dev, df_test], axis=0, sort=False)


def min2int(x):
    if isinstance(x, str):
        return x.split(" ")[0]
    else:
        return x


def boxOffice(x):
    if isinstance(x, str):
        x = x.replace("M", "")
        x = x.replace("k", "")
        return x.replace("$", "")
    else:
        return 0


def drop_metadata_columns(df: pd.DataFrame, cols=None,) -> pd.DataFrame:
    """
    Drops long text columns.
    :param df: dataframe to drop columns from
    :param cols: columns to drop
    :return: dataframe without cols
    """
    if cols is None:
        cols = [
            # "actors",
            "Website",
            # "writer",
            "title",
            # "plot",
            "imdbID",
            "awards",
            "language",
            "type",
            "poster",
            "tomatoConsensus",
        ]
    return df.drop(cols, axis=1)


def director(x):
    if isinstance(x, str):
        return x.split(",")[0]
    else:
        return np.nan


def preprocess_metadata_only(
    df: pd.DataFrame, user_rating_columns: List[str]
) -> pd.DataFrame:
    dff = df.drop(user_rating_columns, axis=1)
    dff = pd.concat(
        [dff, pd.get_dummies(dff["Production"], prefix="prod", dummy_na=True)], axis=1
    )
    dff = dff.drop("Production", axis=1)
    dff = pd.concat(
        [dff, pd.get_dummies(dff["genre"], prefix="genre", dummy_na=True)], axis=1
    )
    dff = dff.drop("genre", axis=1)
    dff = pd.concat(
        [dff, pd.get_dummies(dff["country"], prefix="c", dummy_na=True)], axis=1
    )
    dff = dff.drop("country", axis=1)
    dff = pd.concat(
        [dff, pd.get_dummies(dff["director"], prefix="dir", dummy_na=True)], axis=1
    )
    dff = dff.drop("director", axis=1)
    dff = dff.fillna(0)
    return dff


def preprocess_user_ratings(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Preprocesses user ratings dataframe.
    :param df: preprocessed metadata dataframe
    :param cols: columns to sub-select for user ratings
    :return: user ratings dataframe
    """
    dff = df[cols]
    dff = pd.concat(
        [dff, pd.get_dummies(dff["tomatoImage"], prefix="img", dummy_na=True)], axis=1
    )
    dff = pd.concat(
        [dff, pd.get_dummies(dff["rated"], prefix="rated", dummy_na=True)], axis=1
    )
    dff.drop("tomatoImage", axis=1, inplace=True)
    dff.drop("rated", axis=1, inplace=True)
    return dff


def preprocess_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses metadata dataframe.
    :param df: metadata dataframe
    :param num_cols: numerical columns
    :return: preprocessed dataframe
    """
    num_cols = [
        "imdbVotes",
        "tomatoRotten",
        "tomatoUserReviews",
        "metascore",
        "tomatoUserRating",
        "tomatoFresh",
        "tomatoReviews",
        "imdbRating",
        "tomatoMeter",
        "tomatoUserMeter",
    ]
    df = df.replace("N/A", np.nan)
    df["released"] = pd.to_datetime(df["released"]).dt.year
    df["released"] = df["released"].interpolate()
    df["DVD"] = pd.to_datetime(df["DVD"]).dt.year
    df[num_cols] = df[num_cols].fillna(0)
    # df["imdbVotes"] = [x.replace(",", "") for x in df["imdbVotes"]]
    # df["country"] = [x.split(",")[0] for x in df["country"]]
    # df["genre"] = [x.split(",")[0] for x in df["genre"]]
    # df["director"] = df["director"].apply(director)
    # df["runtime"] = pd.to_numeric(df["runtime"].apply(min2int))
    # df["BoxOffice"] = pd.to_numeric(df["BoxOffice"].apply(boxOffice))
    df[num_cols] = df[num_cols].apply(pd.to_numeric)
    return df


def main():  # input_filepath, output_filepath
    """ Runs data processing scripts to turn processed data from
        (../processed) into cleaned data ready to be analyzed
        (saved in ../evaluation).
    """
    logger = logging.getLogger(__name__)
    logger.info("Making final datasets from processed data")
    logger.info("Loading datasets.")
    df_audio_dev, df_metadata_dev, df_text_dev, df_visual_dev = load_dev_data()
    df_audio_test, df_metadata_test, df_text_test, df_visual_test = load_test_data()

    # Based on the CoeTraining.csv
    df_labels_dev = (
        pd.read_csv(
            raw_path / "CoeDevLabels.csv",
            usecols=["file_name", "goodforairplanes"],
            # names=["name", "class"],
            # index_col="file_name"
        )
        .rename(columns={"file_name": "name", "goodforairplanes": "class"})
        .set_index("name")
        .sort_index()
    )
    # TODO: Create code which merges CoeTrainingTest.csv and
    # CoeTest.csv to create full list of labels For now I've just
    # copy pasted both into one CSV.
    df_labels_test = (
        pd.read_csv(
            raw_path / "CoeTestLabels.csv",
            usecols=["file_name", "goodforairplanes"],
            # names=["name", "class"],
            # index_col="file_name"
        )
        .rename(columns={"file_name": "name", "goodforairplanes": "class"})
        .set_index("name")
        .sort_index()
    )

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

    df_metadata = join_train_test_datasets(dev_sets["Metadata"], test_sets["Metadata"])
    df_visual = join_train_test_datasets(dev_sets["Visual"], test_sets["Visual"])
    df_text = join_train_test_datasets(dev_sets["Text"], test_sets["Text"])
    df_audio = join_train_test_datasets(dev_sets["Audio"], test_sets["Audio"])

    # Metadata
    logger.info("Creating metadata dataframe.")
    df_metadata = drop_metadata_columns(df_metadata)
    df_metadata = preprocess_metadata(df_metadata)

    # Combine the dataframes to the training sets used in the paper
    logger.info("Creating user ratings dataframe.")
    df_user_ratings = preprocess_user_ratings(df_metadata, user_rating_columns)

    # Create metadata only dataframe
    logger.info("Creating metadata only dataframe.")
    user_rating_columns.remove("train")
    df_metadata_only = preprocess_metadata_only(df_metadata, user_rating_columns)

    # Combine the dataframes to the training sets used in the paper
    logger.info("Combining dev dataframes.")
    dict_dev = {}
    dict_dev["audio"] = df_audio[df_audio["train"] == True]
    dict_dev["user_ratings"] = df_user_ratings[df_user_ratings["train"] == True]
    dict_dev["visual"] = df_visual[df_visual["train"] == True]
    dict_dev["metadata"] = df_metadata_only[df_metadata_only["train"] == True]
    dict_dev["metadata_user_rating"] = pd.concat(
        (dict_dev["metadata"], dict_dev["user_ratings"]), axis=1, sort=False
    )
    dict_dev["text"] = df_text[df_text["train"] == True]
    # TODO: Discuss this below with @PrincMullatahiri
    # dict_dev["jcd"] = df_visual_jcd[df_visual_jcd["train"] == True]
    # dict_dev["jcd_metadata"] = pd.concat(
    #     (dict_dev["metadata"], dict_dev["jcd"]), axis=1, sort=False
    # )

    # Save for feature evaluation
    for k, v in dict_dev.items():
        data = v.copy()
        data = data.drop("train", axis=1)
        data["class"] = df_labels_dev["class"]
        data.to_csv(eval_path / f"{k}_dev.csv", index=False)

    # Combine the dataframes to the test sets used in the paper
    logger.info("Combining test dataframes.")
    dict_test = {}
    dict_test["audio"] = df_audio[df_audio["train"] == False]
    dict_test["user_ratings"] = df_user_ratings[df_user_ratings["train"] == False]
    dict_test["visual"] = df_visual[df_visual["train"] == False]
    dict_test["metadata"] = df_metadata_only[df_metadata_only["train"] == False]
    dict_test["metadata_user_rating"] = pd.concat(
        (dict_test["metadata"], dict_test["user_ratings"]), axis=1, sort=False
    )
    dict_test["text"] = df_text[df_text["train"] == False]
    # TODO: Discuss this below with @PrincMullatahiri
    # dict_test["jcd"] = df_visual_jcd[df_visual_jcd["train"] == False]
    # dict_test["jcd_metadata"] = pd.concat(
    #     (dict_test["metadata"], dict_test["jcd"]), axis=1, sort=False
    # )

    # save for feature evaluation
    for k, v in dict_test.items():
        data = v.copy()
        data = data.drop("train", axis=1)
        data["class"] = df_labels_test["class"]
        data.to_csv(eval_path / f"{k}_test.csv", index=False)

    logger.info("Finished creating dataframes!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # data path
    processed_dir = project_dir / "data/processed"
    # eval path
    eval_path = project_dir / "data/evaluation"
    # raw path
    raw_path = project_dir / "data/raw"

    # user rating columns
    user_rating_columns = [
        "imdbVotes",
        "released",
        "tomatoRotten",
        "tomatoUserReviews",
        "metascore",
        "tomatoUserRating",
        "tomatoImage",
        "tomatoFresh",
        "tomatoReviews",
        "imdbRating",
        "tomatoMeter",
        "rated",
        "tomatoUserMeter",
        "train",
    ]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
