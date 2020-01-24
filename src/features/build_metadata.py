from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np


def main():
    generate_metadata(dev_dir, "dev")
    generate_metadata(test_dir, "test")


def generate_metadata(data_dir, name: str) -> None:
    # directory_in_str = "..\\..\\data\\raw\\Dev\\Metadata"
    pathlist = Path(data_dir).glob("**/*.xml")
    tags = {"tags": []}
    for path in pathlist:
        root = ET.parse(str(path)).getroot()

        for elem in root:
            tag = {}
            # BoxOffice 50% missing values
            for att in attributesOfMovie:
                tag[att] = elem.attrib[att]
            tag["name"] = str(path.name.replace(".xml", ""))
            tags["tags"].append(tag)

    df = pd.DataFrame(tags["tags"])
    df["runtime"] = df["runtime"].str.extract("(\d+)")  # .astype(int)
    df["BoxOffice"] = df["BoxOffice"].str.extract("(\d+)")
    df["genre"] = df["genre"].apply(lambda x: x.split(",")[0])
    df["director"] = df["director"].apply(lambda x: x.split(",")[0])
    # df['genre'] = df['genre'].replace(',','')
    df["country"] = df["country"].apply(lambda x: x.split(",")[0])
    df["language"] = df["language"].apply(lambda x: x.split(",")[0])
    df["imdbVotes"] = df["imdbVotes"].apply(lambda x: x.replace(",", ""))
    # TODO: Discuss this with @PrincMullatahiri
    # df = df.replace("N/A", 0)
    # df = df.fillna(0)
    # print(f"{df.isna().sum()}")
    df.to_csv(
        processed_dir / f"metadata_descriptor_{name}.csv",
        encoding="utf-8",
        index=False,
    )


if __name__ == "__main__":
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # data path
    dev_dir = project_dir / "data/raw/Dev_Set/XML"
    test_dir = project_dir / "data/raw/Test_Set/XML"
    processed_dir = project_dir / "data/processed"

    attributesOfMovie = [
        "title",
        "year",
        "rated",
        "released",
        "runtime",
        "genre",
        "director",
        "language",
        "country",
        "awards",
        "poster",
        "metascore",
        "imdbRating",
        "imdbVotes",
        "imdbID",
        "type",
        "tomatoMeter",
        "tomatoImage",
        "tomatoRating",
        "tomatoReviews",
        "tomatoFresh",
        "tomatoRotten",
        "tomatoConsensus",
        "tomatoUserMeter",
        "tomatoUserRating",
        "tomatoUserReviews",
        "DVD",
        "BoxOffice",
        "Production",
        "Website",
    ]

    main()
