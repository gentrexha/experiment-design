from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np


def main():
    directory_in_str = "..\\..\\data\\raw\\Test\\Metadata"
    pathlist = Path(directory_in_str).glob("**/*.xml")
    tags = {"tags": []}
    for path in pathlist:
        root = ET.parse(str(path)).getroot()

        for elem in root:
            tag = {}
            # BoxOffice 50% missing values
            attributesOfMovie = [
                "year",
                "rated",
                "runtime",
                "genre",
                "language",
                "country",
                "tomatoRating",
                "tomatoReviews",
                "tomatoFresh",
                "tomatoRotten",
                "tomatoUserMeter",
                "tomatoUserRating",
                "tomatoUserReviews",
                "metascore",
                "imdbRating",
                "imdbVotes",
                "tomatoMeter",
                "tomatoImage",
            ]
            for att in attributesOfMovie:
                tag[att] = elem.attrib[att]
            tag["name"] = str(path.name.replace(".xml", ""))
            tags["tags"].append(tag)

    df = pd.DataFrame(tags["tags"])

    df["runtime"] = df["runtime"].str.extract("(\d+)")  # .astype(int)

    df["genre"] = df["genre"].apply(lambda x: x.split(",")[0])
    # df['genre'] = df['genre'].replace(',','')
    df["country"] = df["country"].apply(lambda x: x.split(",")[0])
    df["language"] = df["language"].apply(lambda x: x.split(",")[0])

    # Do one hot encoding for the selected features
    df.loc[df["rated"] == "N/A", "rated"] = "N/A rated"
    df.loc[df["tomatoImage"] == "N/A", "tomatoImage"] = "N/A tomatoImage"
    OHEAttributes = ["rated", "genre", "language", "country", "tomatoImage"]
    for att in OHEAttributes:
        attOHE = pd.get_dummies(df[att])
        df = df.drop(att, axis=1)
        df = df.join(attOHE)

    # Fill missing values with mean
    df["imdbVotes"] = df["imdbVotes"].apply(lambda x: x.replace(",", ""))
    attributesOfMovie = [
        "year",
        "runtime",
        "tomatoRating",
        "tomatoReviews",
        "tomatoFresh",
        "tomatoRotten",
        "tomatoUserMeter",
        "tomatoUserRating",
        "tomatoUserReviews",
        "metascore",
        "imdbRating",
        "imdbVotes",
        "tomatoMeter",
    ]
    for att in attributesOfMovie:
        meanOfAtt = df.loc[df[att] != "N/A", att].astype(float).mean()
        df.loc[df[att] == "N/A", att] = meanOfAtt
        df.loc[df[att].isna(), att] = meanOfAtt

    print(df)

    # One hot encoding per categorical values

    df.to_csv(
        "..\\..\\data\\processed\\metadata_descriptor_test.csv",
        encoding="utf-8",
        index=False,
    )

    # print(audioData)


if __name__ == "__main__":
    main()
