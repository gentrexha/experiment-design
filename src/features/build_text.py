from pathlib import Path
import pandas as pd


def main():
    df = pd.read_csv(
        "..\\..\\data\\raw\\Test\\Text\\tdf_idf_test.csv", na_values=["NaN"]
    )
    # df = df.fillna(0)
    dfcolumns = df.columns
    df = df.T
    df.columns = dfcolumns
    df = df.reset_index()
    df = df.drop(["index"], axis=1)
    df = df.dropna()
    directory_in_str = "..\\..\\data\\raw\\Test\\Audio"
    pathlist = Path(directory_in_str).glob("**/*.csv")
    singleName = []
    for path in pathlist:
        singleName.append(str(path.name.replace(".csv", "")))

    df["name"] = singleName
    df.to_csv(
        "..\\..\\data\\processed\\text_descriptor_test.csv",
        encoding="utf-8",
        index=False,
    )


if __name__ == "__main__":
    main()
