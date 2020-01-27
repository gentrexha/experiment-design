from pathlib import Path
import pandas as pd


def main():
    make_text_dataframe(text_dir_dev, "dev", audio_dir_dev)
    make_text_dataframe(text_dir_test, "test", audio_dir_test)


def make_text_dataframe(dir, name, file_dir):
    df = pd.read_csv(
        dir / f"tdf_idf_{name}.csv", na_values=["NaN"]
    )
    # df = df.fillna(0)
    dfcolumns = df.columns
    df = df.T
    df.columns = dfcolumns
    df = df.reset_index()
    df = df.drop(["index"], axis=1)
    df = df.dropna()
    pathlist = Path(file_dir).glob("**/*.csv")
    singleName = []
    for path in pathlist:
        singleName.append(str(path.name.replace(".csv", "")))
    df["name"] = singleName
    df = df.fillna(0)
    df = df.replace("", 0)
    df.to_csv(
        processed_dir / f"text_descriptor_{name}.csv",
        encoding="utf-8",
        index=False,
    )


if __name__ == "__main__":
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # data path
    processed_dir = project_dir / "data/processed"
    text_dir_dev = project_dir / "data/raw/Dev_Set/text_descriptors"
    text_dir_test = project_dir / "data/raw/Test_Set/text_descriptors"
    audio_dir_test = project_dir / "data/raw/Test_Set/audio_descriptors"
    audio_dir_dev = project_dir / "data/raw/Dev_Set/audio_descriptors"

    main()
