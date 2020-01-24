# -*- coding: utf-8 -*-
import os
import pandas as pd
import click as click
import logging
from pathlib import Path


@click.command()
@click.argument("dev_filepath", type=click.Path(exists=True))
@click.argument("test_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(dev_filepath, test_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Feature preprocessing visual features into dev and test dataframes.")

    visual_col_names = ["name"]
    visual_col_names.extend(["col" + str(i) for i in range(0, 826)])

    dev_names = []
    test_names = []

    for filename in os.listdir(dev_filepath):
        dev_names.append(filename[:-4])
    for filename in os.listdir(test_filepath):
        test_names.append(filename[:-4])

    df_dev = pd.concat(
        [
            pd.read_csv(dev_filepath + filename, header=None).head(1)
            for filename in os.listdir(dev_filepath)
        ],
        ignore_index=True,
    )
    df_test = pd.concat(
        [
            pd.read_csv(test_filepath + filename, header=None).head(1)
            for filename in os.listdir(test_filepath)
        ],
        ignore_index=True,
    )

    # Insert
    df_dev.insert(0, "name", dev_names)
    df_test.insert(0, "name", test_names)

    df_dev.to_csv(
        output_filepath + "visual_descriptor_dev.csv",
        index=False,
        header=visual_col_names,
    )
    df_test.to_csv(
        output_filepath + "visual_descriptor_test.csv",
        index=False,
        header=visual_col_names,
    )

    logger.info("Finished extracting visual features.")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
