import click
import dask.dataframe as dd
import numpy as np
from distributed import Client
from pathlib import Path

from prepare_dataset import process_data 

def _save_datasets(train, test, outdir: Path):
    """Save data sets into nice directory structure and write SUCCESS flag."""
    out_train = outdir / 'train.parquet/'
    out_test = outdir / 'test.parquet/'
    flag = outdir / '.SUCCESS'

    train.to_parquet(str(out_train))
    test.to_parquet(str(out_test))

    flag.touch()


@click.command()
@click.option('--in-csv')
@click.option('--out-dir')
def make_datasets(in_csv, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    to_one_hot_encode = ["taster_name", "country", "variety", "province"]
    drop_duplicates = ['description','title']
    drop = ["Unnamed: 0", "designation", "region_1", "region_2", "taster_twitter_handle", "title", "winery"]
    
    X_train, X_test, y_train, y_test = process_data(load_dir = in_csv, 
                                                   label_column = "points", 
                                                   text_column = "description",
                                                   test_size = 0.2,
                                                   to_one_hot_encode = to_one_hot_encode,
                                                   drop_duplicates = drop_duplicates,
                                                   drop = drop,
                                                   tf_idf_cutoff = 0,
                                                   fill_mean = ["price"],
                                                   normalize = True)
    X_train["points"] = y_train.to_numpy()
    X_test["points"] = y_test.to_numpy()

    _save_datasets(X_train, X_test, out_dir)


if __name__ == '__main__':
    make_datasets()
