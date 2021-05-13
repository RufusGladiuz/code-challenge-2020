import click
import dask.dataframe as dd
import numpy as np
from distributed import Client
from pathlib import Path
import os
import logging

from process_settings import ProcessSettings
import json
from types import SimpleNamespace

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
def make_datasets(in_csv:str, out_dir:str) -> None:
    """Takes a raw dataset and prepares and splits it to prepare it for model training

    Parameters
    ----------
        in_csv: str
            Path to raw datafile including filename
        out_dir:
            Directory to save files to, no filename included
    Returns
    -------
    None
    """
    log = logging.getLogger('make-dataset')
    out_dir = Path(out_dir)

    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)
    
    # Load processing settings
    settings = None
    if os.path.exists("setting.json") == True:
        settings = json.loads("setting.json", object_hook=lambda d: SimpleNamespace(**d))
    else:
        settings = ProcessSettings(label_column = "labels")
    
    # Process dataset
    X_train, X_test, y_train, y_test = process_data(load_dir = in_csv,
                                                    setting = settings)

      
    X_train[settings.label.column] = y_train.to_numpy()
    X_test[settings.label.column] = y_test.to_numpy()
    
    log.info(f"Saving datasets at {out_dir}")
    _save_datasets(X_train, X_test, out_dir)

if __name__ == '__main__':
    make_datasets()
