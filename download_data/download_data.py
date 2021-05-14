from pathlib import Path

import click
import logging
import urllib.request

import os

import json

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option('--name')
@click.option('--url')
@click.option('--out-dir')
def download_data(name, url, out_dir):
    """Download a csv file and save it to local disk.

    Parameters
    ----------
    name: str
        name of the csv file on local disk, without '.csv' suffix.
    url: str
        remote url of the csv file.
    out_dir:
        directory where file should be saved to.

    Returns
    -------
    None
    """
    log = logging.getLogger('download-data')
    assert '.csv' not in name, f'Received {name}! ' \
        f'Please provide name without csv suffix'

    out_path = Path(out_dir) / f'{name}.csv'

    log.info('Downloading dataset')
    log.info(f'Will write to {out_path}')

    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)

    log.info(f"Lable Column: {os.getenv('LABLE_COL')}")
    urllib.request.urlretrieve(url, str(out_path))

    # Saving path of output file to json
    path_json = {}
    path_json["raw_data_dir"] = str(out_path)

    with open(os.getenv("SHARED_FILES"), 'w+') as files:
        json.dump(path_json, files)


if __name__ == '__main__':
    download_data()
