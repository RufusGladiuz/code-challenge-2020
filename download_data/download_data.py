from pathlib import Path

import click
import logging
import urllib.request

import os

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
    log.info(f"Path exists: {os.path.exists(out_dir)}")

    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)

    if os.path.exists(out_dir) == False:
        os.mkdir(Path(out_dir) / 'TEST_TEST_TEST')
    
    log.info(f"Also created exists: {Path(out_dir) / 'TEST_TEST_TEST'}")
    urllib.request.urlretrieve(url, out_path)


if __name__ == '__main__':
    download_data()
