from sklearn.svm import SVR
import click
import logging
import pickle
from datetime import datetime
import pandas as pd
import os
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option('--X-train')
@click.option('--save-dir')
def train_model(x_train, save_dir):
    """Takes a train data set and trains a svm model

    Parameters
    ----------
        x_train: str
            The preprocessed train dataset
        save_dir:
            Direcotry in which the model gets saved
    Returns
    -------
    None
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger('train-model')

    x_train = pd.read_parquet(x_train)

    # Split dataset into labels and features
    y_train = x_train[os.getenv('LABLE_COL')]
    x_train = x_train.drop(columns=[os.getenv('LABLE_COL')])
    now = datetime.now()

    # Using a SVM as the choosen model
    model = SVR(C=0.8, gamma="scale", epsilon=0.1, kernel="rbf")
    model.fit(x_train, y_train)

    time_delta = (datetime.now() - now).seconds / 60

    log.info(f"Training time: {time_delta} min")
    log.info(f"Saving model at {save_path}")

    with open(save_path / 'model.sklearn', 'wb+') as output:
        pickle.dump(model, output, protocol=None)

    # Appending new directory to json file
    path_json = json.load(open(os.getenv("SHARED_FILES"), "r"))
    path_json["model_dir"] = str(save_path / 'model.sklearn')

    with open(os.getenv('SHARED_FILES'), 'w') as outfile:
        json.dump(path_json, outfile)


if __name__ == '__main__':
    train_model()
