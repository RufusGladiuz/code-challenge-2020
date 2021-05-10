from sklearn.svm import SVR
import click
import logging
import pickle
from datetime import datetime
import pandas as pd

from pathlib import Path

logging.basicConfig(level=logging.INFO)


@click.command()
@click.option('--X-train')
@click.option('--save-dir')
def train_model(x_train, save_dir):
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger('train-model')
    x_train = pd.read_parquet(x_train)
    y_train = x_train["points"]
    x_train = x_train.drop(columns = ["points"])
    now = datetime.now()
    model = SVR(C=0.8, gamma="scale", epsilon = 0.1, kernel = "rbf")
    model.fit(x_train, y_train)

    time_delta = (datetime.now()-now).seconds/60
    log.info(f"Training time: {time_delta} min")
    log.info(f"Savinng model at {save_path}")

    with open(save_path / 'model.sklearn', 'wb+') as output:
        pickle.dump(model, output, protocol=None)


if __name__ == '__main__':
    train_model()