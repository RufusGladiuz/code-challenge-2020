from sklearn.svm import SVR
import click
import logging
import pickle
from datetime import datetime
import pandas as pd
logging.basicConfig(level=logging.INFO)


@click.command()
@click.option('--X')
@click.option('--y')
@click.option('--save-path')
def train_model(X, y, save_path:str):

    X = pd.read_parquet(X)
    y = pd.read_parquet(y)
    now = datetime.now()
    model = SVR(C=0.8, gamma="scale", epsilon = 0.1, kernel = "rbf")
    model.fit(X, y)
    time_delta = (datetime.now()-now).seconds/60
    print(f"Training time: {time_delta} min")

    pickle.dumps(model, str(save_path / 'model.sklearn'))


if __name__ == '__main__':
    train_model()