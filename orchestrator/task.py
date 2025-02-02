import logging

import luigi
import os
from pathlib import Path

from util import DockerTask

VERSION = os.getenv('PIPELINE_VERSION', '0.1')

class Debug(DockerTask):
    """Use this task with appropriate image to debug things."""

    @property
    def image(self):
        return f'code-challenge/download-data:{VERSION}'

    @property
    def command(self):
        return [
            'sleep', '3600'
        ]


class DownloadData(DockerTask):
    """Initial pipeline task downloads dataset."""
    fname = luigi.Parameter(default='wine_dataset')
    out_dir = luigi.Parameter(default=os.getenv("SHARE") + 'raw/')
    url = luigi.Parameter(
        default='https://github.com/datarevenue-berlin/code-challenge-2019/'
                'releases/download/0.1.0/dataset_sampled.csv'
    )

    @property
    def image(self):
        return f'code-challenge/download-data:{VERSION}'

    @property
    def command(self):
        return [
            'python', 'download_data.py',
            '--name', self.fname,
            '--url', self.url,
            '--out-dir', self.out_dir
        ]

    def output(self):
        out_dir = Path(self.out_dir)

        return luigi.LocalTarget(
            path=str(out_dir/f'{self.fname}.csv')
        )


class MakeDatasets(DockerTask):
    """Cleans and splits the dataset into train and test"""
    out_dir = luigi.Parameter(default='/usr/share/data/refined/')
    
    @property
    def image(self):
        return f'code-challenge/make-dataset:{VERSION}'

    def requires(self):
        return DownloadData()

    @property
    def command(self):
        return [
            'python', 'dataset.py',
            '--in-csv', self.input().path,
            '--out-dir', self.out_dir
        ]
    

    def output(self):
        return luigi.LocalTarget(path=str(Path(self.out_dir) / '.SUCCESS'))


class TrainModel(DockerTask):
    """Trains the model using the train dataset created by the MakeDataSet Task"""
    out_dir = luigi.Parameter(default='/usr/share/data/model/')

    @property
    def image(self):
        return f'code-challenge/train-model:{VERSION}'

    def requires(self):
        return MakeDatasets()

    @property
    def command(self):
        directory = os.path.dirname(self.input().path)
        train_set_path = directory + '/train.parquet'
        return [
            'python', 'train_model.py',
            '--X-train', train_set_path,
            '--save-dir', self.out_dir 
        ]

    def output(self):
        return luigi.LocalTarget(path=str(Path(self.out_dir) / 'model.sklearn'))

class EvaluateModel(DockerTask):
    """Evaluates the model and creates a markdown report"""
    out_dir = luigi.Parameter(default='/usr/share/data/report/')

    @property
    def image(self):
        return f'code-challenge/evaluate-model:{VERSION}'

    def requires(self):
        return TrainModel()

    @property
    def command(self):
        return [
            'python', 'evaluate_model.py',
            '--out-dir', self.out_dir 
        ]

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / 'report.zip')
        )
