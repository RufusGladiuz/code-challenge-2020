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
    concluded_run = False
    fname = luigi.Parameter(default='wine_dataset')
    out_dir = luigi.Parameter(default='/usr/share/data/raw/')
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

    def run(self):
        super(DownloadData, self).run()
        self.concluded_run = True

    def output(self):
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        return luigi.LocalTarget(
            path=str(out_dir/f'{self.fname}.csv')
        )

    def complete(self):
        return self.concluded_run

class MakeDatasets(DockerTask):

    out_dir = luigi.Parameter(default='/usr/share/data/refined/')
    concluded_run = False

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
        pass
    
    def run(self):
        super(MakeDatasets, self).run()
        self.concluded_run = True

    def output(self):
        target =  luigi.LocalTarget(
            path=str(Path(self.out_dir) / '.SUCCESS')
        )
        print(target.exists())
        print(self.concluded_run)
        print(str(Path(self.out_dir) / '.SUCCESS'))
        return target

    def complete(self):
        return self.concluded_run


class TrainModel(DockerTask):

    out_dir = luigi.Parameter(default='/usr/share/data/model/')
    concluded_run = False

    @property
    def image(self):
        return f'code-challenge/train-model:{VERSION}'

    def requires(self):
        return MakeDatasets()

    @property
    def command(self):

        print(self.input().path)
        print(type(self.input().path))
        directory = os.path.dirname(self.input().path)
        train_set_path = directory + '/train.parquet'
        print(train_set_path)
        return [
            'python', 'train_model.py',
            '--X-train', train_set_path,
            '--save-dir', self.out_dir 
        ]
        pass

    def run(self):
        super(TrainModel, self).run()
        self.concluded_run = True

    def output(self):
        return luigi.LocalTarget(
            path=str(Path(self.out_dir) / 'model.sklearn')
        )

    def complete(self):
        return self.concluded_run
