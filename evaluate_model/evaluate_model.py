import click
import logging
import pweave
import json
import os
from  pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO)

@click.command()
@click.option('--model-dir')
@click.option('--train-set-dir')
@click.option('--test-set-dir') 
@click.option('--raw-data-dir')
@click.option('--save-dir')
def evaluate_model(model_dir, train_set_dir ,test_set_dir , raw_data_dir, save_dir):
    log = logging.getLogger('evaluate-model')
    save_path= Path(save_dir) / 'document.md'
    
    path_json = {}
    path_json["model_dir"] = model_dir
    path_json["train_set_dir"] = train_set_dir
    path_json["test_set_dir"] = test_set_dir
    path_json["raw_data_dir"] = raw_data_dir
    
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)

    with open(os.getenv('JSON_FILE'), 'w+') as outfile:
        json.dump(path_json, outfile)

    pweave.weave(file = "./document.py", 
                informat = "script", 
                doctype = "markdown", 
                output = save_path, 
                figdir = save_dir)

    log.info(f"Savinng document at {save_dir}")
    shutil.make_archive('./usr/share/data/report.zip', 'zip', save_dir)


if __name__ == '__main__':
    evaluate_model()