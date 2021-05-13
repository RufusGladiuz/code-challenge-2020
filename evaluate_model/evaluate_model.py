import click
import logging
import pweave
import json
import os
from  pathlib import Path
import re
import shutil

logging.basicConfig(level=logging.INFO)

_FILE_REGEX = "(?:(?:\/[a-zA-Z_0-9]+)+\/[a-zA-Z_0-9]+\.[a-zA-Z_0-9]+)"

def _clean_up_report(report:str) -> str:
    """Takes a string of the produced report and cleans it.

    Parameters
    ----------
    report: str
        String of the report file
    Returns
    -------
    A String of the cleaned report
    """
    # Finding and replacing all directories to images, as they cant be absolut in a markdown file
    for match in re.findall(_FILE_REGEX, report):
        print(match)
        print(os.path.basename(match))
        report = report.replace(match, os.path.basename(match))

    report = report.replace(")\\", ")")

    return report

@click.command()
@click.option('--model-path')
@click.option('--test-set-path') 
@click.option('--raw-data-path')
@click.option('--out-dir')
def evaluate_model(model_path:str, test_set_path:str, raw_data_path:str, out_dir:str) -> None:
    """Takes the saved model and the test data set to evaluate the model and create a report

    Parameters
    ----------
    model_path: str
        Path to the model file including the filename
    test_set_path: str
        Path to test dataset including the filename
    raw_data_path:
        Path to raw datasat inclding the filename
    out_dir:
        Directory to save files to, no filename included
    Returns
    -------
    None
    """
    log = logging.getLogger('evaluate-model')
    save_path = Path(out_dir) / 'document.md'

    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)
    
    # creating a json file of file paths to hand to the document execution
    path_json = {}
    path_json["model_dir"] = model_path
    path_json["test_set_dir"] = test_set_path
    path_json["raw_data_dir"] = raw_data_path
    
    with open(os.getenv('JSON_FILE'), 'w+') as outfile:
        json.dump(path_json, outfile)
   
    # Running the report creation 
    pweave.weave(file = "./document.py", 
                informat = "script", 
                doctype = "markdown", 
                output = save_path, 
                figdir = out_dir)
    
    # Cleaning up and resaving the file
    clean_report = _clean_up_report(open(save_path, "r").read())
    open(save_path, "w+").write(clean_report)

    # Creating a zip file out of the report files, as it is required by the tast
    log.info(f"Saving document at {out_dir}")
    shutil.make_archive(Path(out_dir) / '../report.zip', 'zip', out_dir)

if __name__ == '__main__':
    evaluate_model()