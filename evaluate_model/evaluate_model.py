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
@click.option('--out-dir')
def evaluate_model(out_dir:str) -> None:
    """Takes the saved model and the test data set to evaluate the model and create a report

    Parameters
    ----------
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