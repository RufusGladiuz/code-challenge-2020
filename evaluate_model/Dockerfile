FROM python:3.9
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Adding a env variable of a json file to share directorys between the pweave doc file and the usual scripts
ENV JSON_FILE=./files.json
ENV PYTHONPATH=/opt/orchestrator:$PYTHONPATH

COPY . /opt/evaluate_model/

RUN python -m pip install -r /opt/evaluate_model/requirements.txt

WORKDIR /opt/evaluate_model/
