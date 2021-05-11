FROM python:3.9
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV JSON_FILE=./files.json
ENV PYTHONPATH=/opt/orchestrator:$PYTHONPATH

COPY . /opt/train_model/

RUN pip install --upgrade pip
RUN python -m pip install -r /opt/train_model/requirements.txt

WORKDIR /opt/train_model/
