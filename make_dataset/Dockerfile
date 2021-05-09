FROM drtools/dask:latest
COPY . /opt/make_dataset

USER root

RUN pip install --upgrade pip
RUN pip install -r /opt/make_dataset/requirements.txt
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader wordnet

WORKDIR /opt/make_dataset
