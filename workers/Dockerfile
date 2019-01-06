FROM python:3.6

MAINTAINER Chris Kittel "christopher.kittel@openknowledgemaps.org"

# set working directory
WORKDIR /workers

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download de_core_news_sm
