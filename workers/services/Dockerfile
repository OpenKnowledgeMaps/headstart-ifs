FROM python:3.6

MAINTAINER Chris Kittel "christopher.kittel@openknowledgemaps.org"

# set working directory
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .
