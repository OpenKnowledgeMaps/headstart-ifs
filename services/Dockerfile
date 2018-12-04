FROM python:3.6

MAINTAINER Chris Kittel "web@christopherkittel.eu"

# set working directory
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .
