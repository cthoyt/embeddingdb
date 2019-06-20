FROM python:3.7
MAINTAINER Charles Tapley Hoyt "cthoyt@gmail.com"

RUN pip install --upgrade pip
RUN pip install gunicorn

COPY . /app
WORKDIR /app
RUN pip install .[web]
