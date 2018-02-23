FROM python:3.6.4
MAINTAINER @wingr

# send SIGQUIT to stop container
STOPSIGNAL SIGQUIT

RUN touch /etc/inside-container

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /src
WORKDIR /src

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      build-essential
