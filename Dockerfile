FROM ubuntu:latest

RUN apt-get -yqq update
RUN apt-get -yqq install git

RUN apt-get -yqq install python3    
RUN apt-get -yqq install python3-pip
RUN apt-get -yqq update

ARG GITHUB_USER=neelnanda-io
RUN git clone https://github.com/$GITHUB_USER/TransformerLens.git

WORKDIR /TransformerLens

RUN pip install -e . 
RUN pip install pytest
