FROM ubuntu:bionic
RUN apt-get update && apt-get install -y libboost-all-dev python3.6-dev libeigen3-dev python3-scipy
