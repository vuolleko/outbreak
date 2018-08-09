FROM ubuntu:bionic
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y libboost-all-dev python3.6-dev libeigen3-dev python3-scipy jupyter python3-matplotlib
RUN pip3 install elfi
