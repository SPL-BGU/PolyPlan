FROM ubuntu:22.04

# Install required packages
RUN apt-get update && apt-get -y install openjdk-8-jdk xvfb mesa-utils x11-xserver-utils xdotool gosu sudo acl curl zip unzip build-essential && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN apt-get update && apt-get -y install software-properties-common
RUN apt-get update && add-apt-repository -y ppa:deadsnakes/ppa -y

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Jerusalem
RUN apt-get update && apt install -y python3.8

RUN apt-get install -y python3.8-distutils
RUN curl https://bootstrap.pypa.io/get-pip.py | sudo python3.8

RUN pip install setuptools==66
RUN pip install wheel==0.38.4
RUN pip install protobuf==3.20.3

ADD PolyPlan /PolyPlan
RUN cd /PolyPlan && pip install -r requirements.txt

ADD PAL /PAL
RUN cd /PAL && ./gradlew --continue --project-cache-dir /tmp/gradle-cache -g /tmp/client-home runclient || true

