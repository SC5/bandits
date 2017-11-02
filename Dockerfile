FROM centos:7

MAINTAINER max.pagels@sc5.io

RUN yum install epel-release -y
RUN yum install -y https://centos7.iuscommunity.org/ius-release.rpm

RUN yum upgrade -y
RUN yum install -y gcc \
  python36u-devel \
  python36u-debug \
  python36u-setuptools \
  python36u-pip \
  python36u-libs \
  libevent-devel

RUN pip3.6 install numpy \
  scipy \
  scikit-learn \
  sanic \
  apscheduler

ADD . /bandit
WORKDIR /bandit

EXPOSE 8000
CMD ["python3.6", "server.py"]
