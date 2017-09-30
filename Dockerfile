FROM centos:7

MAINTAINER max.pagels@sc5.io

RUN yum install epel-release -y
RUN yum install -y https://centos7.iuscommunity.org/ius-release.rpm

RUN yum upgrade -y
RUN yum install -y gcc \
  python35u-devel \
  python35u-debug \
  python35u-setuptools \
  python35u-pip \
  python35u-libs \
  libevent-devel

RUN pip3.5 install numpy \
  scipy \
  scikit-learn \
  sanic

ADD . /bandit
WORKDIR /bandit

EXPOSE 8000
CMD ["python3.5", "server.py"]
