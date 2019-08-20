FROM ubuntu:latest
LABEL maintainer="Carlos Osuna <Carlos.Osuna@meteoswiss.ch>"
LABEL description="Dawn build stage"
# Setup base environment
RUN apt-get update && apt-get install -y \
    build-essential \
    bash \
    make \
    cmake \
    autoconf \
    automake \
    libtool \
    pkg-config \
    curl \
    tar \
    git \
    clang \
    libboost-dev \
    python \
    unzip \
    python3 \
    python3-pip

RUN pip3 install setuptools

# Download and install Google protobuf 3.4
RUN cwd=`pwd`/protobuf/ && \
    git clone https://github.com/google/protobuf.git && \
    cd protobuf && \
    git checkout v3.4.0 && \
    git submodule update --init --recursive && \
    export cwd=`pwd` && \
    mkdir build && \ 
    cd build && \
    cmake ../cmake/ -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX=/usr/share/protobuf/ -Dprotobuf_BUILD_SHARED_LIBS=ON && \
    make -j3 install

RUN cwd=`pwd`/protobuf/ && \
    export PROTOC=${cwd}/build/protoc && \
    cd ${cwd}/python && \
    python3 setup.py build && \
    cp -r ${cwd}/python /usr/share/protobuf/python

# Download yoda
RUN git clone https://github.com/MeteoSwiss-APN/yoda.git

# Download and install dawn
RUN git clone https://github.com/MeteoSwiss-APN/dawn.git && \
    cd dawn && \
    mkdir build && \
    cd build && \
    cmake .. -DYODA_ROOT=/yoda -DProtobuf_DIR=/usr/share/protobuf/lib/cmake/protobuf/ && \
    make install -j3

CMD ["/bin/bash"]

