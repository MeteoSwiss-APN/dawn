FROM ubuntu:focal
RUN apt update && apt install -y --no-install-recommends \
    apt-transport-https ca-certificates \
    gnupg software-properties-common curl && apt clean
# Add CMake repo
RUN curl -L https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add -
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal-rc main'
RUN apt-get install kitware-archive-keyring && apt clean
RUN apt update && apt install -y --no-install-recommends \
    build-essential ninja-build cmake git openssh-client curl \
    llvm-9-dev libclang-9-dev libclang-cpp9 \
    python3 python3-pip libpython3-dev \
    python3-setuptools python3-wheel \
    libboost-dev && apt clean
RUN python3 -m pip install --upgrade pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
