FROM nvcr.io/nvidia/cuda:10.1-devel-ubuntu18.04
RUN apt update && apt install -y --no-install-recommends \
    apt-transport-https ca-certificates \
    gnupg software-properties-common curl && apt clean
# Add CMake repo
RUN curl -L https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add -
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic-rc main'
RUN apt-key --keyring /etc/apt/trusted.gpg del C1F34CDD40CD72DA
# Add new toolchain repo
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
# Install dependencies
RUN apt update && apt install -y --no-install-recommends \
    build-essential openssh-client git \
    kitware-archive-keyring cmake ninja-build \
    gcc-8 g++-8 \
    llvm-9-dev libclang-9-dev \
    python3 libpython3-dev python3-pip python3-setuptools python3-wheel \
    libboost-dev && apt clean
RUN python3 -m pip install --upgrade pip
# Set defaults
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 90 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-8 \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-8
RUN ln -s /usr/local/lib/python3.6 /usr/local/lib/python
