FROM nvcr.io/nvidia/cuda:10.1-devel-ubuntu18.04
RUN apt update && apt install -y --no-install-recommends \
    apt-transport-https ca-certificates gnupg software-properties-common wget
# Add CMake repo
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic-rc main'
RUN apt-key --keyring /etc/apt/trusted.gpg del C1F34CDD40CD72DA
# Add new toolchain repo
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
# Install dependencies
RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    kitware-archive-keyring cmake ninja-build \
    gcc-9 g++-9 \
    llvm-9-dev libclang-9-dev \
    python3 libpython3-dev python3-pip python3-setuptools python3-wheel \
    libboost-dev git curl && apt clean
RUN python3 -m pip install --upgrade pip
# Set defaults
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-9 \
    --slave /usr/bin/gcov gcov /usr/bin/gcov-9
