FROM ubuntu:eoan
RUN apt update && apt install -y --no-install-recommends \
    build-essential ninja-build cmake \
    llvm-9-dev libclang-9-dev \
    python3 python3-pip libpython3-dev \
    python3-setuptools python3-wheel \
    libboost-dev git curl && apt clean
RUN python3 -m pip install --upgrade pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
