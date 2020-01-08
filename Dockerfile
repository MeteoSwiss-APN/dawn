FROM ubuntu:rolling AS dawn-build
RUN apt update && apt install -y \
    build-essential ninja-build cmake \
    python3 libpython-dev python3-setuptools \
    llvm-9-dev libclang-9-dev \
    libboost-dev && apt clean
COPY . /usr/src/dawn
RUN mkdir -p /usr/src/dawn/build
RUN cmake -S /usr/src/dawn -B /usr/src/dawn/build \
    -DBUILD_TESTING=ON \
    -DCMAKE_PREFIX_PATH=/usr/lib/llvm-9 \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -GNinja
RUN cmake --build /usr/src/dawn/build --target install -- -j$(nproc)

FROM ubuntu:rolling AS dawn-exec
LABEL Name=gtclang
COPY --from=dawn-build /usr/local /usr/local
# gtclang built above links to libLLVM-9 dynamically, and we need the C++ headers
RUN apt update && apt install -y llvm-9-dev libclang-9-dev && apt clean
COPY . /usr/src/dawn
CMD /usr/local/bin/gtclang
