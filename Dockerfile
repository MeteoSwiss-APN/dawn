# IMAGE_NAME needs to be an image that has the following dependencies:
# - cmake & ninja
# - llvm & clang
# - python3: pip, setuptools, wheel
# - boost
ARG IMAGE_NAME
FROM ${IMAGE_NAME} AS dawn-env
RUN curl -L https://github.com/protocolbuffers/protobuf/releases/download/v3.10.1/protobuf-all-3.10.1.tar.gz | \
    tar -xz -C /usr/src
RUN cmake -S /usr/src/protobuf-3.10.1/cmake -B /usr/src/protobuf-3.10.1/build \
    -Dprotobuf_BUILD_EXAMPLES=OFF \
    -Dprotobuf_BUILD_TESTS=OFF \
    -Dprotobuf_INSTALL_EXAMPLES=OFF \
    -Dprotobuf_BUILD_PROTOC_BINARIES=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DBUILD_SHARED_LIBS=ON \
    -GNinja && \
    cmake --build /usr/src/protobuf-3.10.1/build --target install -j $(nproc) && \
    rm -rf /usr/src/protobuf-3.10.1/build
RUN cd /usr/src/protobuf-3.10.1/python && python setup.py build && \
    mv /usr/src/protobuf-3.10.1/python/build/lib/google /usr/local/lib/google
RUN curl -L https://github.com/GridTools/gridtools/archive/v1.0.4.tar.gz | \
    tar -xz -C /usr/src
RUN cmake -S /usr/src/gridtools-1.0.4 -B /usr/src/gridtools-1.0.4/build \
    -DBUILD_TESTING=OFF \
    -DINSTALL_TOOLS=OFF \
    -DGT_INSTALL_EXAMPLES=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -GNinja && \
    cmake --build /usr/src/gridtools-1.0.4/build -j $(nproc) --target install && \
    rm -rf /usr/src/gridtools-1.0.4/build

FROM dawn-env AS dawn
COPY . /usr/src/dawn
RUN /usr/src/dawn/scripts/build-and-test -j $(nproc) -i /usr/local
