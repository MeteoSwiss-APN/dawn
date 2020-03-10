# IMAGE_NAME needs to be an image that has the following dependencies:
# - cmake & ninja
# - llvm & clang
# - python3: pip, setuptools, wheel
# - boost
ARG IMAGE_NAME
FROM ${IMAGE_NAME} AS dawn-env
RUN curl -L https://github.com/protocolbuffers/protobuf/releases/download/v3.10.1/protobuf-all-3.10.1.tar.gz | \
    tar -xz -C /usr/src
RUN mkdir -p /usr/src/protobuf-3.10.1/build
RUN cmake -S /usr/src/protobuf-3.10.1/cmake -B /usr/src/protobuf-3.10.1/build \
    -Dprotobuf_BUILD_EXAMPLES=OFF \
    -Dprotobuf_BUILD_TESTS=OFF \
    -Dprotobuf_INSTALL_EXAMPLES=OFF \
    -Dprotobuf_BUILD_PROTOC_BINARIES=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DBUILD_SHARED_LIBS=ON \
    -GNinja
RUN cmake --build /usr/src/protobuf-3.10.1/build --target install -j $(nproc)
RUN rm -rf /usr/src/protobuf-3.10.1/build
RUN cd /usr/src/protobuf-3.10.1/python && python setup.py build
RUN mv /usr/src/protobuf-3.10.1/python/build/lib/google /usr/local/lib/google
RUN curl -L https://github.com/GridTools/gridtools/archive/v1.0.4.tar.gz | \
    tar -xz -C /usr/src
RUN mkdir -p /usr/src/gridtools-1.0.4/build
RUN cmake -S /usr/src/gridtools-1.0.4 -B /usr/src/gridtools-1.0.4/build \
    -DBUILD_TESTING=OFF \
    -DINSTALL_TOOLS=OFF \
    -DGT_INSTALL_EXAMPLES=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -GNinja
RUN cmake --build /usr/src/gridtools-1.0.4/build -j $(nproc) --target install
RUN rm -rf /usr/src/gridtools-1.0.4/build

FROM dawn-env AS dawn
COPY . /usr/src/dawn
RUN mkdir -p /usr/src/dawn/build
RUN cmake -S /usr/src/dawn -B /usr/src/dawn/build \
    -DBUILD_TESTING=ON \
    -DCMAKE_PREFIX_PATH=/usr/lib/llvm-9 \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DGridTools_DIR=/usr/local/lib/cmake \
    -DPROTOBUF_PYTHON_DIR=/usr/local/lib/python3.7/dist-packages \
    -GNinja
RUN cmake --build /usr/src/dawn/build -j $(nproc) --target install
RUN python -m pip install -e /usr/src/dawn/dawn
RUN cd /usr/src/dawn/build && ctest -j$(nproc) --progress --output-on-failure
RUN rm -rf /usr/src/dawn/build
