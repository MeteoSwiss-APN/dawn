# IMAGE needs to be an image that already has the following dependencies:
# - cmake & ninja
# - llvm & clang
# - python3: pip, setuptools, wheel
# - boost
ARG IMAGE
FROM $IMAGE
# ---------------------- ECBuild ----------------------
RUN curl -L https://github.com/ecmwf/ecbuild/archive/3.3.0.tar.gz | \
    tar -xz -C /usr/src
ENV ECBUILD_BIN /usr/src/ecbuild-3.3.0/bin/ecbuild
# ---------------------- ECKit ----------------------
RUN curl -L https://github.com/ecmwf/eckit/archive/1.4.7.tar.gz | \
    tar -xz -C /usr/src
RUN mkdir -p /usr/src/eckit-1.4.7/build && cd /usr/src/eckit-1.4.7/build && \
    ${ECBUILD_BIN} \
    -DCMAKE_INSTALL_PREFIX=/usr/local/eckit \
    -DCMAKE_BUILD_TYPE=Release \
    -GNinja -- ../ && \
    cmake --build . -j $(nproc) --target install && rm -rf /usr/src/eckit-1.4.7/build
# ---------------------- Atlas ----------------------
RUN curl -L https://github.com/ecmwf/atlas/archive/0.19.0.tar.gz | \
    tar -xz -C /usr/src
RUN mkdir -p /usr/src/atlas-0.19.0/build && cd /usr/src/atlas-0.19.0/build && \
    ${ECBUILD_BIN} \
    -DCMAKE_INSTALL_PREFIX=/usr/local/atlas \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_ATLAS_RUN=OFF \
    -DECKIT_PATH=/usr/local/eckit \
    -GNinja -- ../ && \
    cmake --build . -j $(nproc) --target install && rm -rf /usr/src/atlas-0.19.0/build
# ---------------------- Protobuf ----------------------
RUN curl -L https://github.com/protocolbuffers/protobuf/releases/download/v3.10.1/protobuf-all-3.10.1.tar.gz | \
    tar -xz -C /usr/src
# These files seem to have a high UID/GID by default, so update this
RUN chown root:root /usr/src/protobuf-3.10.1 -R
RUN cmake -S /usr/src/protobuf-3.10.1/cmake -B /usr/src/protobuf-3.10.1/build \
    -Dprotobuf_BUILD_EXAMPLES=OFF \
    -Dprotobuf_BUILD_TESTS=OFF \
    -Dprotobuf_INSTALL_EXAMPLES=OFF \
    -Dprotobuf_BUILD_PROTOC_BINARIES=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local/protobuf \
    -DBUILD_SHARED_LIBS=ON \
    -GNinja && \
    cmake --build /usr/src/protobuf-3.10.1/build --target install -j $(nproc) && \
    rm -rf /usr/src/protobuf-3.10.1/build
RUN cd /usr/src/protobuf-3.10.1/python && \
    PROTOC=/usr/local/protobuf/bin/protoc python setup.py build && \
    mkdir -p /usr/local/protobuf/lib/python && \
    mv /usr/src/protobuf-3.10.1/python/build/lib/google /usr/local/protobuf/lib/python/google
# # ---------------------- GridTools ----------------------
# RUN curl -L https://github.com/GridTools/gridtools/archive/v1.0.4.tar.gz | \
#     tar -xz -C /usr/src
# RUN cmake -S /usr/src/gridtools-1.0.4 -B /usr/src/gridtools-1.0.4/build \
#     -DBUILD_TESTING=OFF \
#     -DINSTALL_TOOLS=OFF \
#     -DGT_INSTALL_EXAMPLES=OFF \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_INSTALL_PREFIX=/usr/local/gridtools \
#     -GNinja && \
#     cmake --build /usr/src/gridtools-1.0.4/build -j $(nproc) --target install && \
#     rm -rf /usr/src/gridtools-1.0.4/build
# Other python dependencies for using and testing dawn
RUN python -m pip install pytest
