FROM nvidia/cuda:10.0-devel-ubuntu18.04 AS builder
LABEL maintainer="Carlos Osuna <Carlos.Osuna@meteoswiss.ch>"
LABEL description="GTClang build stage"
RUN apt-get update                                                                                                              && \
    apt-get install -y clang-6.0 libclang-6.0-dev git cmake openssh-client python3.7 libboost-all-dev                           && \
    mkdir /usr/local/gtclang_build                                                                                              && \
    chown -R 1001:1001 /usr/local/gtclang_build
ENV PYTHON_DIR $(which python3.7)
USER 1001
RUN cd /usr/local/gtclang_build                                                                                                 && \
    git clone https://github.com/MeteoSwiss-APN/dawn.git                                                                        && \
    cd dawn/gtclang                                                                                                             && \
    sed -i 's/git@github.com:MeteoSwiss-APN\/dawn.git/https:\/\/github.com\/MeteoSwiss-APN\/dawn.git/g' bundle/CMakeLists.txt   && \
    cd bundle/ && mkdir build && cd build && \ 
    cmake -DPYTHON_EXECUTABLE="${PYTHON_DIR}" -DCMAKE_INSTALL_PREFIX=/usr/local/gtclang_build/install ..                        && \
    make -j2

FROM nvidia/cuda:10.0-devel-ubuntu18.04
LABEL maintainer="Carlos Osuna <Carlos.Osuna@meteoswiss.ch>"
LABEL description="GTClang execution stage"
RUN apt-get update                                                                                                              && \
    apt-get install -y clang-6.0                                                                                                 && \
    mkdir /usr/local/gtclang  && mkdir /usr/local/gtclang_build                                                                 && \
    chown -R 1001:1001 /usr/local/gtclang                                                                                       && \
    chown -R 1001:1001 /usr/local/gtclang_build
USER 1001
WORKDIR /usr/local/gtclang
COPY --from=builder /usr/local/gtclang_build/install .
COPY --from=builder /usr/local/gtclang_build/gtclang/bundle/build/dawn-prefix/src/dawn-build/prefix/protobuf/lib/libprotobuf.so .
RUN mkdir -p /usr/local/gtclang_build/gtclang/src /usr/local/gtclang/data
COPY --from=builder /usr/local/gtclang_build/gtclang/src /usr/local/gtclang_build/gtclang/src
ENV LD_LIBRARY_PATH=/usr/local/gtclang

ENTRYPOINT ["/usr/local/gtclang/bin/gtclang"]
