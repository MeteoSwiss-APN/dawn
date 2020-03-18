# IMAGE needs be be set to one of the docker/dawn-env.dockerfile images
ARG IMAGE=gtclang/dawn-env-cuda10.1-ubuntu18.04
FROM $IMAGE
ARG BUILD_TYPE=Release
COPY . /usr/src/dawn
RUN /usr/src/dawn/scripts/build-and-test \
    --dawn-install-dir /usr/local/dawn \
    --parallel $(nproc) \
    --config $BUILD_TYPE \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_PREFIX_PATH=/usr/lib/llvm-9 \
    -DProtobuf_DIR=/usr/local/protobuf/lib/cmake/protobuf \
    -DPROTOBUF_PYTHON_DIR=/usr/local/lib/python3.7/dist-packages \
    -DGridTools_DIR=/usr/local/gridtools/lib/cmake \
    -Datlas_DIR=/usr/local/atlas/lib/cmake/atlas \
    -Deckit_DIR=/usr/local/eckit/lib/cmake/eckit \
    -GNinja
CMD /bin/bash
