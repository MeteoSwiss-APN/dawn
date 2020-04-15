# IMAGE needs be be set to one of the docker/dawn-env.dockerfile images
ARG IMAGE=gtclang/dawn-env-ubuntu19.10
FROM $IMAGE
ARG BUILD_TYPE=Release
COPY . /usr/src/dawn
RUN /usr/src/dawn/scripts/build-and-test \
    --dawn-install-dir /usr/local/dawn \
    --parallel $(nproc) \
    --docker-env \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE
CMD /bin/bash
