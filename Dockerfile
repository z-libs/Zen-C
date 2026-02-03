from ubuntu:24.04

RUN apt-get update && \
    apt-get install -y build-essential clang gcc git cmake python3 python3-pip python3-pytest nano

COPY . /Zen-C



# Set a safe default shell
SHELL ["/bin/bash", "-l", "-c"]

CMD ["/bin/bash"]