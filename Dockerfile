FROM artefact.skao.int/ska-tango-images-pytango-builder:9.3.35

USER root

RUN apt-get update && apt-get install -y cmake

COPY . ./

RUN poetry export --format requirements.txt --output poetry-requirements.txt --without-hashes && \
    pip install -r poetry-requirements.txt && \
    rm poetry-requirements.txt 
